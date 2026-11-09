"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _independence_number(triples):
    """Compute the independence number using branch-and-bound bitmask search."""
    n = 16
    if not triples:
        return n
    
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # For each point, which triples contain it
    point_to_triples = [[] for _ in range(n)]
    for ti, tm in enumerate(triple_masks):
        for p in range(n):
            if tm & (1 << p):
                point_to_triples[p].append(ti)
    
    best = [0]
    
    def _check_independent(mask):
        """Check if mask is independent (contains no full triple)."""
        for tm in triple_masks:
            if (mask & tm) == tm:
                return False
        return True
    
    def _branch_and_bound(remaining_points, current_mask, current_size):
        if not remaining_points:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + popcount(remaining_points)
        if current_size + bin(remaining_points).count('1') <= best[0]:
            return
        
        # Pick a point with the most constraints (branch on it)
        # Try including the first point
        # Greedy: pick point with most triples in remaining
        best_pt = -1
        best_cnt = -1
        temp = remaining_points
        while temp:
            lsb = temp & (-temp)
            p = lsb.bit_length() - 1
            # Count how many triples are "threatened" if we add p
            # Actually, just pick first point for simplicity, or use heuristic
            best_pt = p
            break
        temp = 0
        # Better heuristic: pick point that appears in most triples within remaining
        temp = remaining_points
        while temp:
            lsb = temp & (-temp)
            p = lsb.bit_length() - 1
            cnt = len(point_to_triples[p])
            if cnt > best_cnt:
                best_cnt = cnt
                best_pt = p
            temp ^= lsb
        
        if best_pt == -1:
            return
        
        p_bit = 1 << best_pt
        
        # Option 1: Include best_pt
        new_mask = current_mask | p_bit
        # Check if adding best_pt completes any triple
        valid = True
        for ti in point_to_triples[best_pt]:
            tm = triple_masks[ti]
            if (new_mask & tm) == tm:
                valid = False
                break
        if valid:
            new_remaining = remaining_points & ~p_bit
            _branch_and_bound(new_remaining, new_mask, current_size + 1)
        
        # Option 2: Exclude best_pt
        _branch_and_bound(remaining_points & ~p_bit, current_mask, current_size)
    
    _branch_and_bound((1 << n) - 1, 0, 0)
    return best[0]

def _is_valid(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i+1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Algebraic design using GF(4)^2 representation
    gf4_add = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    gf4_mul = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
    ]
    
    def point_id(x, y):
        return x * 4 + y
    
    triples = []
    seen = set()
    
    for px in range(4):
        for py in range(4):
            p = point_id(px, py)
            for dx in range(4):
                for dy in range(4):
                    if dx == 0 and dy == 0:
                        continue
                    q1x = gf4_add[px][dx]
                    q1y = gf4_add[py][dy]
                    q1 = point_id(q1x, q1y)
                    
                    adx = gf4_mul[2][dx]
                    ady = gf4_mul[2][dy]
                    
                    q2x = gf4_add[px][adx]
                    q2y = gf4_add[py][ady]
                    q2 = point_id(q2x, q2y)
                    
                    t = tuple(sorted([p, q1, q2]))
                    if t not in seen:
                        valid = True
                        for existing in triples:
                            common = set(t) & set(existing)
                            if len(common) > 1:
                                valid = False
                                break
                        if valid:
                            triples.append(list(t))
                            seen.add(t)
    
    if not _is_valid(triples):
        triples = []
        all_triples = list(combinations(range(16), 3))
        random.shuffle(all_triples)
        for t in all_triples:
            t_set = set(t)
            valid = True
            for existing in triples:
                if len(t_set & set(existing)) > 1:
                    valid = False
                    break
            if valid:
                triples.append(list(t))
    
    current_score = _independence_number(triples)
    
    # Local search: try swap moves (replace one triple with another)
    all_triples_list = list(combinations(range(16), 3))
    
    # Precompute which triples are compatible with each existing triple
    # For speed, we'll do a simple swap-based local search
    
    def _can_add(triple_set, triples_list):
        for existing in triples_list:
            if len(triple_set & set(existing)) > 1:
                return False
        return True
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Try swap moves: remove one triple, add a different one
        for _ in range(2000):
            if time.time() >= deadline - 5.0:
                break
            
            # Pick a random triple to remove
            idx = random.randint(0, len(triples) - 1)
            removed = triples[idx]
            remaining = [t for i, t in enumerate(triples) if i != idx]
            
            # Try to find a replacement triple
            # Prefer triples that "hit" many of the large independent sets
            candidate = None
            for _ in range(50):
                cand = set(random.choice(all_triples_list))
                if _can_add(cand, remaining):
                    candidate = list(cand)
                    break
            
            if candidate is not None:
                new_triples = remaining + [candidate]
                new_score = _independence_number(new_triples)
                if new_score < current_score:
                    triples = new_triples
                    current_score = new_score
                    improved = True
                    break
        
        # Also try adding a triple if possible
        if not improved:
            for _ in range(500):
                if time.time() >= deadline - 5.0:
                    break
                cand = set(random.choice(all_triples_list))
                if _can_add(cand, triples):
                    new_triples = triples + [list(cand)]
                    new_score = _independence_number(new_triples)
                    if new_score < current_score:
                        triples = new_triples
                        current_score = new_score
                        improved = True
                        break
        
        if not improved:
            # Try more aggressive: remove two triples, add two
            found = False
            for _ in range(1000):
                if time.time() >= deadline - 5.0:
                    break
                i1 = random.randint(0, len(triples) - 1)
                i2 = random.randint(0, len(triples) - 1)
                if i1 == i2:
                    continue
                remaining = [t for i, t in enumerate(triples) if i != i1 and i != i2]
                
                # Try to add two compatible triples
                added = []
                for _ in range(20):
                    cand = set(random.choice(all_triples_list))
                    if _can_add(cand, remaining + added):
                        added.append(list(cand))
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    new_triples = remaining + added
                    new_score = _independence_number(new_triples)
                    if new_score < current_score:
                        triples = new_triples
                        current_score = new_score
                        found = True
                        break
            
            if not found:
                break
    
    return triples
# EVOLVE-BLOCK-END
