import time
import random
from itertools import combinations
from math import gcd

# EVOLVE-BLOCK-START

def _independence_number(triples, n=16):
    """Fast independence number via bitmask branch-and-bound with good pruning."""
    if not triples:
        return n
    triples = [tuple(t) for t in triples]
    nt = len(triples)
    
    if nt == 0:
        return n
    
    # For each point, list of triple indices containing it
    pt_triples = [[] for _ in range(n)]
    for i, t in enumerate(triples):
        for p in t:
            pt_triples[p].append(i)
    
    # For each triple, its points as a tuple
    triple_pts = triples
    
    best = [0]
    
    # Process points in order 0..15
    # State: pos, selected_mask, triple_counts (how many points of each triple are in selected)
    # But triple_counts is expensive to copy. Use a different approach.
    
    # Alternative: use the "include/exclude" with constraint propagation
    # When we include point p, for each triple containing p, we need at least one of the other two to be excluded.
    # This is a 2-SAT-like constraint. For speed, when including p, we remove all other points in triples with p
    # from consideration. This is safe (may underestimate) but let's do it correctly.
    
    # Correct B&B:
    # avail: set of points that can still be added
    # For each triple, track how many of its points are in 'selected'
    # If a triple has 2 points selected, the third must be excluded
    
    full_mask = (1 << n) - 1
    sel = 0
    avail = full_mask
    triple_sel_count = [0] * nt  # how many points of triple i are in sel
    
    def rec():
        nonlocal sel, avail, triple_sel_count, best
        sel_count = bin(sel).count('1')
        avail_count = bin(avail).count('1')
        if sel_count + avail_count <= best[0]:
            return
        if avail == 0:
            if sel_count > best[0]:
                best[0] = sel_count
            return
        
        # Pick a point to branch on: one that's in the most triples among available
        pt = -1
        max_triples = -1
        temp_avail = avail
        while temp_avail:
            p = (temp_avail & -temp_avail).bit_length() - 1
            if len(pt_triples[p]) > max_triples:
                max_triples = len(pt_triples[p])
                pt = p
            temp_avail &= temp_avail - 1
        
        # Option 1: exclude pt
        old_avail = avail
        avail &= ~(1 << pt)
        rec()
        avail = old_avail
        
        # Option 2: include pt
        # Check if including pt would complete any triple
        can_include = True
        for ti in pt_triples[pt]:
            if triple_sel_count[ti] >= 2:
                can_include = False
                break
        
        if can_include:
            old_sel = sel
            old_avail = avail
            old_counts = triple_sel_count[:]
            
            sel |= (1 << pt)
            avail &= ~(1 << pt)
            
            # For each triple containing pt, increment count
            # If count reaches 2, the remaining point must be excluded
            for ti in pt_triples[pt]:
                triple_sel_count[ti] += 1
                if triple_sel_count[ti] == 2:
                    # Find the point in triple ti that's not in sel and not pt
                    t = triple_pts[ti]
                    for p in t:
                        if p != pt and not (sel & (1 << p)):
                            avail &= ~(1 << p)
                            break
            
            rec()
            
            sel = old_sel
            avail = old_avail
            triple_sel_count = old_counts
    
    rec()
    return best[0]


def _is_linear(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    random.seed(42)
    
    best_system = None
    best_alpha = n + 1
    
    def try_system(system):
        nonlocal best_system, best_alpha
        if not system:
            return
        if not _is_linear(system):
            return
        alpha = _independence_number(system, n)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = [list(t) for t in system]
    
    # Strategy: Greedy with local search
    def greedy_build(seed_offset=0):
        rng = random.Random(12345 + seed_offset)
        system = []
        remaining = set(all_triples)
        
        if not remaining:
            return system
        
        first = rng.choice(list(remaining))
        system.append(list(first))
        remaining.discard(first)
        
        while remaining:
            best_candidate = None
            best_score = -1
            
            covered = set()
            for s in system:
                covered |= set(s)
            
            sample_size = min(20, len(remaining))
            candidates = rng.sample(list(remaining), sample_size)
            
            for t in candidates:
                linear = True
                for s in system:
                    if len(set(t) & set(s)) > 1:
                        linear = False
                        break
                if not linear:
                    continue
                t_set = set(t)
                new_pts = len(t_set - covered)
                shared = sum(len(set(t) & set(s)) for s in system)
                score = new_pts * 4 + shared * 3
                if score > best_score:
                    best_score = score
                    best_candidate = t
            
            if best_candidate is None:
                break
            
            system.append(list(best_candidate))
            remaining.discard(best_candidate)
        
        return system
    
    # Run many greedy builds
    for i in range(300):
        if time.time() > deadline:
            break
        system = greedy_build(i)
        try_system(system)
        if best_alpha <= 7:
            break
    
    # Local search
    if best_system is not None and time.time() < deadline - 20:
        current = [list(t) for t in best_system]
        current_alpha = best_alpha
        
        for iteration in range(3000):
            if time.time() > deadline:
                break
            
            # Try adding a random triple
            t = random.choice(all_triples)
            linear = True
            for s in current:
                if len(set(t) & set(s)) > 1:
                    linear = False
                    break
            if linear:
                new_system = current + [list(t)]
                new_alpha = _independence_number(new_system, n)
                if new_alpha < current_alpha:
                    current = new_system
                    current_alpha = new_alpha
                    best_system = [list(x) for x in current]
                    best_alpha = current_alpha
            
            # Try swapping
            if len(current) >= 2:
                idx = random.randint(0, len(current) - 1)
                removed = current.pop(idx)
                t = random.choice(all_triples)
                linear = True
                for s in current:
                    if len(set(t) & set(s)) > 1:
                        linear = False
                        break
                if linear:
                    new_system = current + [list(t)]
                    new_alpha = _independence_number(new_system, n)
                    if new_alpha < current_alpha:
                        current = new_system
                        current_alpha = new_alpha
                        best_system = [list(x) for x in current]
                        best_alpha = current_alpha
                    else:
                        current.append(removed)
                else:
                    current.append(removed)
    
    if best_system is None:
        best_system = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    return best_system

# EVOLVE-BLOCK-END
