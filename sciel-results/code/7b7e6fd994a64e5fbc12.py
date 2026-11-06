import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            lcm_table[i][j] = i // gcd(i, j) * j
    
    # Precompute bad triples and conflict pairs
    conflict_pairs = [set() for _ in range(n + 1)]
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    bad_triples.append((a, b, c))
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))
    
    # Convert to bit-pair lists for fast checking
    conflict_bits = [None] * (n + 1)
    for x in range(1, n + 1):
        conflict_bits[x] = [(1 << (u - 1), 1 << (v - 1)) for u, v in conflict_pairs[x]]
    
    def can_add(x, mask):
        """Check if element x can be added to the set represented by mask."""
        for bu, bv in conflict_bits[x]:
            if (mask & bu) and (mask & bv):
                return False
        return True
    
    def popcount(mask):
        return mask.bit_count()
    
    def greedy_fill(mask, order=None):
        """Greedily add valid elements in given order (or natural order if None)."""
        if order is None:
            order = list(range(1, n + 1))
        selected = mask
        for x in order:
            if not (selected >> (x - 1)) & 1:
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected
    
    def structured_local_search(mask, time_limit=2.0):
        """
        Structured local search with 1-for-2 and 2-for-3 moves.
        Returns the best mask found.
        """
        best_mask = mask
        best_size = popcount(mask)
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            improved = False
            
            # Phase 1: Greedy add pass
            for x in range(1, n + 1):
                if not (mask >> (x - 1)) & 1:
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
            
            sz = popcount(mask)
            if sz > best_size:
                best_size = sz
                best_mask = mask
            
            if not improved:
                # Phase 2: Try 1-for-2 moves (remove u, add v, w)
                s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
                found_move = False
                for u in s_list:
                    if time.time() > end_time:
                        break
                    mask_no_u = mask & ~(1 << (u - 1))
                    # Find all candidates that can be added to mask_no_u
                    candidates = []
                    for v in range(1, n + 1):
                        if not (mask_no_u >> (v - 1)) & 1:
                            if can_add(v, mask_no_u):
                                candidates.append(v)
                    
                    if len(candidates) >= 2:
                        # Try to add two candidates that are compatible
                        added = 0
                        new_mask = mask_no_u
                        for v in candidates:
                            if can_add(v, new_mask):
                                new_mask |= (1 << (v - 1))
                                added += 1
                                if added == 2:
                                    break
                        
                        if added == 2:
                            new_sz = popcount(new_mask)
                            if new_sz > best_size:
                                best_size = new_sz
                                best_mask = new_mask
                                mask = new_mask
                                improved = True
                                found_move = True
                                break
                if found_move:
                    continue
                
                # Phase 3: Try 2-for-3 moves (remove u, v, add w, x, y)
                s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
                found_move = False
                for i in range(len(s_list)):
                    if time.time() > end_time:
                        break
                    for j in range(i + 1, len(s_list)):
                        if time.time() > end_time:
                            break
                        u = s_list[i]
                        v = s_list[j]
                        mask_no_uv = mask & ~(1 << (u - 1)) & ~(1 << (v - 1))
                        
                        # Find all candidates that can be added to mask_no_uv
                        candidates = []
                        for w in range(1, n + 1):
                            if not (mask_no_uv >> (w - 1)) & 1:
                                if can_add(w, mask_no_uv):
                                    candidates.append(w)
                        
                        if len(candidates) >= 3:
                            added = 0
                            new_mask = mask_no_uv
                            for w in candidates:
                                if can_add(w, new_mask):
                                    new_mask |= (1 << (w - 1))
                                    added += 1
                                    if added == 3:
                                        break
                            
                            if added == 3:
                                new_sz = popcount(new_mask)
                                if new_sz > best_size:
                                    best_size = new_sz
                                    best_mask = new_mask
                                    mask = new_mask
                                    improved = True
                                    found_move = True
                                    break
                    if found_move:
                        break
                if found_move:
                    continue
                
                # Phase 4: Random perturbation if stuck
                if not s_list:
                    break
                
                # Try random removals and refill
                for _ in range(20):
                    if time.time() > end_time:
                        break
                    if random.random() < 0.5:
                        x = random.choice(s_list)
                        mask_trial = mask & ~(1 << (x - 1))
                    else:
                        if len(s_list) < 2:
                            continue
                        x1, x2 = random.sample(s_list, 2)
                        mask_trial = mask & ~(1 << (x1 - 1)) & ~(1 << (x2 - 1))
                    
                    # Greedy fill in random order
                    fill_order = list(range(1, n + 1))
                    random.shuffle(fill_order)
                    new_mask = greedy_fill(mask_trial, fill_order)
                    
                    new_sz = popcount(new_mask)
                    if new_sz > best_size:
                        best_size = new_sz
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
                        break
        
        return best_mask
    
    def anchor_based_search(mask, time_limit=3.0):
        """
        Anchor-based search: fix a core of elements, then optimize around them.
        """
        best_mask = mask
        best_size = popcount(mask)
        end_time = time.time() + time_limit
        
        # Identify "anchor" elements: those that appear in many bad triples
        degrees = [len(conflict_pairs[i]) for i in range(1, n + 1)]
        
        # Try different anchor sets
        anchor_sizes = [1, 2, 3]
        for anchor_size in anchor_sizes:
            if time.time() > end_time:
                break
            
            # Select top-degree elements as anchors
            sorted_by_degree = sorted(range(1, n + 1), key=lambda x: -degrees[x])
            
            for anchor_start in range(0, min(len(sorted_by_degree), 20), anchor_size):
                if time.time() > end_time:
                    break
                
                anchors = sorted_by_degree[anchor_start:anchor_start + anchor_size]
                anchor_mask = 0
                for a in anchors:
                    anchor_mask |= (1 << (a - 1))
                
                # Check if anchors are mutually compatible
                valid_anchors = True
                for i in range(len(anchors)):
                    for j in range(i + 1, len(anchors)):
                        a, b = anchors[i], anchors[j]
                        # Check if adding b to set containing a creates a bad triple
                        # This requires checking if there's a c such that (a,b,c) is bad
                        # For now, just greedily build around anchors
                        pass
                
                # Greedily fill around anchors
                fill_mask = anchor_mask
                for x in range(1, n + 1):
                    if not (fill_mask >> (x - 1)) & 1:
                        if can_add(x, fill_mask):
                            fill_mask |= (1 << (x - 1))
                
                # Apply structured local search
                result = structured_local_search(fill_mask, time_limit=0.5)
                sz = popcount(result)
                if sz > best_size:
                    best_size = sz
                    best_mask = result
        
        return best_mask
    
    best_mask = 0
    best_size = 0
    
    # Phase 1: Multi-start with diverse orders
    num_restarts = 0
    while time.time() < deadline - 8:
        num_restarts += 1
        
        r = random.random()
        if r < 0.3:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif r < 0.5:
            order = sorted(range(1, n + 1), key=lambda x: len(conflict_pairs[x]))
        elif r < 0.7:
            order = sorted(range(1, n + 1), key=lambda x: -len(conflict_pairs[x]))
        elif r < 0.85:
            # Weighted by inverse degree
            weights = [1.0 / max(1, len(conflict_pairs[x])) for x in range(1, n + 1)]
            order = list(range(1, n + 1))
            random.shuffle(order)
            order = sorted(order, key=lambda x: random.random() * len(conflict_pairs[x]))
        else:
            # Prime-priority
            def is_prime(k):
                if k < 2:
                    return False
                for d in range(2, int(k**0.5) + 1):
                    if k % d == 0:
                        return False
                return True
            order = sorted(range(1, n + 1), key=lambda x: (0 if is_prime(x) else 1, x))
        
        mask = greedy_fill(0, order)
        mask = structured_local_search(mask, time_limit=1.5)
        
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Phase 2: Anchor-based search on best solution
    if time.time() < deadline - 5:
        best_mask = anchor_based_search(best_mask, time_limit=3.0)
        best_size = popcount(best_mask)
    
    # Phase 3: Final structured local search with more time
    if time.time() < deadline - 2:
        best_mask = structured_local_search(best_mask, time_limit=3.0)
        best_size = popcount(best_mask)
    
    # Final greedy polish
    for _ in range(3):
        changed = False
        for x in range(1, n + 1):
            if not (best_mask >> (x - 1)) & 1:
                if can_add(x, best_mask):
                    best_mask |= (1 << (x - 1))
                    best_size += 1
                    changed = True
        if not changed:
            break
    
    result = [i for i in range(1, n + 1) if (best_mask >> (i - 1)) & 1]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    print(res)
