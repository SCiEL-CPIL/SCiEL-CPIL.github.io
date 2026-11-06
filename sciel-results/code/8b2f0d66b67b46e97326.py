import time
import random
import math

def construct():
    t0 = time.time()
    # Safety margin: 90s limit, aim for 80s
    DEADLINE = t0 + 80.0
    
    N = 256
    
    # Precompute conflict masks using Python integers as bitsets
    # conflict_mask[i] is a bitmask of j such that (i+1, j+1) is a conflicting pair
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    def popcount(x):
        return bin(x).count('1')
    
    def get_size(mask):
        return popcount(mask)
    
    def can_add(idx, mask):
        return (conflict_mask[idx] & mask) == 0

    def valid_set(mask):
        indices = [i for i in range(N) if (mask >> i) & 1]
        for ii in range(len(indices)):
            for jj in range(ii+1, len(indices)):
                a = indices[ii]
                b = indices[jj]
                if (conflict_mask[a] >> b) & 1:
                    return False
        return True

    # Get a good initial solution via greedy
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Greedy: start with all odds, then add compatible evens
    mask_odds = 0
    for i in odds_indices:
        mask_odds |= (1 << i)
    
    current_mask = mask_odds
    for i in evens_indices:
        if can_add(i, current_mask):
            current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = get_size(current_mask)
    
    # Improved greedy coloring for upper bound
    def greedy_color_upper_bound(candidates):
        """
        Returns an upper bound on the maximum independent set size within candidates.
        Uses greedy coloring with degree-based ordering.
        """
        if not candidates:
            return 0
        
        # Get vertices
        vertices = []
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
        
        if not vertices:
            return 0
        
        # Compute degrees within candidate set
        deg = {}
        for v in vertices:
            deg[v] = popcount(conflict_mask[v] & candidates)
        
        # Sort by degree descending
        vertices.sort(key=lambda v: -deg[v])
        
        # Greedy coloring
        color_of = {}
        max_color = 0
        
        for v in vertices:
            # Find colors used by neighbors of v in candidates
            used_colors = set()
            nbr_mask = conflict_mask[v] & candidates
            while nbr_mask:
                lsb = nbr_mask & (-nbr_mask)
                u = lsb.bit_length() - 1
                if u in color_of:
                    used_colors.add(color_of[u])
                nbr_mask ^= lsb
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            max_color = max(max_color, c + 1)
        
        return max_color
    
    # Local search: try to improve by swaps and additions
    def local_search(mask, deadline):
        """
        Local search with swap moves (remove one, add one) and additions.
        Returns improved mask.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = [i for i in range(N) if (mask >> i) & 1]
            
            # Try single additions
            for i in range(N):
                if not ((mask >> i) & 1) and can_add(i, mask):
                    mask |= (1 << i)
                    improved = True
            
            # Try swap moves: remove one element, try to add two
            for i in indices:
                if time.time() > deadline:
                    break
                temp_mask = mask & ~(1 << i)
                added = 0
                for j in range(N):
                    if not ((temp_mask >> j) & 1) and can_add(j, temp_mask):
                        temp_mask |= (1 << j)
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    mask = temp_mask
                    improved = True
            
            # Try multi-element exchange: remove up to 2, add up to 3
            if not improved:
                for i in indices:
                    if time.time() > deadline:
                        break
                    for k in indices:
                        if k <= i:
                            continue
                        if time.time() > deadline:
                            break
                        temp_mask = mask & ~(1 << i) & ~(1 << k)
                        added = 0
                        for j in range(N):
                            if not ((temp_mask >> j) & 1) and can_add(j, temp_mask):
                                temp_mask |= (1 << j)
                                added += 1
                                if added >= 3:
                                    break
                        if added >= 3:
                            mask = temp_mask
                            improved = True
        
        return mask
    
    # Run local search first
    deadline_ls = DEADLINE - 30.0
    best_mask = local_search(best_mask, deadline_ls)
    best_size = get_size(best_mask)
    
    # Branch and bound with aggressive pruning
    def bb_mis(candidates, current_best_size, deadline):
        """
        Branch and bound to find MIS in candidates, trying to beat current_best_size.
        Returns best independent set found.
        """
        if time.time() > deadline:
            return 0
        
        if not candidates:
            return 0
        
        # Upper bound via greedy coloring
        ub = greedy_color_upper_bound(candidates)
        
        if ub <= current_best_size:
            return 0
        
        # If small enough, solve exactly
        cand_size = popcount(candidates)
        if cand_size <= 25:
            # Use recursive exact MIS for small sets
            def small_mis(cands):
                if not cands:
                    return 0
                v = (cands & (-cands)).bit_length() - 1
                rem1 = cands & ~(conflict_mask[v] | (1 << v))
                sol1 = (1 << v) | small_mis(rem1)
                rem2 = cands & ~(1 << v)
                sol2 = small_mis(rem2)
                if popcount(sol1) >= popcount(sol2):
                    return sol1
                else:
                    return sol2
            
            return small_mis(candidates)
        
        # Select vertex to branch on: highest degree
        vertices = []
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
        
        if not vertices:
            return 0
        
        # Pick vertex with max degree
        deg = {}
        for v in vertices:
            deg[v] = popcount(conflict_mask[v] & candidates)
        
        v = max(vertices, key=lambda x: deg[x])
        
        # Branch 1: Include v
        include_mask = 1 << v
        remaining = candidates & ~(conflict_mask[v] | (1 << v))
        
        sub_best = bb_mis(remaining, current_best_size - 1, deadline)
        candidate1 = include_mask | sub_best
        size1 = get_size(candidate1)
        
        if size1 > current_best_size:
            current_best_size = size1
        
        # Branch 2: Exclude v
        remaining2 = candidates & ~(1 << v)
        sub_best2 = bb_mis(remaining2, current_best_size, deadline)
        candidate2 = sub_best2
        size2 = get_size(candidate2)
        
        if size1 >= size2:
            return candidate1
        else:
            return candidate2
    
    # Run B&B to try to improve
    deadline_bb = DEADLINE - 15.0
    full_candidates = (1 << N) - 1
    bb_result = bb_mis(full_candidates, best_size, deadline_bb)
    
    if get_size(bb_result) > best_size:
        best_mask = bb_result
        best_size = get_size(best_mask)
    
    # Final local search polish
    deadline_final = DEADLINE - 5.0
    best_mask = local_search(best_mask, deadline_final)
    best_size = get_size(best_mask)
    
    # Final polish: try to add any node that fits
    for i in range(N):
        if not ((best_mask >> i) & 1) and can_add(i, best_mask):
            best_mask |= (1 << i)
    
    if not valid_set(best_mask):
        # Fallback to all odds
        best_mask = 0
        for i in odds_indices:
            best_mask |= (1 << i)
    
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    return result
