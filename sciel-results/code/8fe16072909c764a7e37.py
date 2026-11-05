import sys

def construct(q, n, radius):
    if not (q == 3 and n == 7 and radius == 1):
        # Fallback for other cases (not expected in this specific task optimization)
        # Simple greedy
        total = q ** n
        def idx_to_list(idx):
            res = []
            for _ in range(n):
                res.append(idx % q)
                idx //= q
            return res
        
        # Precompute neighbors for greedy
        def get_cover(idx):
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % q)
                tmp //= q
            mask = 1 << idx
            for k in range(n):
                orig = digits[k]
                for s in range(q):
                    if s == orig: continue
                    new_d = digits[:]
                    new_d[k] = s
                    j = 0
                    mult = 1
                    for val in new_d:
                        j += val * mult
                        mult *= q
                    mask |= (1 << j)
            return mask

        cover_masks = [get_cover(i) for i in range(total)]
        uncovered = (1 << total) - 1
        code = []
        while uncovered:
            best_c = -1
            best_count = -1
            for c in range(total):
                cnt = (uncovered & cover_masks[c]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            code.append(best_c)
            uncovered &= ~cover_masks[best_c]
        return [idx_to_list(i) for i in code]

    # Specific solver for q=3, n=7, radius=1
    total = 2187
    sys.setrecursionlimit(100000)
    
    # Precomputation
    # 1. Convert indices to digits for neighbor calculation
    all_digits = []
    for i in range(total):
        d = []
        tmp = i
        for _ in range(7):
            d.append(tmp % 3)
            tmp //= 3
        all_digits.append(d)

    # 2. Precompute cover masks (radius 1) and conflict masks (radius 2)
    # cover_masks[i]: bitmask of points within dist 1 of i
    # conflict_masks[i]: bitmask of points within dist 2 of i
    cover_masks = [0] * total
    conflict_masks = [0] * total
    
    # Helper to compute index from digits
    def digits_to_idx(digits):
        idx = 0
        mult = 1
        for val in digits:
            idx += val * mult
            mult *= 3
        return idx

    for i in range(total):
        d_i = all_digits[i]
        mask1 = 1 << i
        mask2 = 1 << i
        
        # Radius 1 neighbors
        for k in range(7):
            orig = d_i[k]
            for s in range(3):
                if s == orig: continue
                new_d = d_i[:]
                new_d[k] = s
                j = digits_to_idx(new_d)
                mask1 |= (1 << j)
                mask2 |= (1 << j)
                
                # Radius 2 neighbors: change another position or same position to other value?
                # Distance 2 means exactly 2 positions differ.
                # We can generate them by taking the radius 1 neighbors and adding one more change,
                # or directly iterating pairs.
        
        # To get mask2 (dist <= 2), we can take mask1 and for each point in mask1, add its dist 1 neighbors?
        # That would be dist <= 2.
        # But it's more efficient to generate dist 2 directly or combine.
        # Let's generate dist 2 explicitly to be safe and fast.
        
        # Dist 2: choose 2 positions k1, k2. Change d[k1] to s1 != d[k1], d[k2] to s2 != d[k2].
        for k1 in range(7):
            for k2 in range(k1 + 1, 7):
                orig1 = d_i[k1]
                orig2 = d_i[k2]
                for s1 in range(3):
                    if s1 == orig1: continue
                    for s2 in range(3):
                        if s2 == orig2: continue
                        new_d = d_i[:]
                        new_d[k1] = s1
                        new_d[k2] = s2
                        j = digits_to_idx(new_d)
                        mask2 |= (1 << j)
                        
        cover_masks[i] = mask1
        conflict_masks[i] = mask2

    # Greedy Initial Upper Bound
    def greedy_solve():
        uncovered = (1 << total) - 1
        code = []
        while uncovered:
            best_c = -1
            best_count = -1
            for c in range(total):
                cnt = (uncovered & cover_masks[c]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            code.append(best_c)
            uncovered &= ~cover_masks[best_c]
        return code

    initial_code = greedy_solve()
    best_size = len(initial_code)
    best_code = initial_code[:]

    # B&B Solver
    nodes_visited = 0
    node_limit = 200000 # Adjust based on performance needs
    
    def packing_lb(mask):
        lb = 0
        m = mask
        while m:
            # Heuristic: pick the point that has the smallest conflict neighborhood within m?
            # For speed, let's just pick the first one. 
            # Or, to tighten, pick the one with min (m & conflict_masks[idx]).bit_count()
            # This loop is O(n) per pick, and picks are ~100. Total ~100*2187 ops. Acceptable.
            # But doing this for EVERY node in B&B might be too slow.
            # Let's use a faster LB: just pick first set bit.
            lsb = m & -m
            idx = lsb.bit_length() - 1
            lb += 1
            m &= ~conflict_masks[idx]
        return lb

    def solve(mask, depth, current_code):
        nonlocal best_size, best_code, nodes_visited
        
        if nodes_visited > node_limit:
            return
        
        nodes_visited += 1
        
        if mask == 0:
            if depth < best_size:
                best_size = depth
                best_code = current_code[:]
            return
        
        lb = packing_lb(mask)
        if depth + lb >= best_size:
            return
            
        # Branching
        # Find the first uncovered point
        lsb = mask & -mask
        pivot_idx = lsb.bit_length() - 1
        
        # Candidates are codewords c such that pivot_idx is in cover_masks[c]
        # i.e., d(c, pivot_idx) <= 1
        candidates = []
        for c in range(total):
            if (cover_masks[c] & lsb): # Check if pivot is covered by c
                # Calculate how many NEW points this candidate covers
                new_cover = (mask & cover_masks[c]).bit_count()
                candidates.append((new_cover, c))
                
        # Sort by coverage (descending) to find good solutions faster
        candidates.sort(reverse=True)
        
        for _, c in candidates:
            new_mask = mask & ~cover_masks[c]
            current_code.append(c)
            solve(new_mask, depth + 1, current_code)
            current_code.pop()
            
    full_mask = (1 << total) - 1
    solve(full_mask, 0, [])

    # Convert result to lists
    def idx_to_list(idx):
        res = []
        for _ in range(7):
            res.append(idx % 3)
            idx //= 3
        return res
            
    return [idx_to_list(i) for i in best_code]
