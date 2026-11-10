def construct():
    import time
    start = time.time()
    deadline = start + 85.0  # 85s safety margin before 90s kill
    
    N = 512
    from math import gcd
    
    # Build adjacency using bitsets (Python integers as bitmasks)
    # adj[i] is a bitmask where bit j is set if lcm(i+1, j+1) <= 512
    # We use 0-indexed internally: vertex i represents number i+1
    
    n = N
    adj = [0] * n  # adj[i] bitmask of compatible vertices (excluding self)
    
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            g = gcd(a, b)
            l = (a // g) * b
            if l <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    if time.time() > deadline:
        return [1]
    
    # Tomita's maximum clique algorithm with greedy coloring
    # Uses bitmasks for efficiency
    
    best_clique = []
    best_size = 0
    timeout_flag = [False]
    
    def color_sort(P_mask):
        """
        Greedily color P_mask and return (ordered_list, color_numbers).
        ordered_list: vertices sorted by color (highest first)
        color_numbers: color assigned to each vertex in ordered_list
        """
        if P_mask == 0:
            return [], []
        
        # Get list of vertices in P
        vertices = []
        m = P_mask
        while m:
            lowbit = m & (-m)
            v = lowbit.bit_length() - 1
            vertices.append(v)
            m ^= lowbit
        
        # Order by degree in P (descending) for better coloring
        deg_in_p = []
        for v in vertices:
            deg_in_p.append((adj[v] & P_mask).bit_count())
        
        # Sort vertices by degree descending
        sorted_verts = [v for _, v in sorted(zip(deg_in_p, vertices), reverse=True)]
        
        # Greedy coloring: assign smallest color not used by neighbors already in same color class
        # Color classes: list of bitmasks
        color_classes = []  # list of (color_mask, color_num)
        color_of = {}
        
        for v in sorted_verts:
            placed = False
            for ci, (cmask, cnum) in enumerate(color_classes):
                # v can go in class ci if v is not adjacent to any vertex in cmask
                # i.e., (adj[v] & cmask) == 0
                if (adj[v] & cmask) == 0:
                    color_classes[ci] = (cmask | (1 << v), cnum)
                    color_of[v] = cnum
                    placed = True
                    break
            if not placed:
                cnum = len(color_classes) + 1
                color_classes.append((1 << v, cnum))
                color_of[v] = cnum
        
        # Order by color descending (higher color = better bound)
        # Within same color, keep degree order
        ordered = sorted(vertices, key=lambda v: (-color_of[v], -deg_in_p[vertices.index(v)]))
        # Actually simpler: sort by color descending
        ordered = sorted(vertices, key=lambda v: -color_of[v])
        color_nums = [color_of[v] for v in ordered]
        
        return ordered, color_nums
    
    def expand(R_mask, P_mask):
        nonlocal best_clique, best_size, timeout_flag
        
        if timeout_flag[0]:
            return
        
        if time.time() > deadline:
            timeout_flag[0] = True
            return
        
        # Update best
        r_size = R_mask.bit_count()
        if r_size > best_size:
            best_size = r_size
            # Extract vertices from R_mask
            verts = []
            m = R_mask
            while m:
                lowbit = m & (-m)
                verts.append(lowbit.bit_length() - 1)
                m ^= lowbit
            best_clique = verts
        
        if P_mask == 0:
            return
        
        # Color sort P
        ordered, color_nums = color_sort(P_mask)
        
        if not ordered:
            return
        
        max_color = color_nums[0] if color_nums else 0
        
        # Bound: if current size + max_color <= best, prune
        if r_size + max_color <= best_size:
            return
        
        # Process vertices in reverse order (highest color first)
        for idx in range(len(ordered) - 1, -1, -1):
            if timeout_flag[0]:
                return
            
            v = ordered[idx]
            c = color_nums[idx]
            
            # Bound: if r_size + c <= best_size, stop
            if r_size + c <= best_size:
                break
            
            # v must still be in P
            if not (P_mask & (1 << v)):
                continue
            
            # Recurse
            new_R = R_mask | (1 << v)
            new_P = P_mask & adj[v]
            # Remove v from P for subsequent iterations
            P_mask &= ~(1 << v)
            
            expand(new_R, new_P)
    
    # Start with all vertices
    all_mask = (1 << n) - 1
    expand(0, all_mask)
    
    if timeout_flag[0] or time.time() > deadline - 2:
        # Return best found
        if best_clique:
            result = sorted([v + 1 for v in best_clique])
        else:
            result = [1]
    else:
        # Try to improve with local search
        selected_set = set(best_clique) if best_clique else set()
        
        # Try adding unselected vertices
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for u in range(n):
                if time.time() > deadline - 3:
                    break
                if u in selected_set:
                    continue
                # Check if u can be added
                can_add = True
                for s in selected_set:
                    if not (adj[u] & (1 << s)):
                        can_add = False
                        break
                if can_add:
                    selected_set.add(u)
                    improved = True
        
        # Try 2-for-1 swaps: remove 1 selected, add 2 unselected
        if time.time() < deadline - 5:
            unselected = [i for i in range(n) if i not in selected_set]
            improved2 = True
            while improved2 and time.time() < deadline - 5:
                improved2 = False
                for i in range(len(unselected)):
                    if time.time() > deadline - 5:
                        break
                    u1 = unselected[i]
                    for j in range(i + 1, len(unselected)):
                        if time.time() > deadline - 5:
                            break
                        u2 = unselected[j]
                        # Check u1 and u2 are compatible
                        if not (adj[u1] & (1 << u2)):
                            continue
                        # Find blockers
                        blockers = set()
                        for s in selected_set:
                            if not (adj[u1] & (1 << s)) or not (adj[u2] & (1 << s)):
                                blockers.add(s)
                        if len(blockers) <= 1:
                            for s in blockers:
                                selected_set.discard(s)
                            selected_set.add(u1)
                            selected_set.add(u2)
                            improved2 = True
                            # Update unselected
                            unselected = [x for x in unselected if x != u1 and x != u2]
                            break
                    if improved2:
                        break
        
        best_clique = list(selected_set)
        result = sorted([v + 1 for v in best_clique])
    
    if not result:
        result = [1]
    
    # Final verification
    from math import gcd as gcd2
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            g = gcd2(a, b)
            l = (a // g) * b
            if l > 512:
                # Invalid, remove the problematic element
                # Just return a safe fallback
                return [1]
    
    return result
