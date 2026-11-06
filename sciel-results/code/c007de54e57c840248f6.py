import time

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(1, N - 3 * d + 1):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    aps_by_index = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            aps_by_index[pos].append(idx)

    def solve(k, time_limit):
        colors = [None] * (N + 1)
        colors[1] = 0  # Symmetry breaking
        
        # We can also try to fix c[2] if k > 1?
        # No, c[2] can be 0 or 1.
        
        def backtrack(x):
            if time.time() > time_limit:
                return None
            
            if x > N:
                return colors[1:].copy()
            
            # Heuristic: Try colors that are less used? Or just 0..k-1
            for c in range(k):
                valid = True
                for ap_idx in aps_by_index[x]:
                    ap = aps[ap_idx]
                    assigned_colors = []
                    all_assigned = True
                    for pos in ap:
                        if pos == x:
                            continue
                        if colors[pos] is None:
                            all_assigned = False
                            break
                        assigned_colors.append(colors[pos])
                    
                    if all_assigned:
                        # Check constraint
                        if len({c} | set(assigned_colors)) < 3:
                            valid = False
                            break
                
                if valid:
                    colors[x] = c
                    res = backtrack(x + 1)
                    if res is not None:
                        return res
                    colors[x] = None
            
            return None
        
        return backtrack(2)

    # Try k=3 for 40s
    sol = solve(3, start_time + 40.0)
    if sol:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]
        
    # Try k=4 for 40s
    sol = solve(4, start_time + 80.0)
    if sol:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]

    # Fallback: Greedy
    colors = []
    for x in range(1, N + 1):
        found = False
        for c in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                i = x - 3 * d
                if i < 1:
                    continue
                j = i + d
                k = i + 2 * d
                if len({colors[i-1], colors[j-1], colors[k-1], c}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(c)
                found = True
                break
        if not found:
            colors.append(0)
            
    return colors
