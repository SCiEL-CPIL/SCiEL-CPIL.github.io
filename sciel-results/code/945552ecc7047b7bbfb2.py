import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    aps = []
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
            
    index_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            index_to_aps[pos].append(idx)

    def backtrack(k, pos, coloring):
        if time.time() > deadline:
            return False
            
        if pos == N:
            for ap in aps:
                a, b, c, d = ap
                if len({coloring[a], coloring[b], coloring[c], coloring[d]}) < 3:
                    return False
            return True

        if pos == 0:
            candidates = [0]
        elif pos == 1:
            candidates = [1] if k > 1 else [0]
        elif pos == 2:
            if k > 2:
                candidates = [2]
            else:
                candidates = [0, 1]
        else:
            candidates = list(range(k))

        for color in candidates:
            coloring[pos] = color
            valid = True
            for ap_idx in index_to_aps[pos]:
                a, b, c, d = aps[ap_idx]
                ca, cb, cc, cd = coloring[a], coloring[b], coloring[c], coloring[d]
                
                # Identify which position is 'pos'
                # We don't strictly need to know which one is pos for the set logic,
                # but we need to know which are assigned.
                
                vals = [ca, cb, cc, cd]
                assigned = [v for v in vals if v != -1]
                unassigned_count = 4 - len(assigned)
                distinct_assigned = set(assigned)
                
                if unassigned_count == 0:
                    if len(distinct_assigned) < 3:
                        valid = False
                        break
                elif unassigned_count == 1:
                    # The last one is forced or restricted.
                    # If distinct_assigned has size 3, no restriction.
                    # If size 2, say {A, B}, the last must be C not in {A, B}.
                    # If size 1, say {A}, the last must be B not in {A}.
                    # But we are checking if the CURRENT assignment (which might be the last one) is valid.
                    # If 'pos' is the unassigned one, then 'color' is the last one.
                    # We just check if the resulting set has size >= 3.
                    # Since we set coloring[pos]=color, 'vals' now has all 4.
                    # Wait, if unassigned_count == 1 BEFORE assignment, then after assignment it's 0.
                    # But we are inside the loop where we just assigned 'pos'.
                    # So if 'pos' was the only unassigned, now unassigned_count is 0.
                    # The check `if unassigned_count == 0` above handles this IF we recalculated.
                    # But `vals` was calculated AFTER assignment.
                    # So if 'pos' was the last unassigned, `unassigned_count` is 0.
                    # So the `unassigned_count == 0` block handles it.
                    # What if 'pos' is NOT the last unassigned?
                    # Then `unassigned_count` >= 1.
                    # In that case, we just need to ensure it's POSSIBLE.
                    pass
                elif unassigned_count == 2:
                    # If distinct_assigned has size 1 (A, A), the two unassigned must be distinct and != A.
                    # If distinct_assigned has size 2 (A, B), the two unassigned must allow reaching size 3.
                    # If we have A, B, and two unassigned u1, u2.
                    # Final set {A, B, u1, u2}. Size >= 3.
                    # This is always possible unless k=2? No, k>=3.
                    # If k=3, and we have A, B, we can pick C, C? Set {A, B, C}. Size 3. OK.
                    # So no immediate contradiction for k>=3.
                    pass
                elif unassigned_count == 3:
                    # 1 assigned. Always possible for k>=3.
                    pass

            if valid:
                if backtrack(k, pos + 1, coloring):
                    return True
            
            coloring[pos] = -1
            
        return False

    for k in range(3, 129):
        coloring = [-1] * N
        if backtrack(k, 0, coloring):
            return coloring

    # Fallback: Seed logic
    colors = []
    for x in range(1, N + 1):
        for color in range(N):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
