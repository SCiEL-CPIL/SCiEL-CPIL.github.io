def prune_set_fast(S):
    S = list(S)
    if not S:
        return S
    
    n = len(S)
    # Count how many times each sum is produced
    sum_count = [0] * (2 * N + 2)
    for i in range(n):
        for j in range(i, n):
            sum_count[S[i] + S[j]] += 1
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            elem = S[i]
            # Check if removing elem keeps maximality
            # First, check if elem is "essential" for maximality
            # i.e., is there some x not in S that can only be blocked by sums involving elem?
            
            # Try removing elem
            # Update sum_count: decrement counts for all pairs involving elem
            removed_sums = []
            for j in range(len(S)):
                if j == i:
                    continue
                s = elem + S[j]
                sum_count[s] -= 1
                if sum_count[s] == 0:
                    removed_sums.append(s)
            sum_count[2 * elem] -= 1
            if sum_count[2 * elem] == 0:
                removed_sums.append(2 * elem)
            
            # Now check maximality with the modified sum_count
            # A sum is "present" if sum_count[s] > 0
            S_set = set(S)
            S_set.discard(elem)
            
            is_max = True
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if sum_count[2 * x] > 0:
                    continue
                addable = True
                for a in S_set:
                    if sum_count[x + a] > 0:
                        addable = False
                        break
                if addable:
                    is_max = False
                    break
            
            if is_max:
                # Remove elem
                S.pop(i)
                improved = True
                break
            else:
                # Restore sum_count
                for j in range(len(S)):
                    s = elem + S[j]
                    sum_count[s] += 1
                sum_count[2 * elem] += 1
    return S
