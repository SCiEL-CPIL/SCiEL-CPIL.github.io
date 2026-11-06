def _has_sidon_subset_of_size(arr, target_size):
    """Returns True if arr has a Sidon subset of size target_size."""
    n = len(arr)
    if n < target_size:
        return False
    a_sorted = sorted(arr)
    n = len(a_sorted)
    sums = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sums[i][j] = a_sorted[i] + a_sorted[j]

    found = False

    def dfs(start, chosen, used_sums):
        nonlocal found
        if found:
            return
        if len(chosen) == target_size:
            found = True
            return
        if len(chosen) + (n - start) < target_size:
            return
        for k in range(start, n):
            ok = True
            new_sums = []
            for ci in chosen:
                s = sums[ci][k] if ci <= k else sums[k][ci]
                if s in used_sums:
                    ok = False
                    break
                new_sums.append(s)
            if not ok:
                continue
            s_diag = sums[k][k]
            if s_diag in used_sums:
                continue
            new_sums.append(s_diag)
            
            chosen.append(k)
            for s in new_sums:
                used_sums.add(s)
            dfs(k+1, chosen, used_sums)
            for s in new_sums:
                used_sums.discard(s)
            chosen.pop()
            if found:
                return

    dfs(0, [], set())
    return found
