def _count_progression_diffs_fast(S_arr):
    S_list = [i for i in range(1, 513) if S_arr[i]]
    S_list.sort()
    diffs = set()
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            d = b - a
            # a + 2d = 2b - a
            if S_arr[2 * b - a]:
                diffs.add(d)
    return len(diffs)
