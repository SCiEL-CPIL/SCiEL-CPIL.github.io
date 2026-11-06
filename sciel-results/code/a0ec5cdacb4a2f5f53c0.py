def calc_score(S_set):
    diffs = set()
    # Convert to list for faster iteration? Set iteration is okay.
    # But checking `in S_set` is O(1).
    for x in S_set:
        for y in S_set:
            if x < y:
                d = y - x
                if x + 2 * d in S_set:
                    diffs.add(d)
    return len(diffs)
