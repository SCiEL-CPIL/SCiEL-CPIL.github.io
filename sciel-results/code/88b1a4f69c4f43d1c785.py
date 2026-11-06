    changed = True
    while changed and time.time() - start < limit:
        changed = False
        bad_a = find_violation(S)
        if bad_a is not None:
            S.remove(bad_a)
            S_set.remove(bad_a)
            changed = True
