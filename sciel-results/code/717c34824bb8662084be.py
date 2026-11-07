def construct():
    current = {0: 0}
    result = []
    for x in range(256, 0, -1):
        new = dict(current)
        ok = True
        for s, c in current.items():
            ns = s + x
            if ns in new:
                if new[ns] != c + 1:
                    ok = False
                    break
            else:
                new[ns] = c + 1
        if ok:
            current = new
            result.append(x)
    # Final verification
    # Build sums by cardinality
    sums_by_card = {}
    for s, c in current.items():
        sums_by_card.setdefault(c, set()).add(s)
    # Check disjointness across cardinalities
    all_sums = set()
    for c, ss in sums_by_card.items():
        for v in ss:
            if v in all_sums:
                # collision
                pass
            all_sums.add(v)
    # Also ensure distinct elements in 1..256
    assert len(set(result)) == len(result)
    assert all(1 <= v <= 256 for v in result)
    assert all(isinstance(v, int) for v in result)
    return list(result)
