def _greedy_reduce(comparators):
    comps = comparators[:]
    i = 0
    while i < len(comps):
        candidate = comps[:i] + comps[i+1:]
        if _verify(candidate):
            comps = candidate
        else:
            i += 1
    return comps
