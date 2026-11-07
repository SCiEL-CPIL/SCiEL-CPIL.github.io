    deadline = time.time() + 55
    ...
    for i, (lo, hi) in enumerate(P['gaps']):
        ...
        for m in range(lo + 1, hi):
            ...
            if not has_small:
                result[i] = m
                break
        
        if time.time() > deadline:
            break
