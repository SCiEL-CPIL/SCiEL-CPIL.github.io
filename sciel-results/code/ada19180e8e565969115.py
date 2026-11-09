<<<<<<< SEARCH
    # Greedy deletion: repeatedly try to remove a comparator if still valid.
    cur = list(best)
    # Try each position; iterate to a fixpoint.
    changed = True
    while changed:
        changed = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > TIME_LIMIT:
                break
            cand = cur[:idx] + cur[idx + 1:]
            if _verify(cand):
                cur = cand
                changed = True
        if _elapsed() > TIME_LIMIT:
            break

    if len(cur) < best_len:
        best = cur
        best_len = len(cur)
=======
    # Greedy deletion: repeatedly try to remove a comparator if still valid.
    cur = list(best)
    # Try each position; iterate to a fixpoint.
    changed = True
    while changed:
        changed = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > TIME_LIMIT:
                break
            cand = cur[:idx] + cur[idx + 1:]
            if _verify(cand):
                cur = cand
                changed = True
        if _elapsed() > TIME_LIMIT:
            break

    if len(cur) < best_len:
        best = cur
        best_len = len(cur)
    
    # Additional deletion passes from different starting points
    for restart in range(5):
        if _elapsed() > TIME_LIMIT:
            break
        cur = list(best)
        # Randomize order of deletion attempts
        indices = list(range(len(cur)))
        rng2 = random.Random(1000 + restart)
        rng2.shuffle(indices)
        changed = True
        while changed and _elapsed() < TIME_LIMIT:
            changed = False
            for idx in indices:
                if idx >= len(cur):
                    continue
                cand = cur[:idx] + cur[idx + 1:]
                if _verify(cand):
                    cur = cand
                    changed = True
                    if len(cur) < best_len:
                        best_len = len(cur)
                        best = list(cur)
                    break
            if changed:
                indices = list(range(len(cur)))
                rng2.shuffle(indices)
>>>>>>> REPLACE

