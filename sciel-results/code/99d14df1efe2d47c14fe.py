def _greedy_construct(deadline, rng):
    S = set()
    current_mask = 0
    current_score = 0
    for _ in range(32):
        best_x = -1
        best_s = -1
        for x in range(1, 513):
            if x in S:
                continue
            new_mask = current_mask | (1 << (x - 1))
            s = _score_bitset(new_mask)
            if s > best_s:
                best_s = s
                best_x = x
        if best_x != -1:
            S.add(best_x)
            current_mask |= (1 << (best_x - 1))
            current_score = best_s
        if time.time() > deadline:
            break
    return S, current_score
