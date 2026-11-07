    for i in range(N):
        if not (mask >> i) & 1: continue
        a = i + 1
        max_b = 128 - a
        for j in range(i + 1, min(max_b, N)):
            if not (mask >> j) & 1: continue
            b = j + 1
            ab_bit = i + j + 1
            if ab_bit >= N: continue
            if not (mask >> ab_bit) & 1: continue
            # Check c
            candidates = mask & (mask >> i) & (mask >> j)
            candidates &= ~((1 << i) | (1 << j))
            if candidates: return False
    