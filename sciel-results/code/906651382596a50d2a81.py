def construct():
    N = 180
    bad_roots = set()
    for s in range(1, N + 1):
        s_sq = s * s
        # Find divisors x of s_sq
        # x != y => x != s
        # s + x <= N and s + s_sq//x <= N
        found = False
        # Iterate x up to sqrt(s_sq) or just all divisors
        # Since N is small, we can just check x from 1 to N-s
        for x in range(1, N - s + 1):
            if s_sq % x == 0:
                y = s_sq // x
                if x == y:
                    continue
                if s + y <= N:
                    found = True
                    break
        if found:
            bad_roots.add(s)
    
    return [i for i in range(1, N + 1) if i not in bad_roots]
