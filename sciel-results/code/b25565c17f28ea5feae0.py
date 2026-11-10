def construct():
    import time
    from math import gcd
    N = 512
    start = time.time()
    deadline = start + 140.0  # safety margin under the 150s kill

    def score(S):
        return sum(1.0 / (N - a) for a in S)

    def greedy(order):
        S = []
        for a in order:
            ok = True
            for b in S:
                if gcd(a, b) != 1:
                    ok = False
                    break
            if ok:
                S.append(a)
        return S

    # Initial greedy in descending order (prefers large a -> large 1/(512-a))
    best = greedy(range(511, 0, -1))
    best_score = score(best)

    # Local search: drop one chosen element, then re-greedy descending.
    # Dropping an element can free coprime conflicts so that a higher-weight
    # composite (or set of them) replaces a blocking small prime, raising the sum.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best):
            cand = [a for a in range(511, 0, -1) if a != x]
            S2 = greedy(cand)
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_score = s2
                improved = True
                break

    return best
