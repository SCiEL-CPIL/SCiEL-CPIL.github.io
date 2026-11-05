import time

def construct():
    deadline = time.time() + 75.0
    N = 256

    best = [1]
    best_len = 1

    # state: seq = chosen values, P = prefix sums (P[0]=0, P[i]=sum of first i)
    # new consecutive-block sums when appending x = x + P[i] for i = 0..len(seq)-1
    # i.e. (P_new[i+1]-P[i]) for i = 0..len(seq)  with P_new[len+1]=P[len]+x
    # These must all be distinct and not collide with existing block sums.

    # Precompute existing block sums as a set for collision checks.
    # We maintain sums_set incrementally.

    def block_sums(P, L):
        # return set of all consecutive block sums for a sequence of length L with prefix P
        s = set()
        for i in range(L + 1):
            for j in range(i + 1, L + 1):
                s.add(P[j] - P[i])
        return s

    # Feasibility: can we place at least k more values >= lo such that total stays <= N?
    # Minimal sum of k distinct increasing ints >= lo is lo + (lo+1) + ... + (lo+k-1)
    def feasible(lo, k, N):
        if k <= 0:
            return True
        # sum of lo..lo+k-1
        return lo + (lo + k - 1) * k // 2 <= N

    # Upper bound on how many more values we can still add given current last value.
    def max_more(lo, N):
        # largest k such that lo..lo+k-1 sum <= N
        # k*(2*lo + k - 1)/2 <= N
        import math
        # solve k^2 + (2*lo-1)k - 2N <= 0
        a = 1
        b = 2 * lo - 1
        c = -2 * N
        disc = b * b - 4 * a * c
        if disc < 0:
            return 0
        k = (-b + math.isqrt(disc)) // 2
        # clamp
        while k > 0 and lo + (lo + k - 1) * k // 2 > N:
            k -= 1
        while k + 1 > 0 and lo + (lo + (k + 1) - 1) * (k + 1) // 2 <= N:
            k += 1
        return k

    def dfs(seq, P, sums_set, lo):
        nonlocal best, best_len, deadline
        L = len(seq)
        if L > best_len:
            best_len = L
            best = seq[:]
        if time.time() > deadline:
            return
        if L == N:
            return
        # prune: even with max more values, can't beat best
        # we need current + more > best_len to be worth continuing
        if L + max_more(lo, N) <= best_len:
            return
        # try each candidate x from lo to N
        for x in range(lo, N + 1):
            if time.time() > deadline:
                return
            # new block sums = x + P[i] for i in 0..L-1, plus x (i=L? no)
            # Actually new sums: for each i in 0..L: (P_new[L+1] - P_new[i]) where P_new[L+1]=P[L]+x
            # = (P[L]+x - P[i]) for i in 0..L
            base = P[L] + x
            ok = True
            new_sums = []
            for i in range(L + 1):
                s = base - P[i]
                if s in sums_set:
                    ok = False
                    break
                new_sums.append(s)
            if not ok:
                continue
            # also new_sums must be distinct among themselves
            # they are base - P[i]; P[i] strictly increasing so base-P[i] strictly decreasing -> distinct. Good.
            # extend
            new_P = P + [base]
            new_sums_set = sums_set | set(new_sums)
            dfs(seq + [x], new_P, new_sums_set, x + 1)

    # Greedy seed to get a strong starting best
    def greedy():
        vals = []
        P = [0]
        sums = set()
        for x in range(1, N + 1):
            base = P[-1] + x
            ok = True
            for i in range(len(vals) + 1):
                s = base - P[i]
                if s in sums:
                    ok = False
                    break
            if ok:
                vals.append(x)
                P.append(base)
                for i in range(len(vals)):
                    sums.add(base - P[i])
        return vals

    g = greedy()
    if len(g) > best_len:
        best = g
        best_len = len(g)

    # Now DFS to try to beat greedy. Start from empty.
    # To keep it tractable, limit DFS depth exploration by the time budget.
    dfs([], [0], set(), 1)

    return best
