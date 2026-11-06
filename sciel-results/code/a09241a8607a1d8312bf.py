def construct():
    import random
    from math import gcd
    N = 128
    nums = list(range(1, N+1))
    # Precompute gcd matrix indexed 1..N
    G = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            G[i][j] = g
            G[j][i] = g

    def is_bad_triple(a, b, c):
        return G[a][b] == G[a][c] == G[b][c]

    def can_add(x, S_set, S_list):
        # S_list is list of elements in S_set
        for idx in range(len(S_list)):
            a = S_list[idx]
            for jdx in range(idx+1, len(S_list)):
                b = S_list[jdx]
                g = G[a][b]
                if G[x][a] == g and G[x][b] == g:
                    return False
        return True

    def is_valid(S_list):
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                g = G[a][b]
                for k in range(j+1, n):
                    c = S_list[k]
                    if G[a][c] == g and G[b][c] == g:
                        return False
        return True

    def repair(S_list):
        # Remove elements until valid (greedy removal of a participant in a bad triple)
        S_list = list(S_list)
        changed = True
        while changed:
            changed = False
            n = len(S_list)
            for i in range(n):
                a = S_list[i]
                for j in range(i+1, n):
                    b = S_list[j]
                    g = G[a][b]
                    for k in range(j+1, n):
                        c = S_list[k]
                        if G[a][c] == g and G[b][c] == g:
                            # remove c (or the "least useful"); remove c
                            S_list.remove(c)
                            changed = True
                            break
                    if changed: break
                if changed: break
        return S_list

    best = None
    best_size = 0
    rng = random.Random(12345)

    def greedy(order):
        S = []
        Sset = set()
        for x in order:
            if x in Sset:
                continue
            if can_add(x, Sset, S):
                S.append(x)
                Sset.add(x)
        return S

    def local_improve(S, iters=500):
        S = list(S)
        Sset = set(S)
        for _ in range(iters):
            improved = False
            # 1. Try adds
            for x in nums:
                if x in Sset:
                    continue
                if can_add(x, Sset, S):
                    S.append(x)
                    Sset.add(x)
                    improved = True
            if improved:
                continue
            
            # 2. Try 1-for-1 swaps
            progressed = False
            for a in list(S):
                for b in nums:
                    if b in Sset:
                        continue
                    S2 = [t for t in S if t != a]
                    S2set = Sset - {a}
                    if can_add(b, S2set, S2):
                        # Accept swap if it allows further growth or is neutral
                        # Heuristic: Accept if it doesn't reduce size
                        S.append(b)
                        Sset.add(b)
                        S.remove(a)
                        Sset.discard(a)
                        progressed = True
                        # Try adds again
                        for x in nums:
                            if x in Sset:
                                continue
                            if can_add(x, Sset, S):
                                S.append(x)
                                Sset.add(x)
                        break
                if progressed:
                    break
            if progressed:
                continue
                
            # 3. Try 2-for-2 swaps (more powerful escape)
            progressed = False
            nS = len(S)
            # Limit combinations to avoid O(N^4) explosion if S is large
            # We only do this if S is not too large or we are stuck
            if nS < 60: 
                for i in range(nS):
                    a = S[i]
                    for j in range(i+1, nS):
                        b = S[j]
                        # Remove a, b
                        S2 = [t for t in S if t != a and t != b]
                        S2set = Sset - {a, b}
                        # Try to add two new elements
                        added_count = 0
                        new_S = list(S2)
                        new_Sset = set(S2set)
                        # Try to add any two from nums not in S2
                        candidates = [x for x in nums if x not in new_Sset]
                        # Randomize candidate order for diversity
                        rng.shuffle(candidates)
                        for x in candidates:
                            if can_add(x, new_Sset, new_S):
                                new_S.append(x)
                                new_Sset.add(x)
                                added_count += 1
                                if added_count == 2:
                                    break
                        if added_count == 2 and len(new_S) > len(S):
                            S = new_S
                            Sset = new_Sset
                            progressed = True
                            break
                        # Even if size doesn't increase, accept if it breaks symmetry?
                        # For now, only accept if size increases or (same size and we are stuck)
                        # To keep it simple and fast, only accept size increases in 2-2
                        # Or accept same size swaps if they allow future adds?
                        # Let's just accept if valid and we are stuck.
                        # But checking validity of new_S is expensive. can_add ensures validity incrementally.
                        # So new_S is valid.
                        if added_count == 2:
                            # Same size swap. Accept with some probability or if it helps?
                            # Let's just accept it to escape local minima.
                            S = new_S
                            Sset = new_Sset
                            progressed = True
                            break
                    if progressed:
                        break
            if progressed:
                continue

            # 4. If nothing worked, break
            break
        return S

    # Heuristic base order: numbers sorted by (number of divisors) descending? 
    # Try multiple strategies.
    strategies = []
    # strategy 1: ascending
    strategies.append(list(nums))
    # strategy 2: descending
    strategies.append(list(reversed(nums)))
    # strategy 3: by number of prime factors desc (more composite first)
    def omega(x):
        c = 0
        d = x
        while d % 2 == 0:
            c += 1; d //= 2
        p = 3
        while p*p <= d:
            while d % p == 0:
                c += 1; d //= p
            p += 2
        if d > 1: c += 1
        return c
    strategies.append(sorted(nums, key=lambda x: (-omega(x), x)))
    strategies.append(sorted(nums, key=lambda x: (omega(x), x)))
    # Random orders
    for _ in range(50):
        o = list(nums)
        rng.shuffle(o)
        strategies.append(o)

    for order in strategies:
        S = greedy(order)
        S = local_improve(S)
        if not is_valid(S):
            S = repair(S)
        if len(S) > best_size:
            best_size = len(S)
            best = list(S)

    if best is None or len(best) == 0:
        best = [2]

    # Final safety: ensure valid and nonempty
    if not is_valid(best):
        best = repair(best)
    if len(best) == 0:
        best = [2]
    return best
