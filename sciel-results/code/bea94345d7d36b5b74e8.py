import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute divisibility: div[s][a] = (s % a == 0) for s in [2,512], a in [1,256]
    # Using a list of sets for faster lookup
    # For each a, precompute which sums s are divisible by a
    div_by = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for s in range(2 * a, 513, a):  # s = 2a, 3a, 4a, ... up to 512
            div_by[a].add(s)

    # Start with the "safe" upper set
    K = 171
    S = set(range(K, n + 1))
    S_list = sorted(S)

    def check_add(x, S_set):
        """Check if x can be added to S_set. Returns True if valid."""
        for b in S_set:
            for c in S_set:
                if c < b:
                    continue
                # Triple with smallest element being the divisor
                if x < b:
                    # (x, b, c): x | (b+c)?
                    if (b + c) in div_by[x]:
                        return False
                elif b < x <= c:
                    # (b, x, c): b | (x+c)?
                    if (x + c) in div_by[b]:
                        return False
                elif b < c < x:
                    # (b, c, x): b | (c+x)?
                    if (c + x) in div_by[b]:
                        return False
        return True

    # Greedy add smaller numbers from K-1 down to 1
    for x in range(K - 1, 0, -1):
        if time.time() > deadline - 25:
            break
        if check_add(x, S):
            S.add(x)

    # Local search: try adding elements not in S, with conflict-based removal
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        current_size = len(S)
        for a in range(1, n + 1):
            if time.time() > deadline - 10:
                break
            if a in S:
                continue

            # Check conflicts: which elements in S would need to be removed to add a?
            conflicts = set()
            for b in S:
                for c in S:
                    if c < b:
                        continue
                    if a < b:
                        if (b + c) in div_by[a]:
                            conflicts.add(b)
                            conflicts.add(c)
                    elif b < a <= c:
                        if (a + c) in div_by[b]:
                            conflicts.add(b)
                            conflicts.add(c)
                    elif b < c < a:
                        if (c + a) in div_by[b]:
                            conflicts.add(b)
                            conflicts.add(c)

            if len(conflicts) == 0:
                S.add(a)
                improved = True
            elif len(conflicts) < current_size * 0.25:
                new_S = S - conflicts
                new_S.add(a)
                if len(new_S) > len(S):
                    S = new_S
                    improved = True

    # Second pass: try removing one element and adding two others
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for rem in list(S):
            if time.time() > deadline - 5:
                break
            S_without = S - {rem}
            for add1 in range(1, n + 1):
                if add1 in S_without:
                    continue
                if not check_add(add1, S_without):
                    continue
                S_with1 = S_without | {add1}
                for add2 in range(add1 + 1, n + 1):
                    if add2 in S_with1:
                        continue
                    if not check_add(add2, S_with1):
                        continue
                    new_S = S_with1 | {add2}
                    if len(new_S) > len(S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
            if time.time() > deadline - 5:
                break

    # Third pass: 1-for-1 swaps (remove one, add one different)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for rem in list(S):
            if time.time() > deadline - 2:
                break
            S_without = S - {rem}
            for add in range(1, n + 1):
                if add in S_without or add == rem:
                    continue
                if check_add(add, S_without):
                    S = S_without | {add}
                    improved = True
                    break
            if improved:
                break

    return list(S)
