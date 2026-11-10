import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0
    N = 256

    def count_3aps(S):
        arr = [False] * (N + 1)
        for x in S:
            arr[x] = True
        count = 0
        for a in range(1, N + 1):
            if not arr[a]:
                continue
            for b in range(a + 1, N + 1):
                if not arr[b]:
                    continue
                c = 2 * b - a
                if c > N:
                    break
                if c > b and arr[c]:
                    count += 1
        return count

    def has_4ap(S):
        arr = [False] * (N + 1)
        for x in S:
            arr[x] = True
        for a in range(1, N + 1):
            if not arr[a]:
                continue
            for d in range(1, (N - a) // 3 + 1):
                if arr[a + d] and arr[a + 2 * d] and arr[a + 3 * d]:
                    return True
        return False

    def greedy_init():
        S = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            ok = True
            for d in range(1, N):
                if x + 3 * d <= N and (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                    ok = False
                    break
                if x - d >= 1 and (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                    ok = False
                    break
                if x - 2 * d >= 1 and (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                    ok = False
                    break
                if x - 3 * d >= 1 and (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                    ok = False
                    break
            if ok:
                S.add(x)
        return S

    # Get best initial solution
    best_S = None
    best_score = -1
    for i in range(5):
        if time.time() - start_time > 5:
            break
        S = greedy_init()
        sc = count_3aps(S)
        if sc > best_score:
            best_score = sc
            best_S = S

    if best_S is None:
        best_S = greedy_init()
        best_score = count_3aps(best_S)

    S = set(best_S)
    cur = best_score

    # Local search
    while time.time() - start_time < TIME_LIMIT - 2:
        improved = False

        # Try adding elements
        cands = [x for x in range(1, N + 1) if x not in S]
        if cands:
            random.shuffle(cands)
            for x in cands[:80]:
                newS = S | {x}
                if not has_4ap(newS):
                    sc = count_3aps(newS)
                    if sc > cur:
                        S = newS
                        cur = sc
                        improved = True
                        break

        # Try removing elements
        if not improved and S:
            elems = list(S)
            random.shuffle(elems)
            for x in elems[:40]:
                newS = S - {x}
                sc = count_3aps(newS)
                if sc > cur:
                    S = newS
                    cur = sc
                    improved = True
                    break

        # Try swapping
        if not improved and S and cands:
            elems = list(S)
            random.shuffle(elems)
            adds = cands
            random.shuffle(adds)
            for xo in elems[:20]:
                for xi in adds[:20]:
                    newS = (S - {xo}) | {xi}
                    if not has_4ap(newS):
                        sc = count_3aps(newS)
                        if sc > cur:
                            S = newS
                            cur = sc
                            improved = True
                            break
                if improved:
                    break

        if not improved:
            # Perturbation: remove a random element to escape local optimum
            if S and len(S) > 3:
                elems = list(S)
                random.shuffle(elems)
                x = elems[0]
                S.discard(x)
                cur = count_3aps(S)

    return sorted(S)
