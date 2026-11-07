import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill
    N = 128

    # Generate all forbidden configurations: distinct a<b<c with b+c<=N
    # (then a+b<=N and a+c<=N automatically). Config = {a,b,c,a+b,a+c,b+c}.
    cfg_set = set()
    for b in range(2, N + 1):
        max_c = N - b
        if max_c <= b:
            break
        for c in range(b + 1, max_c + 1):
            for a in range(1, b):
                cfg_set.add(frozenset((a, b, c, a + b, a + c, b + c)))
    configs = list(cfg_set)

    # For each element x, store the set (F \ {x}) for every config F containing x.
    # Adding x to S is unsafe iff some (F\{x}) is a subset of S.
    cbe = defaultdict(list)
    for F in configs:
        for x in F:
            cbe[x].append(F - {x})

    def safe_add(S, x):
        for G in cbe[x]:
            if G <= S:
                return False
        return True

    def valid_after_add(S, x):
        # S already valid; only new violations involve x.
        for G in cbe[x]:
            # G = F\{x}; F = G ∪ {x}; F ⊆ S iff G ⊆ S (x in S)
            if G <= S:
                return False
        return True

    def local_search(S):
        S = set(S)
        iters = 0
        while time.time() < deadline:
            iters += 1
            changed = False
            # Add phase
            for x in range(1, N + 1):
                if x not in S and safe_add(S, x):
                    S.add(x)
                    changed = True
            # Swap phase (neutral size, may enable future adds)
            for x in range(1, N + 1):
                if x in S:
                    continue
                # gather conflicting (F\{x}) sets
                conf = [G for G in cbe[x] if G <= S]
                if not conf:
                    continue
                swapped = False
                for G in conf:
                    for y in G:
                        if y in S:
                            S2 = S - {y}
                            S2.add(x)
                            if valid_after_add(S2, x):
                                S = S2
                                swapped = True
                                changed = True
                                break
                    if swapped or time.time() >= deadline:
                        break
                if time.time() >= deadline:
                    break
            if not changed:
                break
        return S

    best = None
    best_size = 0

    def consider(S):
        nonlocal best, best_size
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)

    # Trial 0: top half
    consider(local_search(set(range(N // 2 + 1, N + 1))))
    # Trial 1: odds
    consider(local_search(set(range(1, N + 1, 2))))

    # Random restarts
    random.seed(12345)
    for trial in range(400):
        if time.time() >= deadline - 3.0:
            break
        S = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if safe_add(S, x):
                S.add(x)
        S = local_search(S)
        consider(S)
        if time.time() >= deadline - 3.0:
            break

    if best is None:
        best = set(range(N // 2 + 1, N + 1))

    return [int(x) for x in sorted(best)]
