import time, random

def _prods_set(S):
    P = {1}
    for x in S:
        P |= {p*x for p in P}
    return P

def _can_add(P, x):
    # P = product set of current S
    for p in P:
        if p*x in P:
            return False
    return True

def _valid(S):
    P = _prods_set(S)
    return len(P) == (1 << len(S))

def _greedy():
    S = []
    P = {1}
    for x in range(2, 41):
        if _can_add(P, x):
            S.append(x)
            P |= {p*x for p in P}
    return S

def _local_search(S, deadline):
    S = list(S)
    P = _prods_set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # adds
        for x in range(2, 41):
            if x in S:
                continue
            if _can_add(P, x):
                S.append(x)
                P |= {p*x for p in P}
                improved = True
                break
        if improved:
            continue
        # swaps: remove a, add b
        for a in list(S):
            S2 = [k for k in S if k != a]
            P2 = _prods_set(S2)
            for b in range(2, 41):
                if b in S2:
                    continue
                if _can_add(P2, b):
                    cand = S2 + [b]
                    if len(cand) > len(S):
                        S = cand
                        P = _prods_set(S)
                        improved = True
                        break
            if improved:
                break
        if improved:
            continue
        # double swap: remove a, add b,c
        for a in list(S):
            base = [k for k in S if k != a]
            Pb = _prods_set(base)
            for b in range(2, 41):
                if b in base:
                    continue
                if not _can_add(Pb, b):
                    continue
                Pb2 = Pb | {p*b for p in Pb}
                for c in range(2, 41):
                    if c in base or c == b:
                        continue
                    if _can_add(Pb2, c):
                        cand = base + [b, c]
                        if len(cand) > len(S):
                            S = cand
                            P = _prods_set(S)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return S

def construct():
    start = time.time()
    deadline = start + 75.0
    best = _greedy()
    best = _local_search(best, deadline)
    rng = random.Random(20240607)
    while time.time() < deadline:
        size = rng.randint(6, 16)
        S = rng.sample(range(2, 41), size)
        # filter to independent
        ind = []
        P = {1}
        for x in S:
            if _can_add(P, x):
                ind.append(x)
                P |= {p*x for p in P}
        S = _local_search(ind, deadline)
        if len(S) > len(best):
            best = S
    if not _valid(best):
        best = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    return [int(x) for x in best]
