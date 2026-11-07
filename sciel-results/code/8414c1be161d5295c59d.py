import time

def _valid(S):
    L = sorted(S)
    n = len(L)
    for i in range(n):
        ai = L[i]
        for j in range(i+1, n):
            aj = L[j]
            for k in range(j+1, n):
                ak = L[k]
                if (ai+aj in S) and (ai+ak in S) and (aj+ak in S):
                    return False
    return True

def _greedy(base, cands, deadline):
    S = set(base)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in cands:
            if x in S:
                continue
            if _valid(S | {x}):
                S.add(x)
                changed = True
    return S

def construct():
    t0 = time.time()
    deadline = t0 + 80.0
    best = set(range(65,129))  # 64
    # try adding from [1..64] with various orders
    cands_desc = list(range(64,0,-1))
    cands_asc = list(range(1,65))
    for order in (cands_desc, cands_asc):
        if time.time() > deadline - 10:
            break
        S = _greedy(range(65,129), order, deadline - 5)
        if _valid(S) and len(S) > len(best):
            best = S
    return [int(v) for v in sorted(best)]
