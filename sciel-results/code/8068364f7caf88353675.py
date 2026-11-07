<<<<<<< SEARCH
def construct():
    t0 = time.time()
    deadline = t0 + 88.0
    random.seed(2024)

    S = [_cyclic()]

    while time.time() < deadline and len(S) < 9:
        prev_len = len(S)
        remaining = deadline - time.time()
        if remaining <= 0.05:
            break
        progress = False
        extra = 0
        while (time.time() < deadline) and (not progress) and (extra < 4):
            rem = deadline - time.time()
            if rem <= 0.05:
                break
            tl = min(1.4, rem * 0.9)
            nl = 250000 if extra == 0 else 120000
            newq = _find_orthogonal(S, nl, tl, time.time())
            extra += 1
            if newq is not None:
                S.append(newq)
                progress = True
        if not progress:
            break

    if _verify(S):
        return S
    for k in range(len(S), 0, -1):
        sub = S[:k]
        if _verify(sub):
            return sub
    return [_cyclic()]
=======
def construct():
    t0 = time.time()
    deadline = t0 + 88.0
    random.seed(2024)

    # Start with cyclic square, then try to add more via search
    S = [_cyclic()]

    while time.time() < deadline and len(S) < 9:
        remaining = deadline - time.time()
        if remaining <= 0.05:
            break
        progress = False
        extra = 0
        while (time.time() < deadline) and (not progress) and (extra < 6):
            rem = deadline - time.time()
            if rem <= 0.05:
                break
            # Scale time limit based on remaining time and how many squares we have
            tl = min(2.0, rem * 0.85)
            nl = 400000 if extra == 0 else 200000
            newq = _find_orthogonal(S, nl, tl, time.time())
            extra += 1
            if newq is not None:
                S.append(newq)
                progress = True
        if not progress:
            break

    if _verify(S):
        return S
    for k in range(len(S), 0, -1):
        sub = S[:k]
        if _verify(sub):
            return sub
    return [_cyclic()]
>>>>>>> REPLACE
