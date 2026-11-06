import random
import time

def _score(S):
    s = set(S)
    diffs = set()
    for a in s:
        for b in s:
            if b > a:
                d = b - a
                if a + 2*d in s:
                    diffs.add(d)
    return len(diffs)

def construct():
    t0 = time.time()
    DL = t0 + 138

    best = 0
    bestS = None
    rng = random.Random(7)

    def ls(init, steps=400):
        nonlocal best, bestS
        cur = list(init)
        cs = _score(cur)
        if cs > best:
            best = cs
            bestS = list(cur)
        for _ in range(steps):
            if time.time() > DL - 2:
                break
            i = rng.randrange(32)
            ov = cur[i]
            nv = rng.randint(1, 512)
            if nv in cur:
                continue
            cur[i] = nv
            ns = _score(cur)
            if ns >= cs:
                cs = ns
                if ns > best:
                    best = ns
                    bestS = list(cur)
            else:
                cur[i] = ov
        return cs

    for off in range(0, 481, 10):
        if time.time() > DL - 5:
            break
        ls([off + 1 + i for i in range(32)], 300)

    for _ in range(80):
        if time.time() > DL - 5:
            break
        ls(rng.sample(range(1, 513), 32), 300)

    for m in [1, 2, 3, 4, 5, 6]:
        if time.time() > DL - 5:
            break
        s = [m*(i+1) for i in range(32)]
        if max(s) <= 512:
            ls(s, 300)

    return bestS
