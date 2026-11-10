import time
import random

_START = time.time()
_DEADLINE = 110.0

# Precompute
_ALL = []
for a in range(16):
    for b in range(a+1,16):
        for c in range(b+1,16):
            _ALL.append((a,b,c))
_NALL = len(_ALL)

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _indep(triples):
    bad = bytearray(65536)
    for (a,b,c) in triples:
        tm = (1<<a)|(1<<b)|(1<<c)
        free = (~tm) & 0xFFFF
        sub = free
        while True:
            bad[tm|sub] = 1
            if sub == 0:
                break
            sub = (sub-1) & free
    best = 0
    for s in range(65536):
        if not bad[s]:
            pc = s.bit_count()
            if pc > best:
                best = pc
    return best

def construct():
    rng = random.Random(12345)
    best_triples = None
    best_val = 99

    while _time_left() > 5.0:
        # Greedy maximal
        triples = []
        used = set()
        order = list(range(_NALL))
        rng.shuffle(order)
        for idx in order:
            a,b,c = _ALL[idx]
            if (a,b) in used or (a,c) in used or (b,c) in used:
                continue
            used.add((a,b)); used.add((a,c)); used.add((b,c))
            triples.append((a,b,c))
        val = _indep(triples)
        if val < best_val:
            best_val = val
            best_triples = list(triples)

        # Local search
        n = len(triples)
        improved_any = True
        while improved_any and _time_left() > 5.0:
            improved_any = False
            for _ in range(n):
                if _time_left() < 5.0:
                    break
                i = rng.randrange(n)
                t0 = triples[i]
                a,b,c = t0
                for p in ((a,b),(a,c),(b,c)):
                    used.discard(p)
                # try candidates
                for _ in range(40):
                    cand = _ALL[rng.randrange(_NALL)]
                    x,y,z = cand
                    if cand == t0:
                        continue
                    if (x,y) in used or (x,z) in used or (y,z) in used:
                        continue
                    used.add((x,y)); used.add((x,z)); used.add((y,z))
                    triples[i] = cand
                    nv = _indep(triples)
                    if nv < val:
                        val = nv
                        improved_any = True
                        if val < best_val:
                            best_val = val
                            best_triples = list(triples)
                        break
                    else:
                        used.discard((x,y)); used.discard((x,z)); used.discard((y,z))
                        triples[i] = t0
                else:
                    # no candidate accepted, restore t0
                    a,b,c = t0
                    for p in ((a,b),(a,c),(b,c)):
                        used.add(p)
                if val <= 4:
                    break
        if best_val <= 4:
            break

    # Return as list of [a,b,c]
    if best_triples is None:
        best_triples = list(triples)
    return [[a,b,c] for (a,b,c) in best_triples]
