# EVOLVE-BLOCK-START
import time
import sys

sys.setrecursionlimit(10000)
_START = time.time()
_DEADLINE = _START + 50.0  # safety margin before the 90s kill

# Squarefree candidates in 2..40 with their prime-support bitmask (2..40 -> bits 0..12)
_CANDS = []
for a in range(2, 41):
    x = a
    mask = 0
    sq = False
    d = 2
    while d * d <= x:
        if x % d == 0:
            x //= d
            if x % d == 0:
                sq = True
                break
            mask |= (1 << d)
        d += 1
    if sq:
        continue
    if x > 1:
        mask |= (1 << x)
    _CANDS.append((mask, a))

_CANDS.sort(key=lambda t: t[0], reverse=True)  # larger support first -> better upper bound
_M = len(_CANDS)
_BM = [m for m, _ in _CANDS]
_BI = [i for _, i in _CANDS]
_S = [0] * (_M + 1)
for _i in range(_M - 1, -1, -1):
    _S[_i] = _S[_i + 1] + _BM[_i]


def _bt(idx, cur, cnt, best):
    if cnt + (_M - idx) <= best[0]:
        return
    if idx == _M:
        if cnt > best[0]:
            best[0] = cnt
            best[1] = list(cur)
        return
    if time.time() > _DEADLINE:
        return
    m = _BM[idx]
    if (m & cur) == 0:
        cur.append(m)
        _bt(idx + 1, cur, cnt + 1, best)
        cur.pop()
    _bt(idx + 1, cur, cnt, best)


def construct():
    best = [0, []]
    _bt(0, [], 0, best)
    if best[1]:
        return [_BI[i] for i in best[1]]
    # Fallback: primes (always valid)
    out = []
    for a in range(2, 41):
        ok = True
        d = 2
        while d * d <= a:
            if a % d == 0:
                ok = False
                break
            d += 1
        if ok:
            out.append(a)
    return out
# EVOLVE-BLOCK-END
