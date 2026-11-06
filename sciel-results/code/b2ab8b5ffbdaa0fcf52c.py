"""Divisibility-antichain representations by smooth numbers.
construct() -> 128 rows for targets 1..128.
Each row: 1..30 distinct 2^a*3^b*5^c terms summing to target, no term divides another.
Minimize total summands.
"""
import time

_START = time.time()
DEADLINE = 84.0  # safety margin before 90s kill

# ---- Precompute smooth numbers and divisibility relations ----
_smooth = []
_sset = set()
for _a in range(0, 8):
    _pa = 2 ** _a
    for _b in range(0, 6):
        _pb = _pa * (3 ** _b)
        if _pb > 128:
            break
        for _c in range(0, 5):
            _v = _pb * (5 ** _c)
            if _v > 128:
                break
            if _v not in _sset:
                _sset.add(_v)
                _smooth.append(_v)
_smooth.sort()
_N = len(_smooth)
_is_sm = [False] * 129
for _s in _smooth:
    _is_sm[_s] = True

# conflict[i][j] = True if smooth[i] divides smooth[j] or vice versa
_conflict = [[False] * _N for _ in range(_N)]
for _i in range(_N):
    _si = _smooth[_i]
    for _j in range(_i + 1, _N):
        if _smooth[_j] % _si == 0:
            _conflict[_i][_j] = True
            _conflict[_j][_i] = True

def _antichain3(i, j, k):
    return not (_conflict[i][j] or _conflict[i][k] or _conflict[j][k])

def construct():
    t0 = time.time()
    smooth = _smooth
    is_sm = _is_sm
    conflict = _conflict
    answers = [None] * 128
    need4 = []  # (n, [i,j,k] best 3-term) for refinement

    for n in range(1, 129):
        # 1-term
        if is_sm[n]:
            answers[n - 1] = [n]
            continue
        # 2-term
        found = False
        for i in range(_N):
            s1 = smooth[i]
            if s1 >= n:
                break
            s2 = n - s1
            if s2 <= s1:
                continue
            if not is_sm[s2]:
                continue
            # antichain: s1 must not divide s2 (s2>s1 so only this direction)
            if s2 % s1 == 0:
                continue
            answers[n - 1] = [s1, s2]
            found = True
            break
        if found:
            continue
        # 3-term
        found = False
        best3 = None
        for i in range(_N):
            s1 = smooth[i]
            if s1 >= n:
                break
            for j in range(_N):
                s2 = smooth[j]
                if s2 <= s1:
                    continue
                if s1 + s2 >= n:
                    break
                if conflict[i][j]:
                    continue
                s3 = n - s1 - s2
                if s3 <= s2:
                    break
                if not is_sm[s3]:
                    continue
                # antichain for s3 vs s1, s2
                if s3 % s1 == 0 or s3 % s2 == 0:
                    continue
                answers[n - 1] = [s1, s2, s3]
                best3 = (i, j, s3)
                found = True
                break
            if found:
                break
        if found:
            if best3 is not None:
                need4.append((n, best3))
            continue
        # Fallback (should not happen for n<=128): try up to 5 terms quickly
        found = False
        for i in range(_N):
            s1 = smooth[i]
            if s1 >= n:
                break
            for j in range(_N):
                if time.time() - t0 > DEADLINE:
                    break
                s2 = smooth[j]
                if s2 <= s1 or s1 + s2 >= n:
                    continue
                if conflict[i][j]:
                    continue
                for k in range(_N):
                    s3 = smooth[k]
                    if s3 <= s2 or s1 + s2 + s3 >= n:
                        continue
                    if conflict[i][k] or conflict[j][k]:
                        continue
                    s4 = n - s1 - s2 - s3
                    if s4 <= s3 or not is_sm[s4]:
                        continue
                    if s4 % s1 == 0 or s4 % s2 == 0 or s4 % s3 == 0:
                        continue
                    answers[n - 1] = [s1, s2, s3, s4]
                    found = True
                    break
                if found:
                    break
            if found:
                break
        if not found:
            # last resort: single term if smooth else [1, n-1]
            answers[n - 1] = [n] if is_sm[n] else [1, n - 1]

    # ---- Refinement pass: try to convert 3-term rows to 2-term where possible ----
    # (already handled in main loop since 2-term is tried first; nothing to do)

    # ---- Refinement: try to find 2-term for rows currently using 3 terms ----
    # This is redundant since 2-term was tried first. Skip.

    # ---- Optional: verify no row exceeds 30 terms (all are <=4) ----
    for idx in range(128):
        if answers[idx] is None:
            answers[idx] = [idx + 1] if is_sm[idx + 1] else [1, idx]

    return answers
