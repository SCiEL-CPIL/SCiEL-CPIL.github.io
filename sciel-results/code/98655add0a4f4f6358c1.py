# EVOLVE-BLOCK-START
import time
import numpy as np

_P = 7
_Q = 49
_K = 3
_TOTAL = _Q * _Q
_NDIR = _Q + 1
_DEADLINE = 540.0


def _build_lines():
    L = np.empty((_NDIR, _TOTAL), dtype=np.int16)
    for d in range(_NDIR):
        if d < _Q:
            m = d
            dx1, dx2 = 1, 0
            dy1, dy2 = m % _P, m // _P
        else:
            dx1, dx2 = 0, 0
            dy1, dy2 = 1, 0
        for p in range(_TOTAL):
            x1, x2 = p % _P, (p // _P) % _P
            y1, y2 = (p // _Q) % _P, (p // _Q) // _P
            nx1 = (x1 + dx1) % _P
            nx2 = (x2 + dx2) % _P
            ny1 = (y1 + dy1) % _P
            ny2 = (y2 + dy2) % _P
            L[d, p] = nx1 + _P * nx2 + _Q * (ny1 + _P * ny2)
    return L


def _build_seed_s():
    s = np.zeros(_TOTAL, dtype=np.int8)
    y_special = set()
    for u in range(_P):
        for v in range(_P):
            y_special.add(((u * u) % _P, v))
    for p in range(_TOTAL):
        xa = p % _P
        ya = (p // _Q) % _P
        yb = (p // _Q) // _P
        in_unital = (2 * xa + ya * ya - yb * yb * _K) % _P == 0
        if in_unital and (ya, yb) not in y_special:
            s[p] = 1
    return s


def _validate(s, L):
    n = (1 - s).astype(np.int8)
    for d in range(_NDIR):
        other = L[d]
        if np.all(n[other] == 1):
            return True
    return False


def _line_groups():
    """For each direction d, list of lines (each a list of _TOTAL-ids of size _Q)."""
    groups = []
    for d in range(_NDIR):
        if d < _Q:
            m = d
            dx1, dx2 = 1, 0
            dy1, dy2 = m % _P, m // _P
        else:
            dx1, dx2 = 0, 0
            dy1, dy2 = 1, 0
        seen = np.full(_TOTAL, -1, dtype=np.int32)
        lines = []
        for p in range(_TOTAL):
            if seen[p] != -1:
                continue
            line = []
            cur = p
            for _ in range(_Q):
                seen[cur] = len(line)
                line.append(cur)
                x1, x2 = cur % _P, (cur // _P) % _P
                y1, y2 = (cur // _Q) % _P, (cur // _Q) // _P
                nx1 = (x1 + dx1) % _P
                nx2 = (x2 + dx2) % _P
                ny1 = (y1 + dy1) % _P
                ny2 = (y2 + dy2) % _P
                cur = nx1 + _P * nx2 + _Q * (ny1 + _P * ny2)
            lines.append(np.array(line, dtype=np.int16))
        groups.append(lines)
    return groups


def nikodym_set() -> np.ndarray:
    t0 = time.time()
    L = _build_lines()
    groups = _line_groups()

    s = _build_seed_s()
    if not _validate(s, L):
        s = np.zeros(_TOTAL, dtype=np.int8)
        y_special = set()
        for u in range(_P):
            for v in range(_P):
                y_special.add(((u * u) % _P, v))
        for p in range(_TOTAL):
            xa = p % _P
            ya = (p // _Q) % _P
            yb = (p // _Q) // _P
            in_unital = (2 * xa + ya * ya - yb * yb * _K) % _P == 0
            if in_unital and (ya, yb) in y_special:
                s[p] = 1
        if not _validate(s, L):
            s = np.zeros(_TOTAL, dtype=np.int8)

    n = (1 - s).astype(np.int8)

    # cover[p] = number of directions d for which p is the unique N-point on its
    # direction-d line (so removing p from N would break that direction).
    cover = np.zeros(_TOTAL, dtype=np.int32)
    for d in range(_NDIR):
        for line in groups[d]:
            cnt = int(np.sum(n[line] == 1))
            if cnt == 1:
                for q in line:
                    if n[q] == 1:
                        cover[q] += 1

    # Greedy: add to S the in-N point with smallest cover that keeps validity.
    in_n_mask = (s == 0)
    for _ in range(2000000):
        if time.time() > _DEADLINE:
            break
        idx = np.where(in_n_mask)[0]
        if len(idx) == 0:
            break
        covs = cover[idx]
        order = np.argsort(covs)
        placed = False
        for j in order[:8]:
            cand = int(idx[j])
            if covs[j] > 0:
                continue
            s[cand] = 1
            n[cand] = 0
            if _validate(s, L):
                in_n_mask[cand] = False
                # update cover for all lines containing cand
                for d in range(_NDIR):
                    for line in groups[d]:
                        if cand in line:
                            cnt = int(np.sum(n[line] == 1))
                            if cnt == 1:
                                for q in line:
                                    if n[q] == 1:
                                        cover[q] += 1
                            else:
                                for q in line:
                                    if n[q] == 1:
                                        cover[q] -= 1
                cover[cand] = 0
                placed = True
                break
            else:
                s[cand] = 0
                n[cand] = 1
                cover[cand] = 10**9
        if not placed:
            break

    if not _validate(s, L):
        s = _build_seed_s()

    pts = []
    for p in range(_TOTAL):
        if s[p] == 0:
            xa = p % _P
            xb = (p // _P) % _P
            ya = (p // _Q) % _P
            yb = (p // _Q) // _P
            pts.append([[xa, xb], [ya, yb]])
    return np.array(pts, dtype=np.int64)


# EVOLVE-BLOCK-END
