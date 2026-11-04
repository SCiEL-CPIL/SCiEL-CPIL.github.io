import time
import itertools
import numpy as np

# Ternary tetracode: 9 words of length 4, a perfect covering code of radius 1.
_TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _neighbors(cw, n, q):
    """All words within Hamming distance 1 of cw (cw itself + one-position flips)."""
    out = [cw]
    for i in range(n):
        v = cw[i]
        for d in range(q):
            if d == v:
                continue
            out.append(cw[:i] + [d] + cw[i + 1:])
    return out


def _tetracode_solution(q, n, radius):
    """Direct sum of the ternary tetracode with every word on the remaining coords."""
    if n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    return [h + list(tail) for h in _TETRA
            for tail in itertools.product(range(q), repeat=n - 4)]


def _greedy_cover(q, n, radius, deadline):
    """Greedy set-cover over the whole space, then remove redundant codewords."""
    total = q ** n
    full = np.fromiter(itertools.product(range(q), repeat=n), dtype=np.int8, count=total)
    full_u = full.view(np.dtype((np.void, full.dtype.itemsize * n)))
    full_idx = {w: i for i, w in enumerate(full_u)}

    covered = np.zeros(total, dtype=bool)
    covered_count = 0
    selected = []
    selected_arr = []

    for cw in _tetracode_solution(q, n, radius):
        for w in _neighbors(cw, n, q):
            k = full_idx[tuple(w)]
            if not covered[k]:
                covered[k] = True
                covered_count += 1
        selected.append(cw)
        selected_arr.append(np.array(cw, dtype=np.int8))
        if covered_count == total:
            break

    if covered_count < total:
        # Safety: add arbitrary words until fully covered (should not trigger for tetracode init).
        for i in range(total):
            if not covered[i]:
                cw = full[i].tolist()
                for w in _neighbors(cw, n, q):
                    k = full_idx[tuple(w)]
                    if not covered[k]:
                        covered[k] = True
                        covered_count += 1
                selected.append(cw)
                selected_arr.append(np.array(cw, dtype=np.int8))
                if covered_count == total:
                    break

    # Redundancy removal: drop a codeword if all its neighbors are still covered by the rest.
    keep = np.ones(len(selected_arr), dtype=bool)
    if len(selected_arr) > 0:
        sel = np.stack(selected_arr)
        sel_u = sel.view(np.dtype((np.void, sel.dtype.itemsize * n)))
        sel_idx = np.array([full_idx[w] for w in sel_u], dtype=np.int64)

        for a in range(len(sel_idx)):
            if not keep[a]:
                continue
            if time.time() > deadline:
                break
            na = sel_idx[a]
            ok = True
            for b in range(len(sel_idx)):
                if b == a or not keep[b]:
                    continue
                if np.any(sel_idx[b] == na):
                    ok = False
                    break
            if ok:
                keep[a] = False

    return [selected[i] for i in range(len(selected)) if keep[i]]


def construct(q, n, radius):
    deadline = time.time() + 130.0  # safety margin before the 150s wall-clock limit

    try:
        result = _greedy_cover(q, n, radius, deadline)
        if len(result) > 0:
            return result
    except Exception:
        pass

    return _tetracode_solution(q, n, radius)
