import random
import time

_N = 16
_NK4 = 1820  # C(16,4)
_NEDGE = 120


def _build_k4_edges():
    k4s = []
    for a in range(_N):
        for b in range(a + 1, _N):
            for c in range(b + 1, _N):
                for d in range(c + 1, _N):
                    k4s.append(((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)))
    return k4s


_K4_EDGES = _build_k4_edges()
# For each edge (i,j), the list of (k4_index, local_position) it participates in.
_EDGE_TO_K4 = [[] for _ in range(_NEDGE)]
for _k4i, _e6 in enumerate(_K4_EDGES):
    for _pos, _e in enumerate(_e6):
        _EDGE_TO_K4[_e[0] * (_N - 1) // 2 + _e[1] - _e[0] - 1].append((_k4i, _pos))


def _edge_idx(i, j):
    return i * (_N - 1) // 2 + j - i - 1


def _validity(col):
    for e6 in _K4_EDGES:
        a = col[_edge_idx(*e6[0])]
        b = col[_edge_idx(*e6[1])]
        c = col[_edge_idx(*e6[2])]
        d = col[_edge_idx(*e6[3])]
        e = col[_edge_idx(*e6[4])]
        f = col[_edge_idx(*e6[5])]
        if len({a, b, c, d, e, f}) < 5:
            return False
    return True


def construct():
    deadline = time.time() + 115.0
    rng = random.Random(12345)

    best = [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]
    best_count = len(set(best))

    for restart in range(600000):
        if time.time() > deadline:
            break
        col = best[:]
        used = set(col)
        # local search: try to reduce colour count
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            order = list(range(_NEDGE))
            rng.shuffle(order)
            for ei in order:
                if time.time() > deadline:
                    break
                cur = col[ei]
                # candidate colours: all currently used colours
                cands = list(used)
                if cur not in cands:
                    cands.append(cur)
                rng.shuffle(cands)
                for nc in cands:
                    if nc == cur:
                        continue
                    # only accept if it reduces the used-set size and stays valid
                    # quick check: nc must be the only occurrence of cur, or cur
                    # remains used elsewhere
                    col[ei] = nc
                    # count occurrences of cur to see if used-set shrinks
                    still_cur = False
                    for k in range(_NEDGE):
                        if k != ei and col[k] == cur:
                            still_cur = True
                            break
                    if still_cur:
                        # used-set size unchanged -> not a strict reduction; revert
                        col[ei] = cur
                        continue
                    # used-set may shrink by 1; verify validity
                    if _validity(col):
                        used.discard(cur)
                        used.add(nc)
                        best_count = len(used)
                        improved = True
                        break
                    else:
                        col[ei] = cur
                # end for nc
        # end while improved
        if len(set(col)) < best_count:
            best_count = len(set(col))
            best = col[:]

    return best
