"""Improved biclique partition: maximal greedy extraction + merge + bounded local search."""
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_ADJ = [set() for _ in range(P['n'])]
_EDGES = []
for _a, _b in P['edges']:
    _a, _b = int(_a), int(_b)
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
    _EDGES.append((_a, _b))


def _is_biclique(L, R):
    """True iff every (l,r) in L x R is a host edge."""
    for l in L:
        al = _ADJ[l]
        for r in R:
            if r not in al:
                return False
    return True


def _extract_maximal(L, R, rem, remL, remR):
    """Grow (L,R) to a maximal biclique within the remaining edge set."""
    L = list(L); R = list(R)
    changed = True
    while changed:
        changed = False
        common = set(R)
        for l in L:
            common &= remL[l]
        for v in common:
            if v not in set(R) and v not in set(L):
                R.append(v); changed = True
        common2 = set(L)
        for r in R:
            common2 &= remR[r]
        for v in common2:
            if v not in set(L) and v not in set(R):
                L.append(v); changed = True
    return L, R


def _greedy_extract(rem, remL, remR, max_biclique=True, deadline=None):
    """Extract a partition of the remaining edges into bicliques."""
    out = []
    rem = dict(rem)
    remL = {k: set(v) for k, v in remL.items()}
    remR = {k: set(v) for k, v in remR.items()}
    while rem:
        if deadline is not None and time.time() > deadline:
            break
        # pick anchor edge with max (|remL[u]|, |remL[v]|)
        best = None
        for (u, v) in list(rem.keys()):
            s = max(len(remL[u]), len(remL[v]))
            if best is None or s > best[0]:
                best = (s, u, v)
        _, u, v = best
        if max_biclique:
            L, R = _extract_maximal([u], [v], rem, remL, remR)
        else:
            L, R = [u], [v]
        out.append((L, R))
        for l in L:
            for r in R:
                e = (l, r) if l < r else (r, l)
                rem.pop(e, None)
                remL[l].discard(r)
                remR[r].discard(l)
    return out


def _merge_pass(bics):
    """Merge bicliques whose union is still a valid biclique (repeatedly)."""
    bics = [[list(L), list(R)] for L, R in bics]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(bics):
            j = i + 1
            while j < len(bics):
                L = set(bics[i][0]) | set(bics[j][0])
                R = set(bics[i][1]) | set(bics[j][1])
                if _is_biclique(L, R):
                    bics[i] = [sorted(L), sorted(R)]
                    del bics[j]
                    changed = True
                else:
                    j += 1
            i += 1
    return bics


def _local_search(bics, deadline):
    """Try to reduce count by swapping edges between two bicliques."""
    bics = [[list(L), list(R)] for L, R in bics]
    improved = True
    while improved:
        improved = False
        for i in range(len(bics)):
            for j in range(i + 1, len(bics)):
                if time.time() > deadline:
                    return bics
                Li, Ri, Lj, Rj = bics[i][0], bics[i][1], bics[j][0], bics[j][1]
                # candidate: move all edges from biclique i that are also coverable by j
                moved = []
                for l in Li:
                    for r in Ri:
                        if r in _ADJ[l] and (l in set(Lj) or r in set(Rj)):
                            moved.append((l, r))
                if not moved:
                    continue
                # try removing 'moved' from i and see if i becomes empty
                newLi = set(Li); newRi = set(Ri)
                for l, r in moved:
                    newLi.discard(l); newRi.discard(r)
                if not newLi or not newRi:
                    # i gone; add moved to j and check j still biclique
                    newLj = set(Lj) | {l for l, r in moved}
                    newRj = set(Rj) | {r for l, r in moved}
                    if _is_biclique(newLj, newRj):
                        bics[j] = [sorted(newLj), sorted(newRj)]
                        del bics[i]
                        improved = True
                        break
            if improved:
                break
    return bics


def construct():
    start = time.time()
    deadline = start + 40.0  # safety margin well under 60s

    # Phase 1: greedy maximal extraction
    bics = _greedy_extract({(a, b) if a < b else (b, a) for a, b in _EDGES},
                           {u: set(_ADJ[u]) for u in range(P['n'])},
                           {u: set(_ADJ[u]) for u in range(P['n'])},
                           max_biclique=True, deadline=deadline)

    # Phase 2: merge compatible bicliques
    bics = _merge_pass(bics)

    # Phase 3: bounded local search (edge swaps)
    if time.time() < deadline:
        bics = _local_search(bics, deadline)

    # Phase 4: final merge
    bics = _merge_pass(bics)

    # Build result: each biclique [left, right]
    result = []
    for L, R in bics:
        L = [int(x) for x in L]
        R = [int(x) for x in R]
        if not L or not R:
            continue
        result.append([L, R])

    # Fallback: ensure validity; if something is off, use star partition
    if not result:
        result = [[[u], [int(v) for v in _ADJ[u]]] for u in range(P['n']) if _ADJ[u]]

    return result
