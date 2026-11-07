<<<<<<< SEARCH
def _greedy_init():
    remaining = {v: set(_ADJ[v]) for v in _ALLV}
    part = []
    while remaining:
        L, R = _find_max_biclique(remaining, _ADJ, _N)
        if L is None:
            for v in list(remaining.keys()):
                nbrs = list(remaining[v])
                if not nbrs:
                    del remaining[v]
                    continue
                for w in nbrs:
                    remaining[v].discard(w)
                    remaining[w].discard(v)
                part.append([[v], [w]] if False else [[v], [w]])
            continue
        for a in L:
            for b in R:
                remaining[a].discard(b)
                remaining[b].discard(a)
        for v in list(remaining.keys()):
            if not remaining[v]:
                del remaining[v]
        part.append([list(L), list(R)])
    return part
=======
def _greedy_init():
    remaining = {v: set(_ADJ[v]) for v in _ALLV}
    part = []
    while remaining:
        result = _find_max_biclique(remaining, _ADJ, _N)
        if result is None:
            # Fallback: find any edge in remaining and make a 1x1 biclique
            found = False
            for v in list(remaining.keys()):
                if not remaining[v]:
                    del remaining[v]
                    continue
                # Pick the first neighbor
                w = next(iter(remaining[v]))
                part.append([[v], [w]])
                remaining[v].discard(w)
                remaining[w].discard(v)
                found = True
                break
            if not found:
                break
            continue
        L, R = result
        for a in L:
            for b in R:
                remaining[a].discard(b)
                remaining[b].discard(a)
        for v in list(remaining.keys()):
            if not remaining[v]:
                del remaining[v]
        part.append([list(L), list(R)])
    return part
>>>>>>> REPLACE

**Edit 2: Improve the deadline to be safer (e.g., 60s instead of 80s) and ensure we return a valid result even if search fails**

The current `construct` sets deadline to 80s. The limit is 90s. This is tight. Let's reduce it to 60s to be safe. Also, the fallback `best = _greedy_init()` might fail if `_greedy_init` has issues, but with the fix above, it should be robust.

