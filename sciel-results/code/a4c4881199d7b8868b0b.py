import time
from itertools import combinations

def _is_2_colorable_fast(edges, n=9):
    """Check if 3-uniform hypergraph is 2-colorable. Returns True if 2-colorable."""
    m = len(edges)
    if m == 0:
        return True
    # Brute force 2^n for n=9 is 512
    for mask in range(1 << n):
        ok = True
        for edge in edges:
            c0 = (mask >> edge[0]) & 1
            c1 = (mask >> edge[1]) & 1
            c2 = (mask >> edge[2]) & 1
            if c0 == c1 == c2:
                ok = False
                break
        if ok:
            return True
    return False

def _is_3_colorable_fast(edges, n=9):
    """Check if 3-uniform hypergraph is 3-colorable. Returns True if 3-colorable."""
    # Backtracking
    edges_sorted = [tuple(sorted(e)) for e in edges]
    
    def backtrack(pos, ca):
        if pos == n:
            for edge in edges_sorted:
                if ca[edge[0]] == ca[edge[1]] == ca[edge[2]]:
                    return False
            return True
        for c in range(3):
            ca[pos] = c
            # Prune
            prune = False
            for edge in edges_sorted:
                if pos in edge:
                    e0, e1, e2 = edge
                    if e0 <= pos and e1 <= pos and e2 <= pos:
                        if ca[e0] == ca[e1] == ca[e2]:
                            prune = True
                            break
            if not prune:
                if backtrack(pos + 1, ca):
                    return True
            ca[pos] = -1
        return False
    
    return backtrack(0, [-1] * n)

def _find_min_edges(deadline):
    n = 9
    all_edges = list(combinations(range(n), 3))
    
    for k in range(1, 10):
        if time.time() > deadline:
            return None
        
        if k <= 3:
            # C(84,k) is small
            for combo in combinations(all_edges, k):
                edges = [list(e) for e in combo]
                if _is_3_colorable_fast(edges, n) and not _is_2_colorable_fast(edges, n):
                    return edges
        elif k == 4:
            # C(84,4) ~ 1.9M, need to be fast
            count = 0
            for combo in combinations(all_edges, 4):
                if time.time() > deadline:
                    return None
                edges = [list(e) for e in combo]
                # Quick check: 3-colorable first (usually faster to reject)
                if not _is_3_colorable_fast(edges, n):
                    continue
                if not _is_2_colorable_fast(edges, n):
                    return edges
                count += 1
                if count > 500000:
                    break
        else:
            # Random sampling
            import random
            rng = random.Random(k * 7919)
            max_tries = 20000
            for _ in range(max_tries):
                if time.time() > deadline:
                    return None
                combo = rng.sample(all_edges, k)
                edges = [list(e) for e in combo]
                if _is_3_colorable_fast(edges, n) and not _is_2_colorable_fast(edges, n):
                    return edges
    return None

def _find_valid_coloring(edges, n=9):
    edges_sorted = [tuple(sorted(e)) for e in edges]
    
    def backtrack(pos, ca):
        if pos == n:
            for edge in edges_sorted:
                if ca[edge[0]] == ca[edge[1]] == ca[edge[2]]:
                    return None
            return ca[:]
        for c in range(3):
            ca[pos] = c
            prune = False
            for edge in edges_sorted:
                if pos in edge:
                    e0, e1, e2 = edge
                    if e0 <= pos and e1 <= pos and e2 <= pos:
                        if ca[e0] == ca[e1] == ca[e2]:
                            prune = True
                            break
            if not prune:
                res = backtrack(pos + 1, ca)
                if res is not None:
                    return res
            ca[pos] = -1
        return None
    
    return backtrack(0, [-1] * n)

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    result = _find_min_edges(deadline)
    
    if result is not None:
        edges = result
        colors = _find_valid_coloring(edges, 9)
        if colors is None:
            colors = [0, 0, 1, 1, 2, 0, 0, 0, 0]
        return {'edges': [list(e) for e in edges], 'colors': [int(c) for c in colors]}
    else:
        edges = [list(e) for e in combinations(range(5), 3)]
        colors = [0, 0, 1, 1, 2, 0, 0, 0, 0]
        return {'edges': edges, 'colors': colors}
