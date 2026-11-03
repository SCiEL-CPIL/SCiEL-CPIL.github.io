"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

D = 11
DEADLINE = 140.0

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time

D = 11
DEADLINE = 140.0


def _norm_sq(v):
    return int(np.dot(v, v))


def _build_candidates():
    """Build diverse integer vectors in R^11 with small norms."""
    cands = []
    
    # Norm^2 = 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            cands.append(v)
    
    # Norm^2 = 2
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                cands.append(v)
    
    # Norm^2 = 4: four ±1 entries
    for idx in combinations(range(D), 4):
        i, j, k, l = idx
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        cands.append(v)
    
    # Norm^2 = 4: one ±2 entry
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            cands.append(v)
    
    return cands


def _greedy_build(candidates, norms_sq, max_time=130.0):
    """Greedy construction with multiple orderings, keep best."""
    start = time.time()
    n = len(candidates)
    best_clique = []
    
    # Precompute compatibility as bitsets
    compat = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = candidates[i] - candidates[j]
            dist_sq = int(np.dot(diff, diff))
            threshold = max(norms_sq[i], norms_sq[j])
            if dist_sq >= threshold:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    # Try different orderings
    orderings = [
        sorted(range(n), key=lambda i: (norms_sq[i], i)),
        sorted(range(n), key=lambda i: (norms_sq[i], -bin(compat[i]).count('1'), i)),
        sorted(range(n), key=lambda i: -bin(compat[i]).count('1')),
    ]
    
    for order in orderings:
        if time.time() - start > max_time:
            break
        
        clique = []
        in_clique = set()
        
        for i in order:
            if time.time() - start > max_time:
                break
            ok = True
            for j in in_clique:
                if not (compat[i] >> j) & 1:
                    ok = False
                    break
            if ok:
                clique.append(i)
                in_clique.add(i)
        
        if len(clique) > len(best_clique):
            best_clique = list(clique)
    
    # Local improvement: try to add vectors from remaining
    remaining = [i for i in range(n) if i not in set(best_clique)]
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        for i in remaining[:]:
            ok = True
            for j in best_clique:
                if not (compat[i] >> j) & 1:
                    ok = False
                    break
            if ok:
                best_clique.append(i)
                remaining.remove(i)
                improved = True
    
    return best_clique


def construct():
    start_time = time.time()
    
    candidates = _build_candidates()
    n = len(candidates)
    norms_sq = [_norm_sq(v) for v in candidates]
    
    clique = _greedy_build(candidates, norms_sq, max_time=DEADLINE - 10)
    
    result = np.array([candidates[i] for i in clique], dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END

