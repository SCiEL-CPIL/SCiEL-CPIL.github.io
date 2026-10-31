"""Improved program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations


def _validate(vectors):
    """Check that all vectors are nonzero, and min pairwise distance >= max norm."""
    if len(vectors) == 0:
        return False
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        return False
    max_norm = np.max(norms)
    # Compute pairwise distances
    n = len(vectors)
    # Use dot product: ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    dot = vectors @ vectors.T
    min_dist_sq = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d2 = dot[i, i] + dot[j, j] - 2.0 * dot[i, j]
            if d2 < min_dist_sq:
                min_dist_sq = d2
    return min_dist_sq >= max_norm * max_norm - 1e-9


def _greedy_select(candidates, max_vectors=None, deadline=None):
    """Greedy selection: repeatedly pick the vector with max min-distance to selected set."""
    n = len(candidates)
    if n == 0:
        return np.empty((0, D), dtype=np.int64)
    
    # Precompute dot products
    dot = candidates @ candidates.T
    norms_sq = np.diag(dot).copy()
    
    selected = []
    # Track min distance from each candidate to selected set
    min_dist_to_selected = np.full(n, np.inf)
    
    start = time.time()
    
    while (max_vectors is None or len(selected) < max_vectors):
        if deadline and (time.time() - start) > deadline:
            break
        # Find candidate with max min_dist_to_selected
        # Exclude already selected
        best_idx = -1
        best_val = -np.inf
        for i in range(n):
            if i in selected:
                continue
            if min_dist_to_selected[i] > best_val:
                best_val = min_dist_to_selected[i]
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        # Update min_dist_to_selected for remaining
        for i in range(n):
            if i not in selected:
                d2 = norms_sq[i] + norms_sq[best_idx] - 2.0 * dot[i, best_idx]
                if d2 < min_dist_to_selected[i]:
                    min_dist_to_selected[i] = d2
    
    return candidates[selected]


def _gen_vectors_with_restarts(n_restarts=5, deadline=None):
    """Generate vectors with multiple restarts of greedy with different orderings."""
    start = time.time()
    best = np.empty((0, D), dtype=np.int64)
    
    # Base pool: all vectors with 1 or 2 nonzero entries (±1)
    # Vectors with one nonzero: D * 2 = 22
    # Vectors with two nonzeros: C(11,2) * 4 = 220
    # Total: 242
    pool = []
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            pool.append(v)
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                pool.append(v)
    
    pool = np.array(pool, dtype=np.int64)
    n_pool = len(pool)
    
    for r in range(n_restarts):
        if deadline and (time.time() - start) > deadline:
            break
        # Shuffle with different seed
        rng = np.random.RandomState(42 + r * 1000)
        order = rng.permutation(n_pool)
        shuffled = pool[order]
        
        remaining = 100 - (time.time() - start)
        sel = _greedy_select(shuffled, deadline=remaining)
        
        if len(sel) > len(best):
            best = sel
    
    return best


def _expand_vectors(base_vectors, max_iter=3, deadline=None):
    """Try to add more vectors by searching for new candidates that fit."""
    start = time.time()
    vectors = list(base_vectors)
    n = len(vectors)
    
    if n == 0:
        return np.array(vectors, dtype=np.int64)
    
    vectors_arr = np.array(vectors, dtype=np.int64)
    norms = np.linalg.norm(vectors_arr, axis=1)
    max_norm = np.max(norms)
    
    # Generate candidate vectors with small integer coordinates
    # Try vectors with entries in {-2, -1, 0, 1, 2} but limit search
    # For efficiency, try vectors with at most 3 nonzeros
    
    added = 0
    for iter_num in range(max_iter):
        if deadline and (time.time() - start) > deadline:
            break
        if not vectors:
            break
        vecs = np.array(vectors, dtype=np.int64)
        max_n = np.max(np.linalg.norm(vecs, axis=1))
        
        # Try to find a new vector that maintains the constraint
        # Search over vectors with small coordinates
        found = False
        # Generate candidates: vectors with at most 3 nonzero entries, values in {-2,-1,1,2}
        candidates = []
        for i in range(D):
            for s in (-2, -1, 1, 2):
                v = [0] * D
                v[i] = s
                candidates.append(v)
        for i, j in combinations(range(D), 2):
            for si in (-2, -1, 1, 2):
                for sj in (-2, -1, 1, 2):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    candidates.append(v)
        for i, j, k in combinations(range(D), 3):
            for si in (-1, 1):
                for sj in (-1, 1):
                    for sk in (-1, 1):
                        v = [0] * D
                        v[i], v[j], v[k] = si, sj, sk
                        candidates.append(v)
        
        cand_arr = np.array(candidates, dtype=np.int64)
        cand_norms = np.linalg.norm(cand_arr, axis=1)
        
        # Filter: norm must be >= current max norm (or at least nonzero)
        # Actually we need: for all existing v, ||cand - v|| >= max(max_norm, ||cand||)
        # This is tricky. Let's just check validity.
        
        for c in cand_arr:
            if deadline and (time.time() - start) > deadline:
                break
            cn = np.linalg.norm(c)
            if cn == 0:
                continue
            # Check against all existing vectors
            valid = True
            for v in vectors:
                d = np.linalg.norm(c - v)
                threshold = max(cn, np.linalg.norm(v))
                if d < threshold - 1e-9:
                    valid = False
                    break
            if valid:
                vectors.append(c)
                added += 1
                found = True
                if deadline and (time.time() - start) > deadline:
                    break
    
    return np.array(vectors, dtype=np.int64)


def construct():
    start = time.time()
    deadline_margin = 20  # seconds before the 120s limit
    
    # Phase 1: Greedy with restarts on the base pool
    remaining = 80 - (time.time() - start)
    vectors = _gen_vectors_with_restarts(n_restarts=10, deadline=remaining)
    
    # Phase 2: Try to expand
    remaining = 120 - deadline_margin - (time.time() - start)
    if remaining > 5:
        vectors = _expand_vectors(vectors, max_iter=5, deadline=remaining)
    
    # Final validation
    if len(vectors) > 0:
        if not _validate(vectors):
            # Fallback to simple known-good set
            simple = []
            for i, j in combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        simple.append(v)
            vectors = np.array(simple, dtype=np.int64)
    
    return vectors
# EVOLVE-BLOCK-END
