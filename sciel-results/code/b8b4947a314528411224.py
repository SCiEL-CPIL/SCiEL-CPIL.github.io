"""Greedy maximum independent set solver for kissing number in R^11."""
import time
import numpy as np

D = 11
DEADLINE = 120.0
SAFETY = 5.0


def _build_pool(max_coord=2):
    """Build a pool of nonzero integer vectors with small coordinates."""
    pool = []
    seen = set()
    # Generate all nonzero integer vectors with coords in [-max_coord, max_coord]
    # but limit total count to keep the conflict graph manageable.
    # We prioritize low-norm vectors since they allow denser packing.
    import itertools
    # Generate by norm to prioritize low-norm vectors
    from collections import defaultdict
    by_norm = defaultdict(list)
    for coords in itertools.product(range(-max_coord, max_coord + 1), repeat=D):
        v = coords
        if all(c == 0 for c in v):
            continue
        # Canonicalize: avoid duplicates up to sign and permutation is NOT valid here
        # since vectors are directional. But we can avoid trivial sign duplicates
        # to reduce pool size. Actually, sign matters for distance, so keep all.
        # To limit pool size, use max_coord=1 first, then 2.
        n2 = sum(c * c for c in v)
        by_norm[n2].append(v)
    
    # Sort by norm and take up to a cap
    cap = 20000
    for n2 in sorted(by_norm.keys()):
        for v in by_norm[n2]:
            if len(pool) >= cap:
                break
            pool.append(v)
        if len(pool) >= cap:
            break
    return pool


def _norm_sq(v):
    return int(np.dot(v, v))


def _dist_sq(a, b):
    d = a - b
    return int(np.dot(d, d))


def construct():
    t0 = time.time()
    
    # Build pool of candidate vectors
    pool = _build_pool(max_coord=2)
    n = len(pool)
    
    # Compute norms squared
    norms = np.array([_norm_sq(v) for v in pool], dtype=np.int64)
    
    # Maximum norm squared
    max_norm_sq = int(norms.max())
    
    # Build conflict graph: two vectors conflict if dist^2 < max_norm_sq
    # We want an independent set: all pairwise distances >= max_norm_sq
    # Note: the constraint is min pairwise distance >= max norm (not max norm squared)
    # So dist >= max_norm, i.e., dist^2 >= max_norm_sq
    
    # For greedy MIS, we need to check conflicts.
    # With n up to 20000, full conflict matrix is 20000^2 = 400M entries - too large.
    # Use a smarter approach: for each vector, check conflicts incrementally.
    
    # Greedy approach: sort vectors by some heuristic, add if compatible with all selected.
    # Heuristic: prefer low-norm vectors (they have smaller max_norm, making constraint easier)
    # But max_norm is global (max over selected set). This is tricky.
    
    # Alternative: iterate with a fixed threshold. Start with threshold = smallest norm^2,
    # build MIS, then the actual max norm of selected set determines validity.
    
    # Better: greedy that tracks current max norm.
    # Sort by norm ascending. Add vector if dist to all selected >= current_max_norm.
    # But adding a higher-norm vector increases the constraint for all future vectors.
    
    # Let's use a different strategy: 
    # 1. Start with the set of all norm^2 = 2 vectors (the seed construction).
    # 2. Try to add more vectors from higher norms.
    
    # Actually, let's use a cleaner greedy:
    # Sort candidates by norm ascending.
    # Maintain selected set S and current_max_norm_sq.
    # For each candidate v in order:
    #   if dist(v, u)^2 >= current_max_norm_sq for all u in S, AND norm(v)^2 >= current_max_norm_sq:
    #     Wait, the constraint is: min pairwise dist >= max norm.
    #     max norm is max over all vectors in the final set.
    #     So when we add v, the new max_norm = max(old_max_norm, norm(v)).
    #     We need: for all u in S, dist(v,u) >= new_max_norm
    #     AND for all u in S, dist(u,v) >= new_max_norm (same thing)
    #     But also, the existing pairwise distances in S must still be >= new_max_norm.
    #     If norm(v) > old_max_norm, then new_max_norm = norm(v), and we need to recheck all pairs in S.
    
    # This is complex. Let's simplify: use a fixed threshold approach.
    # Try threshold T = max_norm_sq. Build MIS with constraint dist^2 >= T.
    # The selected set will have max_norm <= sqrt(T) if all vectors have norm^2 <= T.
    # But we need min_dist >= max_norm, so if all vectors have norm^2 <= T and min_dist^2 >= T,
    # then min_dist >= sqrt(T) >= max_norm. Valid!
    
    # So: pick T, filter pool to vectors with norm^2 <= T, build MIS with dist^2 >= T.
    # Try multiple T values and take the best.
    
    best_set = []
    best_score = 0
    
    # Candidate thresholds: use norms that appear in the pool
    unique_norms = sorted(set(int(x) for x in norms))
    
    for T in unique_norms:
        if time.time() - t0 > DEADLINE - SAFETY:
            break
        
        # Filter to vectors with norm^2 <= T
        mask = norms <= T
        candidates = [pool[i] for i in range(n) if mask[i]]
        if not candidates:
            continue
        
        # Greedy MIS: sort by norm ascending, add if compatible
        candidates.sort(key=lambda v: _norm_sq(v))
        
        selected = []
        for v in candidates:
            if time.time() - t0 > DEADLINE - SAFETY:
                break
            ok = True
            for u in selected:
                if _dist_sq(v, u) < T:
                    ok = False
                    break
            if ok:
                selected.append(v)
        
        if len(selected) > best_score:
            best_score = len(selected)
            best_set = selected[:]
    
    # Also try the original construction as a baseline
    if best_score == 0:
        import itertools
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_set = vectors
        best_score = len(vectors)
    
    return np.array(best_set, dtype=np.int64)
