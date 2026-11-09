# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = 590.0  # safety margin before 630s limit
    
    best_points = None
    best_k = 2
    best_set = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    
    def try_update(points, k):
        nonlocal best_points, best_k, best_set
        if k > best_k:
            best_k = k
            best_points = points
            best_set = list(points)
    
    # Strategy: For each weight w (number of non-zero entries), generate all
    # vectors with entries in {-1, 0, 1} and weight w. All have norm^2 = w.
    # Need pairwise dot products <= floor(w/2) for mutual compatibility.
    # Find a large independent set in the conflict graph.
    # 
    # Key structural insight: For weight-2 vectors, the conflict graph decomposes
    # by support. Vectors with different supports are always compatible (dot <= 1).
    # For each support {i,j}, the 4 sign patterns form a small conflict graph where
    # we can pick at most 3. This gives 55*3 = 165 vectors with no cross-support
    # conflicts. We can verify this is optimal for weight-2.
    #
    # For weight-3, vectors with supports sharing 2 positions can conflict.
    # This requires a more careful search.
    
    # First, try the standard basis: 22 vectors (e_i and -e_i), norm^2 = 1
    basis_vectors = []
    for i in range(d):
        v1 = np.zeros(d, dtype=np.int64)
        v1[i] = 1
        v2 = np.zeros(d, dtype=np.int64)
        v2[i] = -1
        basis_vectors.append(v1)
        basis_vectors.append(v2)
    try_update(np.array(basis_vectors, dtype=np.int64), len(basis_vectors))
    
    # For each weight w from 2 to 11, try to find a large compatible set
    for w in range(2, 12):
        if time.time() > deadline - 30:
            break
        
        max_dot_allowed = w // 2  # floor(w/2)
        
        # Generate all weight-w vectors with entries in {-1, 0, 1}
        vecs = []
        for positions in itertools.combinations(range(d), w):
            for signs in itertools.product([-1, 1], repeat=w):
                v = np.zeros(d, dtype=np.int64)
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                vecs.append(v)
        
        n = len(vecs)
        if n == 0:
            continue
        
        vecs_arr = np.array(vecs, dtype=np.int64)
        
        # Compute all pairwise dot products
        dots = vecs_arr @ vecs_arr.T
        
        # Conflict: dot > max_dot_allowed (since dot is integer, conflict iff dot >= max_dot_allowed + 1)
        conflict = (dots > max_dot_allowed)
        np.fill_diagonal(conflict, False)
        
        # For weight-2, we know the structure: decompose by support
        if w == 2:
            # For each support {i,j}, pick 3 compatible sign patterns
            # Patterns: (1,1), (1,-1), (-1,1) are mutually compatible
            # (dot products: 0, 0, -1 all <= 1)
            selected_indices = set()
            # Map from (i,j) to list of indices with that support
            support_to_indices = defaultdict(list)
            for idx in range(n):
                nz = np.nonzero(vecs_arr[idx])[0]
                if len(nz) == 2:
                    key = (int(nz[0]), int(nz[1]))
                    support_to_indices[key].append(idx)
            
            for key, indices in support_to_indices.items():
                # Pick 3 out of 4 that are mutually compatible
                # The 4 patterns are: ++, +-, -+, --
                # ++ conflicts with -- (dot = -2 < -1? No, we check dot > 1, so -2 is fine)
                # Wait: conflict iff dot > 1. So:
                # ++ . -- = -2, not > 1, so compatible!
                # ++ . +- = 0, compatible
                # ++ . -+ = 0, compatible
                # +- . -+ = -1, compatible
                # +- . -- = -1, compatible
                # -+ . -- = 1, compatible
                # So actually all 4 are mutually compatible for w=2!
                # Let me re-check: dot > 1 means dot >= 2.
                # Max dot for same support with same signs: 2. 2 > 1, conflict!
                # So ++ and ++ conflict (but they're the same vector, handled by diagonal)
                # ++ and +-: dot = 0. OK.
                # ++ and -+: dot = 0. OK.
                # ++ and --: dot = -2. OK.
                # So for a given support, all 4 sign patterns are mutually compatible
                # EXCEPT that a vector conflicts with itself (diagonal).
                # So we can pick all 4!
                # Wait, but the task says "points must be distinct", so we can't pick the same vector twice.
                # But the 4 sign patterns are distinct vectors. So we can pick all 4.
                for idx in indices:
                    selected_indices.add(idx)
            
            if len(selected_indices) > best_k:
                selected_arr = vecs_arr[list(selected_indices)]
                try_update(selected_arr, len(selected_indices))
        
        else:
            # For other weights, use greedy with multiple restarts
            best_selected = set()
            
            for attempt in range(50):
                if time.time() > deadline - 30:
                    break
                
                if attempt == 0:
                    order = np.arange(n)
                else:
                    order = np.random.permutation(n)
                
                selected = set()
                for idx in order:
                    idx = int(idx)
                    # Check if idx conflicts with any selected
                    compatible = True
                    for sel in selected:
                        if conflict[idx, sel]:
                            compatible = False
                            break
                    if compatible:
                        selected.add(idx)
                
                if len(selected) > len(best_selected):
                    best_selected = selected
            
            k = len(best_selected)
            if k > best_k:
                selected_arr = vecs_arr[list(best_selected)]
                try_update(selected_arr, k)
    
    # Also try a mixed approach: combine weight-2 vectors with some weight-1 vectors
    # But as analyzed, weight-1 vectors (norm^2=1) can't be added to weight-2 set (norm^2=2)
    # because max norm becomes 2, and weight-1 vectors may be too close to weight-2 vectors.
    
    # Try to improve with local search: 1-swap moves
    if best_points is not None and best_k >= 2:
        if time.time() > deadline - 30:
            return _finalize(best_points, d)
        
        # For the best set found, try to swap in/out vectors
        # This is complex, so we'll rely on the greedy above
        
        pass
    
    return _finalize(best_points, d)


def _finalize(points, d):
    if points is None or len(points) < 2:
        return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)
    
    result = np.around(points).astype(np.int64)
    k = len(result)
    if k >= 2 and result.shape[1] == d:
        norms_sq = np.sum(result ** 2, axis=1)
        if np.all(norms_sq > 0):
            max_norm_sq = np.max(norms_sq)
            min_dist_sq = np.inf
            for i in range(k):
                for j in range(i + 1, k):
                    diff = result[i] - result[j]
                    dist_sq = np.sum(diff ** 2)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
            if min_dist_sq >= max_norm_sq:
                return result
    
    return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)

# EVOLVE-BLOCK-END
