"""Kissing number in dimension 11 - find max set of integer vectors with min dist >= max norm."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def _norm_sq(v):
    return int(np.dot(v, v))

def _is_valid_set(vectors):
    """Check if a set of vectors satisfies: min pairwise dist >= max norm."""
    if len(vectors) <= 1:
        return True
    norms_sq = [np.dot(v, v) for v in vectors]
    max_norm_sq = max(norms_sq)
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            diff = vectors[i] - vectors[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < max_norm_sq:
                return False
    return True

def _check_pair(v1, v2, max_norm_sq):
    """Check if adding v2 to a set with max_norm_sq is valid with v1."""
    diff = v1 - v2
    dist_sq = np.dot(diff, diff)
    return dist_sq >= max_norm_sq

def _generate_candidates(max_coord=2, max_norm_sq=6):
    """Generate candidate integer vectors with small norms."""
    candidates = []
    # Generate all vectors with coordinates in range [-max_coord, max_coord]
    # and norm_sq <= max_norm_sq, excluding zero
    for coords in itertools.product(range(-max_coord, max_coord+1), repeat=D):
        v = np.array(coords, dtype=np.int64)
        ns = np.dot(v, v)
        if ns > 0 and ns <= max_norm_sq:
            candidates.append(v)
    return candidates

def _greedy_construct(candidates, max_norm_sq, deadline):
    """Greedy construction: add vectors one by one if valid."""
    selected = []
    selected_set = set()
    
    for v in candidates:
        if time.time() > deadline:
            break
        # Check against all selected vectors
        valid = True
        for s in selected:
            diff = s - v
            if np.dot(diff, diff) < max_norm_sq:
                valid = False
                break
        if valid:
            selected.append(v)
    
    return selected

def _local_search(selected, candidates, max_norm_sq, deadline, iterations=50):
    """Try to improve by swapping or adding vectors."""
    selected = list(selected)
    selected_set = set(tuple(v) for v in selected)
    
    for _ in range(iterations):
        if time.time() > deadline:
            break
        improved = False
        # Try adding each candidate
        for v in candidates:
            if time.time() > deadline:
                break
            vt = tuple(v)
            if vt in selected_set:
                continue
            valid = True
            for s in selected:
                diff = s - v
                if np.dot(diff, diff) < max_norm_sq:
                    valid = False
                    break
            if valid:
                selected.append(v)
                selected_set.add(vt)
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add another
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            removed = selected[i]
            remaining = selected[:i] + selected[i+1:]
            for v in candidates:
                vt = tuple(v)
                if vt == tuple(removed):
                    continue
                if vt in selected_set:
                    continue
                valid = True
                for s in remaining:
                    diff = s - v
                    if np.dot(diff, diff) < max_norm_sq:
                        valid = False
                        break
                if valid:
                    selected = remaining + [v]
                    selected_set.discard(tuple(removed))
                    selected_set.add(vt)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with the seed solution: all ±1 pairs
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vectors.append(np.array(v, dtype=np.int64))
    
    # Try to improve with norm_sq = 3 vectors
    # Generate candidates with norm_sq = 3
    candidates_ns3 = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i], v[j], v[k] = si, sj, sk
                            candidates_ns3.append(np.array(v, dtype=np.int64))
    
    # Also try norm_sq = 2 candidates (same as seed)
    candidates_ns2 = list(seed_vectors)
    
    # Strategy: work with max_norm_sq = 2 first (seed), then try mixing
    # Actually, let's try a different approach:
    # Use vectors of norm_sq = 2 and norm_sq = 3 together
    # The constraint: for any two vectors u, v, |u-v|^2 >= max(|u|^2, |v|^2)
    
    # Let's try: all vectors with norm_sq in {2, 3}
    # For two norm-2 vectors: need |u-v|^2 >= 2, i.e., u·v <= 0
    # For two norm-3 vectors: need |u-v|^2 >= 3, i.e., 6 - 2u·v >= 3, u·v <= 1.5, so u·v <= 1
    # For norm-2 and norm-3: need |u-v|^2 >= 3 (max is 3), i.e., 5 - 2u·v >= 3, u·v <= 1
    
    # Generate all candidates with norm_sq = 2 or 3
    all_candidates = candidates_ns2 + candidates_ns3
    
    # Use greedy with max_norm_sq = 3 (the maximum possible in our set)
    # But we need to be careful: the actual max norm in the selected set determines the constraint
    # Let's just use max_norm_sq = 3 as a conservative bound
    
    # Greedy construction
    selected = _greedy_construct(all_candidates, 3, deadline)
    
    # Local search to improve
    if time.time() < deadline:
        selected = _local_search(selected, all_candidates, 3, deadline)
    
    # Verify and return
    result = np.array(selected, dtype=np.int64)
    
    # Final verification
    if len(result) > 1:
        norms_sq = np.array([np.dot(v, v) for v in result])
        max_ns = int(np.max(norms_sq))
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                diff = result[i] - result[j]
                if np.dot(diff, diff) < max_ns:
                    # Shouldn't happen, but if it does, fall back to seed
                    return np.array(seed_vectors, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
