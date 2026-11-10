# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def _e8_minimal_vectors():
    """Generate the 240 minimal vectors of the E8 lattice."""
    vecs = []
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) - 112 vectors
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0.0] * 8
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
    # Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs - 128 vectors
    for bits in product([1, -1], repeat=8):
        if sum(1 for b in bits if b == -1) % 2 == 0:
            vecs.append([b * 0.5 for b in bits])
    return np.array(vecs, dtype=np.float64)


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points forming a kissing configuration.
    
    Strategy: Use E8 lattice minimal vectors embedded in R^11 as a base, then perform
    local search by adding integer vectors with small squared norms that maintain
    the minimum squared distance constraint.
    """
    start_time = time.time()
    deadline = start_time + 550  # 550s deadline, 50s safety margin
    
    d = 11
    
    # Get E8 minimal vectors and embed in R^11 (pad with zeros)
    e8_vecs = _e8_minimal_vectors()  # shape (240, 8)
    e8_in_11 = np.zeros((len(e8_vecs), d), dtype=np.float64)
    e8_in_11[:, :8] = e8_vecs
    
    # Scale to integers: multiply by 2 to get integer coordinates
    # E8 minimal vectors have norm squared = 2, so scaled by 2 gives norm squared = 8
    base_points = (e8_in_11 * 2).astype(np.int64)
    
    # Remove zero vector if present
    mask = np.any(base_points != 0, axis=1)
    base_points = base_points[mask]
    
    # Remove duplicates
    base_points = np.unique(base_points, axis=0)
    
    # Generate additional candidate integer vectors with small norms
    candidates = []
    
    # Vectors with exactly two non-zero entries, each ±1 (norm squared = 2)
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Vectors with exactly two non-zero entries, one ±2 and one 0... no, let's do (±2, ±2, 0...)
    # Actually, let's add vectors with one non-zero entry ±1 (norm squared = 1)
    for i in range(d):
        for si in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = si
            candidates.append(v)
    
    # Vectors with all entries ±1 (norm squared = 11)
    for bits in product([1, -1], repeat=d):
        candidates.append(np.array(bits, dtype=np.int64))
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Combine with base points
    all_candidates = np.vstack([base_points, candidates])
    all_candidates = np.unique(all_candidates, axis=0)
    
    # Remove zero vector
    mask = np.any(all_candidates != 0, axis=1)
    all_candidates = all_candidates[mask]
    
    # Sort by norm (prefer lower norm)
    norms = np.sum(all_candidates ** 2, axis=1)
    order = np.argsort(norms)
    all_candidates = all_candidates[order]
    norms_sorted = norms[order]
    
    # Greedy selection with vectorized distance checks
    selected = []
    selected_arr = None
    current_max_norm = 0
    
    n_candidates = len(all_candidates)
    for i in range(n_candidates):
        if time.time() > deadline - 30:
            break
        
        candidate = all_candidates[i]
        candidate_norm = norms_sorted[i]
        
        if len(selected) < 2:
            selected.append(candidate)
            current_max_norm = max(current_max_norm, candidate_norm)
            continue
        
        # Vectorized check: compute distances from candidate to all selected points
        if selected_arr is None:
            selected_arr = np.array(selected)
        
        # Compute squared distances from candidate to all selected points
        diffs = selected_arr - candidate  # shape (k, d)
        dists = np.sum(diffs ** 2, axis=1)  # shape (k,)
        
        # New max norm would be max(current_max_norm, candidate_norm)
        new_max_norm = max(current_max_norm, candidate_norm)
        
        # Check if all distances are >= new_max_norm
        if np.all(dists >= new_max_norm):
            selected.append(candidate)
            current_max_norm = new_max_norm
            selected_arr = np.array(selected)
    
    # Convert to array
    if selected:
        result = np.array(selected, dtype=np.int64)
    else:
        result = np.array([np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)])
    
    # Verify the result
    def check_valid(points):
        if len(points) < 2:
            return True
        n = len(points)
        max_norm = int(np.max(np.sum(points ** 2, axis=1)))
        min_dist = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = int(np.sum(diff ** 2))
                if dist < min_dist:
                    min_dist = dist
        return min_dist >= max_norm
    
    if not check_valid(result):
        # Fall back to trivial solution
        result = np.array([np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)])
    
    # Ensure shape is (k, 11)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    
    return result


# EVOLVE-BLOCK-END
