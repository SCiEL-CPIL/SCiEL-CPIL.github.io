"""Kissing number in dimension 11 - integer vector construction."""
import time
import numpy as np

D = 11


def _check_valid(vectors):
    """Check if vectors satisfy the distance constraint."""
    if len(vectors) == 0:
        return True
    norms = np.linalg.norm(vectors, axis=1)
    max_norm = norms.max()
    if max_norm == 0:
        return False
    # min pairwise distance must be >= max_norm
    if len(vectors) < 2:
        return True
    # Subsample for speed if large
    if len(vectors) > 300:
        idx = np.random.choice(len(vectors), 300, replace=False)
        v = vectors[idx]
    else:
        v = vectors
    # Compute pairwise distances
    diffs = v[:, np.newaxis, :] - v[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    return min_dist >= max_norm


def _generate_e8_vectors():
    """Generate scaled E8 lattice minimal vectors (240 vectors) in R^8."""
    vecs = []
    # Type 1: two +2/-2, rest 0 (112 vectors)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 8
                    v[i], v[j] = 2 * si, 2 * sj
                    vecs.append(v)
    # Type 2: all +-1 with even number of minus signs (128 vectors)
    for mask in range(256):
        bits = [1 if (mask >> k) & 1 else -1 for k in range(8)]
        if sum(1 for b in bits if b == -1) % 2 == 0:
            vecs.append(bits)
    return np.array(vecs, dtype=np.int64)


def _generate_seed_vectors():
    """Generate initial candidate vectors from known constructions."""
    vecs = []
    
    # E8 scaled by 2, padded to 11D (240 vectors)
    e8 = _generate_e8_vectors()
    for v in e8:
        vecs.append(np.concatenate([v, [0, 0, 0]]))
    
    # Additional vectors in the last 3 coordinates
    for i in range(8, 11):
        for s in (1, -1):
            v = np.zeros(11, dtype=np.int64)
            v[i] = 2 * s
            vecs.append(v)
    
    # Some mixed vectors
    for i in range(8):
        for j in range(8, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(11, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
    
    return np.array(vecs, dtype=np.int64)


def _try_add_vector(existing, candidate):
    """Check if candidate can be added to existing set."""
    if len(existing) == 0:
        return True
    existing_norms = np.linalg.norm(existing, axis=1)
    cand_norm = np.linalg.norm(candidate)
    new_max_norm = max(existing_norms.max(), cand_norm)
    
    # Check distances from candidate to all existing
    diffs = existing - candidate
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    min_dist = dists.min()
    return min_dist >= new_max_norm


def _local_improvement(vectors, time_limit):
    """Try to improve the set by removing bad vectors and adding new ones."""
    start = time.time()
    vectors = list(vectors)
    
    def elapsed():
        return time.time() - start
    
    # Try to remove vectors that allow adding more
    for remove_idx in range(len(vectors)):
        if elapsed() > time_limit:
            break
        remaining = vectors[:remove_idx] + vectors[remove_idx + 1:]
        if not _check_valid(np.array(remaining)):
            continue
        
        # Try to add new vectors
        added = 0
        for _ in range(10):
            if elapsed() > time_limit:
                break
            # Try random small vectors
            for _trial in range(50):
                if elapsed() > time_limit:
                    break
                cand = np.zeros(D, dtype=np.int64)
                n_nonzero = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
                positions = np.random.choice(D, n_nonzero, replace=False)
                for pos in positions:
                    cand[pos] = np.random.choice([-2, -1, 1, 2])
                if _try_add_vector(np.array(remaining), cand):
                    remaining.append(cand)
                    added += 1
        
        if len(remaining) > len(vectors):
            vectors = remaining
    
    return vectors


def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    # Start with known construction
    vectors = _generate_seed_vectors()
    
    # Verify and fix if needed
    if not _check_valid(vectors):
        # Fall back to simpler construction
        vectors = []
        for i in range(D):
            for s in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = s
                vectors.append(v)
    
    best_vectors = list(vectors)
    best_count = len(best_vectors)
    
    # Greedy extension: try to add more vectors
    while time.time() < deadline - 10:
        added_any = False
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            # Generate random candidate
            cand = np.zeros(D, dtype=np.int64)
            n_nonzero = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
            positions = np.random.choice(D, n_nonzero, replace=False)
            for pos in positions:
                cand[pos] = np.random.choice([-2, -1, 1, 2])
            
            if _try_add_vector(np.array(best_vectors), cand):
                best_vectors.append(cand)
                added_any = True
        
        if not added_any:
            break
    
    # Local improvement
    if time.time() < deadline - 5:
        improved = _local_improvement(best_vectors, deadline - 5 - time.time())
        if len(improved) > len(best_vectors):
            best_vectors = improved
    
    return np.array(best_vectors, dtype=np.int64)
