"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np

_TIME_LIMIT = 150.0  # hard wall-clock limit
_MARGIN = 5.0        # safety margin


def _score(vectors):
    """Compute the kissing number: max k such that min pairwise dist >= max norm."""
    if len(vectors) == 0:
        return 0
    norms = np.linalg.norm(vectors, axis=1)
    max_norm = norms.max()
    if max_norm == 0:
        return 0
    # Compute all pairwise distances
    k = len(vectors)
    # Use matrix operations for speed
    # dist^2[i,j] = |vi|^2 + |vj|^2 - 2*vi.vj
    sq_norms = np.sum(vectors**2, axis=1)
    gram = vectors @ vectors.T
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = dist_sq.min()
    if min_dist_sq < max_norm**2 - 1e-9:
        return 0
    return k


def _greedy_add(vectors, candidates, max_norm_sq, deadline):
    """Greedily add candidates that maintain min_dist >= max_norm."""
    added = []
    for c in candidates:
        if time.time() > deadline:
            break
        cn_sq = float(c @ c)
        if cn_sq == 0:
            continue
        # Check against existing vectors
        if vectors:
            # dist^2 = |v|^2 + |c|^2 - 2 v.c
            dots = vectors @ c
            dist_sqs = np.sum(vectors**2, axis=1) + cn_sq - 2 * dots
            if dist_sqs.min() < max_norm_sq - 1e-9:
                continue
        # Also check against previously added candidates in this round
        if added:
            dots_a = np.array(added) @ c
            dist_sqs_a = np.sum(np.array(added)**2, axis=1) + cn_sq - 2 * dots_a
            if dist_sqs_a.min() < max_norm_sq - 1e-9:
                continue
        added.append(c)
    return added


def _random_candidates(n, d, max_coord=3, rng=None):
    """Generate random integer vectors with small coordinates."""
    if rng is None:
        rng = np.random.default_rng()
    coords = rng.integers(-max_coord, max_coord + 1, size=(n, d))
    # Ensure nonzero
    zero_mask = np.all(coords == 0, axis=1)
    while np.any(zero_mask):
        idx = np.where(zero_mask)[0]
        for i in idx:
            coords[i] = rng.integers(-max_coord, max_coord + 1, size=d)
            if np.all(coords[i] == 0):
                coords[i, 0] = 1
        zero_mask = np.all(coords == 0, axis=1)
    return coords


def _local_search(vectors, d, max_coord=4, n_iter=200, deadline=None):
    """Try to add more vectors via local search around existing ones."""
    vectors = list(vectors)
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        if deadline and time.time() > deadline:
            break
        # Generate a batch of perturbed candidates
        batch = []
        for _ in range(50):
            # Pick a random base vector and perturb
            if vectors:
                base = vectors[rng.integers(len(vectors))]
            else:
                base = np.zeros(d, dtype=np.int64)
            c = base + rng.integers(-2, 3, size=d)
            c = np.clip(c, -max_coord, max_coord)
            if np.any(c != 0):
                batch.append(c)
        
        if not batch:
            continue
        
        # Try to greedily add from this batch
        norms = np.array([float(v @ v) for v in vectors]) if vectors else np.array([])
        max_norm_sq = norms.max() if len(norms) > 0 else 1.0
        
        added = _greedy_add(np.array(vectors, dtype=np.int64) if vectors else np.empty((0, d)),
                           np.array(batch, dtype=np.int64), max_norm_sq, deadline)
        vectors.extend(added)
    return vectors


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT - _MARGIN
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Start with the seed configuration (all ±1,±1 pairs)
    vectors = []
    for i, j in [(i, j) for i in range(D) for j in range(i+1, D)]:
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    current_score = _score(vectors)
    
    # Phase 2: Try to add more vectors via random search + greedy
    # Generate diverse random candidates
    all_candidates = []
    
    # Small coordinate vectors
    for max_coord in [1, 2, 3]:
        n_gen = 2000
        cands = _random_candidates(n_gen, D, max_coord, rng)
        all_candidates.append(cands)
    
    all_candidates = np.vstack(all_candidates)
    
    # Remove duplicates
    # Use a set of tuples for dedup
    seen = set()
    unique_cands = []
    for c in all_candidates:
        t = tuple(c)
        if t not in seen and any(c):
            seen.add(t)
            unique_cands.append(c)
    
    unique_cands = np.array(unique_cands, dtype=np.int64)
    
    # Compute max norm of current vectors
    norms = np.linalg.norm(vectors, axis=1)
    max_norm = norms.max()
    max_norm_sq = max_norm**2
    
    # Greedily add candidates
    added = _greedy_add(vectors, unique_cands, max_norm_sq, deadline)
    if added:
        vectors = np.vstack([vectors, np.array(added, dtype=np.int64)])
        current_score = _score(vectors)
    
    # Phase 3: Local search - try to find more vectors by perturbing existing ones
    vectors_list = [vectors[i] for i in range(len(vectors))]
    vectors_list = _local_search(vectors_list, D, max_coord=4, n_iter=500, deadline=deadline)
    
    if len(vectors_list) > len(vectors):
        vectors = np.array(vectors_list, dtype=np.int64)
        current_score = _score(vectors)
    
    # Phase 4: Try adding vectors with slightly larger coordinates
    if time.time() < deadline:
        for max_coord in [4, 5, 6]:
            if time.time() > deadline:
                break
            cands = _random_candidates(3000, D, max_coord, rng)
            
            # Dedup against existing
            existing_set = set(tuple(v) for v in vectors)
            new_cands = []
            for c in cands:
                t = tuple(c)
                if t not in existing_set and any(c):
                    existing_set.add(t)
                    new_cands.append(c)
            
            if new_cands:
                new_cands = np.array(new_cands, dtype=np.int64)
                norms = np.linalg.norm(vectors, axis=1)
                max_norm_sq = norms.max()**2
                added = _greedy_add(vectors, new_cands, max_norm_sq, deadline)
                if added:
                    vectors = np.vstack([vectors, np.array(added, dtype=np.int64)])
                    current_score = _score(vectors)
    
    # Phase 5: Final local search
    if time.time() < deadline:
        vectors_list = [vectors[i] for i in range(len(vectors))]
        vectors_list = _local_search(vectors_list, D, max_coord=5, n_iter=1000, deadline=deadline)
        if len(vectors_list) > len(vectors):
            vectors = np.array(vectors_list, dtype=np.int64)
            current_score = _score(vectors)
    
    # Ensure we return a valid configuration
    final_score = _score(vectors)
    if final_score == 0 and len(vectors) > 0:
        # Fallback to seed configuration
        vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vectors.append(v)
        vectors = np.array(vectors, dtype=np.int64)
    
    return vectors
# EVOLVE-BLOCK-END
