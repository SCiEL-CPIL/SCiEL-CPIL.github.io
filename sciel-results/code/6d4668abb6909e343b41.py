# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that
    their maximum squared norm is smaller than their minimum pairwise squared distance,
    aiming to maximize the number of points.

    Uses ±1 vectors (norm² = 11) and requires pairwise squared distance > 11,
    which for ±1 vectors means pairwise inner product ≤ 10 (Hamming distance ≥ 2).
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s kill

    best_points = np.array([np.ones(d), -np.ones(d)]).astype(np.int64)
    best_k = 2

    rng = np.random.default_rng(42)

    def is_valid_set(pts):
        """Check if set of ±1 vectors satisfies min pairwise dist² > max norm² (=11)."""
        n = len(pts)
        if n < 2:
            return False
        # All ±1 vectors have norm² = 11
        # Need pairwise dist² >= 12, i.e., ip <= 10
        for i in range(n):
            for j in range(i + 1, n):
                ip = int(np.dot(pts[i], pts[j]))
                if ip > 10:
                    return False
        return True

    def greedy_build(candidate_pool, order):
        """Greedily add candidates in given order if they satisfy the constraint."""
        selected = []
        for idx in order:
            v = candidate_pool[idx]
            ok = True
            for s in selected:
                ip = int(np.dot(v, s))
                if ip > 10:
                    ok = False
                    break
            if ok:
                selected.append(v)
        return selected

    # Phase 1: Try structured constructions
    # 1a) All ±1 vectors with even weight (or some structured subset)
    # Generate a pool of random ±1 vectors and use greedy with multiple orderings
    
    n_pool = 2000
    pool = rng.choice([-1, 1], size=(n_pool, d)).astype(np.int64)
    
    # Add some structured vectors
    structured = []
    # All ones, all negatives
    structured.append(np.ones(d, dtype=np.int64))
    structured.append(-np.ones(d, dtype=np.int64))
    # Vectors with one -1 and rest +1
    for i in range(d):
        v = np.ones(d, dtype=np.int64)
        v[i] = -1
        structured.append(v)
        v2 = -np.ones(d, dtype=np.int64)
        v2[i] = 1
        structured.append(v2)
    structured = np.array(structured)
    
    # Combine pool
    full_pool = np.vstack([structured, pool])
    n_full = len(full_pool)
    
    # Try multiple greedy orderings
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart % 3 == 0:
            order = list(range(n_full))
            rng.shuffle(order)
        elif restart % 3 == 1:
            # Sort by norm (all same for ±1, so just shuffle with bias toward structured)
            order = list(range(n_full))
            rng.shuffle(order)
            # Put structured first
            structured_indices = list(range(len(structured)))
            random_indices = list(range(len(structured), n_full))
            rng.shuffle(random_indices)
            order = structured_indices + random_indices
        else:
            # Descending: start with random, then add structured
            order = list(range(n_full))
            rng.shuffle(order)
            structured_indices = list(range(len(structured)))
            random_indices = list(range(len(structured), n_full))
            rng.shuffle(random_indices)
            order = random_indices + structured_indices
        
        selected = greedy_build(full_pool, order)
        k = len(selected)
        
        if k > best_k:
            best_k = k
            best_points = np.array(selected, dtype=np.int64)
    
    # Phase 2: Local search / improvement on best found
    if time.time() < deadline - 60 and best_k > 2:
        # Try to improve by swapping or adding
        current = best_points.tolist()
        current_set = set(tuple(x) for x in current)
        
        # Try adding vectors not in current set
        for _ in range(200):
            if time.time() > deadline - 60:
                break
            v = rng.choice([-1, 1], size=d).astype(np.int64)
            if tuple(v.tolist()) in current_set:
                continue
            ok = True
            for s in current:
                ip = int(np.dot(v, np.array(s)))
                if ip > 10:
                    ok = False
                    break
            if ok:
                current.append(v.tolist())
                current_set.add(tuple(v.tolist()))
        
        if len(current) > best_k:
            best_k = len(current)
            best_points = np.array(current, dtype=np.int64)
    
    # Final validation
    result = best_points
    result = np.around(result).astype(np.int64)
    
    # Verify
    k = len(result)
    if k < 2:
        return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)
    
    max_norm_sq = 0
    for i in range(k):
        ns = int(np.dot(result[i], result[i]))
        max_norm_sq = max(max_norm_sq, ns)
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = result[i] - result[j]
            ds = int(np.dot(diff, diff))
            min_dist_sq = min(min_dist_sq, ds)
    
    if min_dist_sq <= max_norm_sq:
        # Fallback to valid pair
        return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)
    
    return result


# EVOLVE-BLOCK-END
