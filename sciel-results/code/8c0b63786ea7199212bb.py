"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
import random

DEADLINE = 115  # seconds, safety margin before 120s limit


def _generate_weight_w_vectors(w, D):
    """Generate all nonzero integer vectors with exactly w nonzero entries, each ±1."""
    from itertools import combinations
    vectors = []
    for positions in combinations(range(D), w):
        for signs in product([1, -1], repeat=w):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
    return vectors


def _greedy_select(candidates, max_dot, time_budget):
    """Greedy selection: pick vectors maintaining pairwise dot <= max_dot."""
    selected = []
    candidate_indices = list(range(len(candidates)))
    random.shuffle(candidate_indices)
    
    start = time.time()
    
    for idx in candidate_indices:
        if time.time() - start > time_budget:
            break
        v = candidates[idx]
        compatible = True
        for s in selected:
            if np.dot(v, s) > max_dot:
                compatible = False
                break
        if compatible:
            selected.append(v)
    
    return selected


def _try_weight(w, D, time_budget):
    """Try to build a large kissing set from weight-w vectors."""
    if w < 2:
        return None
    
    max_dot = w // 2  # floor(w/2)
    
    # Generate candidates
    candidates = _generate_weight_w_vectors(w, D)
    
    if not candidates:
        return None
    
    # Greedy selection
    selected = _greedy_select(candidates, max_dot, time_budget)
    
    if not selected:
        return None
    
    return np.array(selected, dtype=np.int64)


def construct():
    start_time = time.time()
    best_result = None
    best_count = 0
    
    # Try weight-4 vectors (should give much better than 220)
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        result = _try_weight(4, D, remaining - 5)
        if result is not None:
            best_result = result
            best_count = len(result)
    
    # Try weight-5 vectors
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        result = _try_weight(5, D, remaining - 5)
        if result is not None and len(result) > best_count:
            best_result = result
            best_count = len(result)
    
    # Try weight-6 vectors
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        result = _try_weight(6, D, remaining - 5)
        if result is not None and len(result) > best_count:
            best_result = result
            best_count = len(result)
    
    # Fallback: use the original weight-2 construction if nothing better
    if best_result is None or best_count < 220:
        from itertools import combinations
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_result = np.array(vectors, dtype=np.int64)
    
    return best_result
# EVOLVE-BLOCK-END
