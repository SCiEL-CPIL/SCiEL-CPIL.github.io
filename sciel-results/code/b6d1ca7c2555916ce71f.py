"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Kissing number in 11D: known to be >= 504
    # Strategy: Use D11 lattice vectors with norm^2 = 4
    # D11: integer vectors with even coordinate sum
    # Norm^2 = 4 vectors:
    # 1. ±2e_i: 22 vectors
    # 2. ±e_i ± e_j ± e_k ± e_l with even sum: C(11,4)*8 = 2640 vectors
    # Total: 2662 candidates
    # Constraint: pairwise dot product <= 2 (since 2*4 - 2*a·b >= 4 => a·b <= 2)
    
    rng = np.random.RandomState(42)
    
    # Generate norm^2=4 candidates in D11
    cands = []
    
    # Type 1: ±2e_i
    for i in range(D):
        v1 = np.zeros(D, dtype=np.int64)
        v1[i] = 2
        cands.append(v1)
        v2 = np.zeros(D, dtype=np.int64)
        v2[i] = -2
        cands.append(v2)
    
    # Type 2: ±e_i ± e_j ± e_k ± e_l with even sum
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            s = sum(signs)
            if s % 2 == 0:  # even sum
                v = np.zeros(D, dtype=np.int64)
                for pos, sign in zip(support, signs):
                    v[pos] = sign
                cands.append(v)
    
    cands = np.array(cands, dtype=np.float64)
    n_cands = len(cands)
    
    # Base: 220 vectors with norm^2=2 (always valid)
    base_vecs = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base_vecs.append(v)
    base_vecs = np.array(base_vecs, dtype=np.int64)
    
    best_vecs = base_vecs
    best_count = len(base_vecs)
    
    # Efficient greedy: add candidate if dot with all existing <= max_dot
    def efficient_greedy(candidates, max_dot, rng, time_limit):
        n = len(candidates)
        order = rng.permutation(n)
        selected = []
        
        for idx in order:
            if time.time() > time_limit:
                break
            v = candidates[idx]
            if len(selected) == 0:
                selected.append(v)
                continue
            dots = np.dot(np.array(selected), v)
            if np.all(dots <= max_dot):
                selected.append(v)
        
        return np.array(selected, dtype=np.int64)
    
    # Try multiple random greedy passes on norm^2=4 candidates
    for trial in range(100):
        if time.time() > deadline - 5:
            break
        rng_t = np.random.RandomState(trial * 137 + 42)
        result = efficient_greedy(cands, 2, rng_t, deadline - 5)
        if len(result) > best_count:
            best_count = len(result)
            best_vecs = result
    
    return best_vecs
# EVOLVE-BLOCK-END
