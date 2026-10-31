"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11
DEADLINE = 120.0  # seconds
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _check_deadline():
    return _elapsed() > DEADLINE - 5.0


# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Score is k (number of vectors).
    
    Strategy: Use lattice-theoretic construction based on the known kissing
    number in dimension 11. We'll use a structured approach:
    
    1. Start with weight-2 vectors (like the seed) as a base.
    2. Add weight-4 and other structured vectors that maintain the distance constraint.
    3. Use greedy selection to maximize the count while maintaining validity.
    
    The key insight: if we normalize to norm 2, we need min pairwise distance >= 2.
    For integer vectors, this means for any two distinct vectors u, v:
        ||u - v|| >= max(||u||, ||v||)
    
    Let's work with vectors of various norms and check the constraint.
    """
    deadline = DEADLINE - 5.0
    
    # Generate candidate vectors of various weights
    # Weight 1: norm 1 (but these are problematic for distance constraints)
    # Weight 2: norm 2 (the seed approach)
    # Weight 4: norm 4 (higher density regions)
    
    candidates = []
    
    # Weight 2 vectors: e_i + e_j, e_i - e_j, -e_i + e_j, -e_i - e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Weight 4 vectors: pick 4 coordinates, assign +/-1
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            candidates.append(v)
    
    # Weight 3 vectors: pick 3 coordinates, assign +/-1
    for coords in itertools.combinations(range(D), 3):
        for signs in itertools.product((1, -1), repeat=3):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            candidates.append(v)
    
    # Convert to numpy array for efficient computation
    candidates_arr = np.array(candidates, dtype=np.int64)
    
    # Compute norms
    norms = np.linalg.norm(candidates_arr, axis=1)
    
    # Sort by norm descending (prefer higher norm vectors for better packing)
    order = np.argsort(-norms)
    candidates_arr = candidates_arr[order]
    norms = norms[order]
    
    # Greedy selection: add vectors if they satisfy the distance constraint
    selected = []
    selected_norms = []
    
    max_norm = 0.0
    
    for idx in range(len(candidates_arr)):
        if _check_deadline():
            break
        
        v = candidates_arr[idx]
        v_norm = norms[idx]
        
        # Check against all selected vectors
        valid = True
        for sel_idx, (sel_v, sel_norm) in enumerate(zip(selected, selected_norms)):
            diff = v - sel_v
            dist = np.linalg.norm(diff)
            required = max(v_norm, sel_norm)
            if dist < required - 1e-9:
                valid = False
                break
        
        if valid:
            selected.append(v)
            selected_norms.append(v_norm)
            max_norm = max(max_norm, v_norm)
    
    if not selected:
        # Fallback to seed approach
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
