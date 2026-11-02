"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
from collections import defaultdict


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # We work with vectors of squared norm 4 (norm = 2).
    # Constraint: min pairwise distance >= max norm = 2
    # ||a-b||^2 >= 4
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 8 - 2*a.b >= 4
    # => a.b <= 2
    # So we need all pairwise dot products <= 2.

    # Generate all integer vectors with squared norm 4 in R^11.
    # Patterns:
    # 1. Two entries of ±2: norm^2 = 4+4=8. No wait, 2^2+2^2=8≠4.
    #    Actually 2^2 = 4, so one entry of ±2 gives norm^2=4.
    # 2. Two entries of ±1 and one entry of ±√2? No, must be integer.
    #    1^2 + 1^2 + 1^2 = 3 ≠ 4
    #    1^2 + 1^2 = 2 ≠ 4
    #    2^2 = 4 ✓ (one entry is ±2)
    #    1^2 + 1^2 + ... no integer combo of squares sums to 4 except 4+0 or 1+1+1+1
    #    1+1+1+1 = 4 ✓ (four entries of ±1)
    
    # So vectors with squared norm 4 are:
    # Type A: one entry is ±2, rest 0. Count: 11 * 2 = 22
    # Type B: four entries are ±1, rest 0. Count: C(11,4) * 2^4 = 330 * 16 = 5280
    # Total: 22 + 5280 = 5302

    # Let's generate all candidates
    candidates = []
    
    # Type A: one entry ±2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
    
    # Type B: four entries ±1
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(support, signs):
                v[pos] = sign
            candidates.append(v)
    
    cands = np.array(candidates, dtype=np.float64)
    n_cands = len(cands)
    
    # Greedy algorithm: select subset with all pairwise dot products <= 2
    # Use a smarter greedy: for each candidate, track how many conflicts it has
    # with already-selected vectors, and pick the one with fewest conflicts first
    
    # For efficiency, we'll use a bitset-like approach or just iterate
    # Since n_cands ~ 5302, and we might select ~300-500, the greedy loop
    # will be O(n_cands * selected) which is ~5302 * 500 = 2.6M dot products.
    # Each dot product is 11-dim, so ~28M operations. Should be fast enough.
    
    # Better greedy: sort candidates by some heuristic, then greedily add
    # Or use a "minimum conflict" greedy
    
    # Let's use a multi-pass greedy with different orderings
    best_set = []
    best_size = 0
    
    # Pass 1: greedy by random order
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_cands)
    
    selected = []
    selected_dots = np.zeros((0, D), dtype=np.float64)
    
    for idx in indices:
        if time.time() > deadline - 5:
            break
        v = cands[idx]
        if len(selected) == 0:
            selected.append(idx)
            selected_dots = v.reshape(1, -1)
            continue
        # Check dot products with all selected
        dots = selected_dots @ v
        if np.all(dots <= 2):
            selected.append(idx)
            selected_dots = np.vstack([selected_dots, v])
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected[:]
    
    # Pass 2: greedy with a different ordering (e.g., by support pattern)
    # Try ordering by type: Type A first, then Type B
    type_a_indices = list(range(22))  # first 22 are Type A
    type_b_indices = list(range(22, n_cands))
    
    # Shuffle Type B
    rng2 = np.random.RandomState(123)
    rng2.shuffle(type_b_indices)
    
    order2 = type_a_indices + type_b_indices
    
    selected2 = []
    selected_dots2 = np.zeros((0, D), dtype=np.float64)
    
    for idx in order2:
        if time.time() > deadline - 5:
            break
        v = cands[idx]
        if len(selected2) == 0:
            selected2.append(idx)
            selected_dots2 = v.reshape(1, -1)
            continue
        dots = selected_dots2 @ v
        if np.all(dots <= 2):
            selected2.append(idx)
            selected_dots2 = np.vstack([selected_dots2, v])
    
    if len(selected2) > best_size:
        best_size = len(selected2)
        best_set = selected2[:]
    
    # Pass 3: try a "minimum conflict" greedy
    # For each step, among all remaining candidates, pick the one with minimum
    # number of conflicts (dot > 2) with already selected
    # This is O(n_cands^2 * D) which is too slow for 5302 candidates
    # Instead, do a few random restarts with different orderings
    
    for trial in range(5):
        if time.time() > deadline - 5:
            break
        rng_t = np.random.RandomState(1000 + trial)
        indices_t = rng_t.permutation(n_cands)
        
        selected_t = []
        selected_dots_t = np.zeros((0, D), dtype=np.float64)
        
        for idx in indices_t:
            if time.time() > deadline - 5:
                break
            v = cands[idx]
            if len(selected_t) == 0:
                selected_t.append(idx)
                selected_dots_t = v.reshape(1, -1)
                continue
            dots = selected_dots_t @ v
            if np.all(dots <= 2):
                selected_t.append(idx)
                selected_dots_t = np.vstack([selected_dots_t, v])
        
        if len(selected_t) > best_size:
            best_size = len(selected_t)
            best_set = selected_t[:]
    
    # Convert best set to array
    if best_size > 0:
        result = cands[best_set].astype(np.int64)
    else:
        # Fallback: just return the 22 Type A vectors
        result = cands[:22].astype(np.int64)
    
    # Verify the result
    if len(result) > 1:
        norms_sq = np.sum(result * result, axis=1)
        max_norm = np.max(norms_sq)
        # Check pairwise distances
        n = len(result)
        # Sample check for large n
        if n <= 1000:
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, n):
                    diff = result[i] - result[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < max_norm - 1e-9:
                        # Invalid! Fall back to safe set
                        return cands[:22].astype(np.int64)
    
    return result
# EVOLVE-BLOCK-END
