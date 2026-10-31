import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Generate candidate vectors with different weights, all with Norm^2 = 4
    # Weight-4 vectors in {-1, 0, 1}^11
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Weight-2 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[positions[0]] = 2 * signs[0]
            v[positions[1]] = 2 * signs[1]
            v2_candidates.append(v)
            
    # Weight-1 vectors in {-4, 0, 4}^11 (Norm^2 = 16) - different norm, need separate handling
    # Actually, let's stick to Norm^2 = 4 for simplicity and maximum density
    # Weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    
    # Combine all candidates with Norm^2 = 4
    all_candidates = v4_candidates + v2_candidates + v1_candidates
    vectors = np.array(all_candidates, dtype=np.int64)
    
    N = len(vectors)
    
    if N == 0:
        return np.array([], dtype=np.int64)
    
    # Compute conflict graph: conflict if dot > 2
    # For norm^2 = 4, distance^2 = 8 - 2*dot, need distance >= 2 => dot <= 2
    dots = vectors @ vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Use a sophisticated greedy independent set algorithm with multiple strategies
    # Strategy 1: Minimum degree greedy with random tie-breaking
    # Strategy 2: Random order greedy
    # Strategy 3: Spectral-guided greedy
    
    best_selected = None
    best_count = 0
    
    # Precompute neighbor counts
    degrees = conflicts.sum(axis=1).astype(np.int64)
    
    # Strategy 1: Minimum degree greedy
    alive = np.ones(N, dtype=bool)
    selected = np.zeros(N, dtype=bool)
    deg_copy = degrees.copy()
    rng = np.random.default_rng(42)
    
    while np.any(alive) and time.time() < deadline - 20:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        alive_degrees = deg_copy[alive_indices]
        min_deg = np.min(alive_degrees)
        cands = alive_indices[alive_degrees == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected[chosen] = True
        to_remove = np.where(conflicts[chosen] | (np.arange(N) == chosen))[0]
        alive[to_remove] = False
        remaining = np.where(alive)[0]
        if len(remaining) > 0:
            deg_copy[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
    
    count1 = selected.sum()
    if count1 > best_count:
        best_count = count1
        best_selected = selected.copy()
    
    # Strategy 2: Multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        rng2 = np.random.default_rng(1000 + restart)
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        order = rng2.permutation(N)
        
        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N) == i))[0]
            alive[to_remove] = False
        
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Strategy 3: Local search improvement
    if best_selected is not None and time.time() < deadline - 5:
        current_selected = best_selected.copy()
        current_count = best_count
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add a non-selected vertex that doesn't conflict with any selected
            non_selected = np.where(~current_selected)[0]
            for v in non_selected:
                if time.time() > deadline - 5:
                    break
                # Check if v conflicts with any selected vertex
                selected_neighbors = np.where(current_selected & conflicts[v])[0]
                if len(selected_neighbors) == 0:
                    # Can add v freely
                    current_selected[v] = True
                    current_count += 1
                    improved = True
                    break
            
            # Try 2-swap: add two vertices, remove their selected neighbors
            if not improved:
                # This is more complex, skip for now due to time constraints
                pass
        
        if current_count > best_count:
            best_count = current_count
            best_selected = current_selected.copy()
    
    result = vectors[best_selected]
    
    return result
