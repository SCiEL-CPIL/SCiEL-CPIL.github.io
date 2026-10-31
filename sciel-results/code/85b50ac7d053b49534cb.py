"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Strategy:
    # Use vectors with norm squared = 4 (Norm = 2).
    # The constraint is min pairwise dist >= max norm.
    # If all vectors have norm 2, we need dist^2 >= 4.
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # 8 - 2 u.v >= 4 => u.v <= 2.
    
    # Type A: Weight 1 vectors in {-2, 0, 2}^11.
    # Norm^2 = 4.
    # Count: 11 positions * 2 signs = 22.
    # Mutual compatibility:
    # Same pos, diff sign: u.v = -4 <= 2.
    # Diff pos: u.v = 0 <= 2.
    # All 22 are compatible.
    
    # Type C: Weight 4 vectors in {-1, 0, 1}^11.
    # Norm^2 = 4.
    # Conflict if u.v > 2.
    # W4 vs W1 (Type A) requires u.v <= 2. (Always true for W4, max dot 2).
    # So we can take ALL 22 W1 vectors plus any independent set of W4 vectors.
    
    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix for W4
    # Conflict if u.v > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency list
    adj = [np.where(conflicts[i])[0] for i in range(N4)]
    
    # 1. Greedy Min-Degree MIS to get a good initial solution
    alive = np.ones(N4, dtype=bool)
    selected_indices = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(42)
    
    while np.any(alive) and time.time() < deadline - 60:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected_indices.append(chosen)
        
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        alive[to_remove] = False
        
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            
    # 2. Simulated Annealing with Kempe-Chain style swaps
    # State: boolean mask of size N4.
    # Move: Pick a vertex v not in set. 
    # If it conflicts with k vertices in the set, we must remove those k vertices.
    # Net change: 1 - k.
    # We also consider "Kempe" style moves where we might swap a set of blockers for a set of non-blockers?
    # For now, let's stick to the single-vertex insertion/removal but with better temperature schedule
    # and possibly multiple restarts if time permits.
    
    in_set = np.zeros(N4, dtype=bool)
    for idx in selected_indices:
        in_set[idx] = True
    
    current_size = in_set.sum()
    best_size = current_size
    best_in_set = in_set.copy()
    
    # Improved SA parameters
    T = 5.0  # Start higher to allow more exploration
    cooling = 0.9995  # Slower cooling
    min_T = 0.5
    
    # Precompute conflict counts for quick evaluation?
    # conflicts is a boolean matrix.
    
    while time.time() < deadline - 10:
        # Pick a random vertex not in the set
        not_in_set_indices = np.where(~in_set)[0]
        if len(not_in_set_indices) == 0:
            break
            
        v = not_in_set_indices[rng.integers(len(not_in_set_indices))]
        
        # Find neighbors of v that are in the set
        neighbors_in_set = np.where(conflicts[v] & in_set)[0]
        k = len(neighbors_in_set)
        
        delta = 1 - k
        
        if delta > 0:
            # Always accept
            in_set[v] = True
            in_set[neighbors_in_set] = False
            current_size += delta
        elif delta == 0:
            # Accept with high probability to move around the plateau
            if rng.random() < 0.8:
                in_set[v] = True
                in_set[neighbors_in_set] = False
        else:
            # Worsening. Accept with prob exp(delta/T)
            if rng.random() < np.exp(delta / T):
                in_set[v] = True
                in_set[neighbors_in_set] = False
                current_size += delta
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        T *= cooling
        if T < min_T:
            T = min_T
            
    # Use the best found
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Combine with W1
    # W1 vectors are compatible with all W4 vectors (max dot 2).
    # W1 vectors are compatible with each other (checked earlier).
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
