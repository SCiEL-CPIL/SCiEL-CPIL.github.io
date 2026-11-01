"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # Strategy:
    # Use vectors with norm squared = 4 (Norm = 2).
    # Constraint: min pairwise dist >= max norm = 2
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 8 - 2 u.v >= 4
    # => u.v <= 2
    
    # Type A: Weight 1 vectors in {-2, 0, 2}^11. Norm^2 = 4. Count: 22.
    # Type C: Weight 4 vectors in {-1, 0, 1}^11. Norm^2 = 4. Count: C(11,4)*16 = 5280.
    
    # W1 vs W4: max dot = 2 (compatible always)
    # W1 vs W1: all compatible (dot <= 0 or -4)
    # W4 vs W4: conflict if dot > 2, i.e., dot >= 3
    # For two W4 vectors with supports S, T:
    # dot = sum of products over intersection
    # If |S ∩ T| = k, max dot = k (all signs agree)
    # Conflict when k >= 3 with enough sign agreement
    # More precisely: dot = (#agreements) - (#disagreements) on S∩T
    # Let a = agreements, d = disagreements, a+d = |S∩T|
    # dot = a - d = 2a - |S∩T|
    # Conflict: 2a - |S∩T| > 2 => 2a > |S∩T| + 2
    
    # Map W4 vectors to binary: position in support (11 bits) + sign pattern (4 bits)
    # Two W4 vectors conflict based on their overlap and sign agreement.
    
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
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency lists for efficient neighbor lookup
    adj = [np.where(conflicts[i])[0] for i in range(N4)]
    
    # Multiple randomized restarts with greedy + local search
    rng = np.random.default_rng(12345)
    
    best_size = 0
    best_in_set = None
    
    num_restarts = 8
    time_per_restart = (deadline - start_time - 5) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Greedy Min-Degree for initial solution
        alive = np.ones(N4, dtype=bool)
        selected = []
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            selected.append(chosen)
            neighbors = adj[chosen]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        in_set = np.zeros(N4, dtype=bool)
        for idx in selected:
            in_set[idx] = True
        current_size = in_set.sum()
        
        # Local search: try to improve by swapping
        # For each vertex not in set, compute cost (blockers in set) and gain (1)
        # Try 1-for-k swaps where k < 1 (improvement) or k = 1 (neutral)
        
        T = 2.0
        cooling = 0.9995
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            # Try multiple candidates per iteration for efficiency
            batch_size = min(50, len(not_in_set))
            trial_indices = not_in_set[rng.choice(len(not_in_set), size=batch_size, replace=False)]
            
            improved_this_iter = False
            for v in trial_indices:
                neighbors_in_set = np.where(conflicts[v] & in_set)[0]
                k = len(neighbors_in_set)
                delta = 1 - k
                
                accept = False
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
                else:
                    if T > 0.01:
                        accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    in_set[v] = True
                    if k > 0:
                        in_set[neighbors_in_set] = False
                    current_size += delta
                    improved_this_iter = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
            
            if not improved_this_iter:
                no_improve_count += 1
            else:
                no_improve_count = 0
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        # Final greedy pass to add any remaining compatible vertices
        changed = True
        while changed:
            changed = False
            for v in range(N4):
                if not in_set[v]:
                    if not np.any(conflicts[v] & in_set):
                        in_set[v] = True
                        current_size += 1
                        changed = True
                        if current_size > best_size:
                            best_size = current_size
                            best_in_set = in_set.copy()
        
        if best_in_set is None:
            best_in_set = in_set
            best_size = current_size
    
    # Use the best found
    if best_in_set is None:
        best_in_set = np.zeros(N4, dtype=bool)
    
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Combine with W1
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
