import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4.
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Type D: Vectors with four non-zero entries +/- 1
    # Norm squared = 4.
    # Compatibility condition: dot(u, v) <= 2.
    # Supports are 4-subsets of {0, ..., 10}.
    # If intersection size >= 3, we must check signs.
    # If intersection size <= 2, max dot is 2 (always compatible).
    
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Precompute all 16 sign patterns for 4 entries
    sign_patterns = list(itertools.product([1, -1], repeat=4))
    
    # Generate all possible Type D vectors
    # Shape: (N_S * 16, D)
    all_vectors_D = []
    for s in supports:
        for signs in sign_patterns:
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)
    
    # Precompute dot products between all Type D vectors?
    # N_all_D is 330 * 16 = 5280.
    # 5280 x 5280 matrix of ints is ~25M entries. 100MB.
    # This is feasible and speeds up checking significantly.
    # dot[u, v] = sum(all_vectors_D[u] * all_vectors_D[v])
    # We need dot[u, v] <= 2 for all selected u, v.
    
    # Let's compute the full Gram matrix of all_vectors_D.
    # G = all_vectors_D @ all_vectors_D.T
    # G[i, j] is dot product.
    # We want to select a subset S such that for all i, j in S, G[i, j] <= 2.
    # Note: G[i, i] = 4.
    
    # This is a maximum clique problem in the graph where edge (i, j) exists if G[i, j] <= 2.
    # Or equivalently, maximum independent set in the conflict graph where edge (i, j) exists if G[i, j] > 2.
    
    # Since N_all_D = 5280, building the conflict graph explicitly is 5280^2 ~ 27M booleans.
    # This is about 27MB. Very feasible.
    
    # However, checking pairwise distances in a greedy loop is O(N^2).
    # With vectorized numpy, we can check a new vector against the current set quickly.
    
    # Let's try a greedy approach with random restarts and local search on the vector level.
    # Or, better, use the support structure to prune.
    
    # Strategy:
    # 1. Greedily select vectors.
    # 2. Use a "conflict mask" for the current set.
    # 3. Try to add vectors that don't conflict.
    # 4. Local search: swap vectors to allow more additions.
    
    # Given the deadline, a robust greedy with randomization and a final "fill" pass is likely best.
    
    # Let's use the precomputed Gram matrix for fast compatibility checks.
    # G = all_vectors_D @ all_vectors_D.T
    # This takes some time but is worth it for 5280 vectors.
    
    # Time check: 5280x11 @ 11x5280. ~300M FLOPs. Fast on modern CPUs.
    
    G = all_vectors_D @ all_vectors_D.T
    
    # Conflict matrix: C[i, j] = True if G[i, j] > 2
    # We want a set of indices such that no pair has C[i, j] = True.
    
    # Let's implement a greedy set packing algorithm with random restarts.
    
    best_count = 0
    best_indices = []
    
    rng = np.random.RandomState(42)
    
    num_restarts = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Random order
        order = np.arange(N_all_D)
        rng.shuffle(order)
        
        in_set = np.zeros(N_all_D, dtype=bool)
        # blocked[i] is True if i conflicts with any vector in the set
        blocked = np.zeros(N_all_D, dtype=bool)
        
        current_size = 0
        
        for v in order:
            if not blocked[v]:
                in_set[v] = True
                current_size += 1
                # Update blocked: v conflicts with anyone where G[v, :] > 2
                # But we only care about those not yet in set?
                # Actually, if we add v, it blocks all u such that G[v, u] > 2.
                # We can vectorize this.
                blocked |= (G[v] > 2)
        
        if current_size > best_count:
            best_count = current_size
            best_indices = np.where(in_set)[0].copy()
            
            # Optional: Local search to improve
            # Try to swap one vector in the set with two out?
            # Or just try to add more? The greedy already did that.
            # The bottleneck is the initial random order.
            
            # Let's try a simple local search:
            # While possible, try to remove a vector and add more.
            # This is expensive. Maybe skip for now and rely on restarts.
            
    # Construct the final vector list from best_indices
    if best_count > 0:
        selected_vectors_D = all_vectors_D[best_indices]
    else:
        selected_vectors_D = np.empty((0, D), dtype=np.int64)
        
    # Combine Type A and Type D
    # Check compatibility between A and D again?
    # Type A: norm^2=4. Type D: norm^2=4.
    # dist^2 = 8 - 2 dot.
    # We need dist^2 >= 4 => dot <= 2.
    # Type A has one entry +/- 2. Type D has four entries +/- 1.
    # Max dot is 2 (if the 2 aligns with a 1 in D).
    # So all Type A are compatible with all Type D.
    
    final_vectors = np.vstack([vectors_A, selected_vectors_D])
    
    return final_vectors
