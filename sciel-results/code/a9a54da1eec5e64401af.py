"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
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

    # Compute conflict matrix for W4 vectors
    # Conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors for fast updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def soft_sa(init_x, max_time, lambda_pen=1.0):
        """
        Optimized Soft-constraint Simulated Annealing.
        Runs in batches to reduce overhead.
        """
        x = init_x.copy()
        # conflict_counts[i] = number of neighbors of i that are in x
        conflict_counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(x)[0]:
            conflict_counts[neighbors[i]] += 1
        
        # Calculate initial energy
        size = x.sum()
        num_conflicts = np.sum(conflict_counts) // 2
        current_E = -size + lambda_pen * num_conflicts
        
        best_E = current_E
        best_x = x.copy()
        
        T = 1.0
        T_min = 0.001
        cooling = 0.999
        batch_size = 5000
        
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            # Run a batch of steps
            for _ in range(batch_size):
                i = rng.integers(N4)
                
                if x[i]:
                    delta_E = 1 - lambda_pen * conflict_counts[i]
                else:
                    delta_E = -1 + lambda_pen * conflict_counts[i]
                
                if delta_E < 0 or (T > T_min and rng.random() < np.exp(-delta_E / T)):
                    x[i] = not x[i]
                    
                    nbrs = neighbors[i]
                    if x[i]:
                        conflict_counts[nbrs] += 1
                    else:
                        conflict_counts[nbrs] -= 1
                    
                    current_E += delta_E
                    
                    if current_E < best_E:
                        best_E = current_E
                        best_x = x.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min

        return best_x

    def greedy_init():
        """Greedy initialization for SA."""
        x = np.zeros(N4, dtype=bool)
        # Use min-degree heuristic
        degrees = neighbor_counts.copy()
        alive = np.ones(N4, dtype=bool)
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            i = cands[rng.integers(len(cands))]
            
            x[i] = True
            # Remove i and its neighbors from alive
            to_remove = np.unique(np.concatenate(([i], neighbors[i])))
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return x

    def repair(x_res):
        """Greedy repair to ensure independence."""
        x_final = x_res.copy()
        active_idx = np.where(x_final)[0]
        for i in active_idx:
            if x_final[i]:
                nbrs = neighbors[i]
                if np.any(x_final[nbrs]):
                    x_final[i] = False
        return x_final

    best_size = 0
    best_x = np.zeros(N4, dtype=bool)

    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 2 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Initialize
        x_init = greedy_init()
        
        # Run SA
        # Dynamically adjust SA time based on remaining time
        remaining = deadline - time.time()
        sa_time = min(1.0, remaining * 0.1) # Use small fraction, allow many restarts
        if sa_time < 0.1:
             sa_time = 0.1
             
        x_res = soft_sa(x_init, sa_time)
        
        # Repair to ensure independence
        x_final = repair(x_res)
        
        size_final = x_final.sum()
        if size_final > best_size:
            best_size = size_final
            best_x = x_final.copy()
        
        # Occasionally try a different initialization strategy if time permits
        if n_restarts % 10 == 0:
             # Try random subset initialization
             x_rand = np.zeros(N4, dtype=bool)
             x_rand[rng.choice(N4, size=50, replace=False)] = True
             x_res2 = soft_sa(x_rand, sa_time)
             x_final2 = repair(x_res2)
             size_final2 = x_final2.sum()
             if size_final2 > best_size:
                 best_size = size_final2
                 best_x = x_final2.copy()

    final_v4_indices = np.where(best_x)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
