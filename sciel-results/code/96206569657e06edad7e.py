"""New program for kissing number in dimension 11 using Weight-3 vectors and Simulated Annealing."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate all weight-3 vectors with entries in {-1, 0, 1}
    v3_list = []
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v3_list.append(v)
    
    vectors_v3 = np.array(v3_list, dtype=np.int64)
    N = len(vectors_v3)

    if N == 0:
        return np.zeros((0, D), dtype=np.int64)

    # Compute pairwise dot products
    dots = vectors_v3 @ vectors_v3.T
    
    # Conflict if dot > 1 (since norm^2=3, condition is dot <= 1.5 -> dot <= 1 for integers)
    conflicts = dots > 1
    np.fill_diagonal(conflicts, False)
    
    # Precompute adjacency lists for faster conflict checks
    adj = [np.where(conflicts[i])[0] for i in range(N)]
    
    rng = np.random.default_rng(42)
    
    def is_compatible(v_idx, current_set):
        """Check if v_idx is compatible with all vectors in current_set."""
        if len(current_set) == 0:
            return True
        # Check if any neighbor of v_idx is in current_set
        # current_set is a boolean mask or list of indices
        if isinstance(current_set, np.ndarray) and current_set.dtype == bool:
            return not np.any(conflicts[v_idx] & current_set)
        else:
            # current_set is list of indices
            return not any(u in current_set for u in adj[v_idx])

    def sa_solve(max_time):
        """Simulated Annealing to find large independent set."""
        # Initialize with a greedy independent set
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        
        # Greedy init: pick random node, remove neighbors
        order = rng.permutation(N)
        for i in order:
            if alive[i]:
                in_set[i] = True
                alive[adj[i]] = False
                alive[i] = False
        
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # SA Parameters
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        ls_deadline = start_time + max_time
        
        while time.time() < ls_deadline:
            # Choose a move
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try Add
                not_in = np.where(~in_set)[0]
                if len(not_in) == 0: continue
                v = not_in[rng.integers(len(not_in))]
                if not np.any(conflicts[v] & in_set):
                    # Accept add
                    in_set[v] = True
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                # If conflict, reject (or handle via swap)
            
            elif move_type < 0.7:
                # Try Remove
                in_indices = np.where(in_set)[0]
                if len(in_indices) == 0: continue
                v = in_indices[rng.integers(len(in_indices))]
                in_set[v] = False
                current_size -= 1
            
            else:
                # Try Swap
                in_indices = np.where(in_set)[0]
                not_in = np.where(~in_set)[0]
                if len(in_indices) == 0 or len(not_in) == 0: continue
                
                v_in = in_indices[rng.integers(len(in_indices))]
                v_out = not_in[rng.integers(len(not_in))]
                
                # Check if v_out is compatible with in_set \ {v_in}
                # Create a mask of in_set without v_in
                temp_set = in_set.copy()
                temp_set[v_in] = False
                
                if not np.any(conflicts[v_out] & temp_set):
                    # Accept swap
                    in_set[v_in] = False
                    in_set[v_out] = True
                    # Size unchanged, but might enable future additions
                    # We don't update best_size here unless size increased, which it didn't
                    # But we accept the move to explore
            
            # Cooling
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_in_set, best_size

    # Run SA for a large chunk of time
    # We can run multiple restarts
    best_final_set = np.zeros(N, dtype=bool)
    best_final_size = 0
    
    # First run: long SA
    t1 = min(60, deadline - time.time() - 5)
    if t1 > 5:
        sol, size = sa_solve(t1)
        best_final_set = sol
        best_final_size = size
        
    # Second run: restart with different seed, shorter time
    rng = np.random.default_rng(999)
    t2 = min(40, deadline - time.time() - 5)
    if t2 > 5:
        # Re-run SA logic with new rng? 
        # The sa_solve function uses the global rng? No, it uses the one passed or defined inside.
        # Let's refactor sa_solve to take rng.
        
        def sa_solve_rng(rng, max_time):
            in_set = np.zeros(N, dtype=bool)
            alive = np.ones(N, dtype=bool)
            
            order = rng.permutation(N)
            for i in order:
                if alive[i]:
                    in_set[i] = True
                    alive[adj[i]] = False
                    alive[i] = False
            
            current_size = in_set.sum()
            best_size = current_size
            best_in_set = in_set.copy()
            
            T = 1.0
            cooling = 0.9995
            min_T = 0.01
            ls_deadline = time.time() + max_time
            
            while time.time() < ls_deadline:
                move_type = rng.random()
                
                if move_type < 0.4:
                    not_in = np.where(~in_set)[0]
                    if len(not_in) == 0: continue
                    v = not_in[rng.integers(len(not_in))]
                    if not np.any(conflicts[v] & in_set):
                        in_set[v] = True
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_in_set = in_set.copy()
                
                elif move_type < 0.7:
                    in_indices = np.where(in_set)[0]
                    if len(in_indices) == 0: continue
                    v = in_indices[rng.integers(len(in_indices))]
                    in_set[v] = False
                    current_size -= 1
                
                else:
                    in_indices = np.where(in_set)[0]
                    not_in = np.where(~in_set)[0]
                    if len(in_indices) == 0 or len(not_in) == 0: continue
                    
                    v_in = in_indices[rng.integers(len(in_indices))]
                    v_out = not_in[rng.integers(len(not_in))]
                    
                    temp_set = in_set.copy()
                    temp_set[v_in] = False
                    
                    if not np.any(conflicts[v_out] & temp_set):
                        in_set[v_in] = False
                        in_set[v_out] = True
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            return best_in_set, best_size

        sol2, size2 = sa_solve_rng(rng, t2)
        if size2 > best_final_size:
            best_final_set = sol2
            best_final_size = size2

    final_indices = np.where(best_final_set)[0]
    return vectors_v3[final_indices]

# EVOLVE-BLOCK-END
