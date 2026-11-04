import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
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
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build conflict graph: conflict if intersection size >= 3
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount for intersection sizes
    # Max bits is 11, so 2^11 = 2048
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3
    
    # Precompute neighbors list for faster access if needed, 
    # but for N=330, matrix ops are very fast.
    
    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_min_degree_init(randomize_ties=True):
        """
        Greedy algorithm that picks the node with the minimum degree 
        in the remaining graph.
        """
        available = np.ones(N_S, dtype=bool)
        selected = np.zeros(N_S, dtype=bool)
        
        while available.any():
            # Calculate degrees within the available subgraph
            # conflict[available] is a submatrix
            # We want the number of True entries in each row for available columns
            # Sum along axis 1 gives the degree of each available node
            sub_conflict = conflict[available][:, available]
            degrees = np.sum(sub_conflict, axis=1)
            
            # Find the minimum degree
            min_deg = np.min(degrees)
            
            # Get indices in the 'available' array that have this min degree
            # These are local indices relative to the 'available' mask
            local_indices = np.where(degrees == min_deg)[0]
            
            if randomize_ties and len(local_indices) > 1:
                # Pick a random one among the minimum degree nodes
                local_idx = py_rng.choice(local_indices)
            else:
                local_idx = local_indices[0]
                
            # Map back to global index
            # available_indices are the global indices where available is True
            global_indices = np.where(available)[0]
            global_idx = global_indices[local_idx]
            
            selected[global_idx] = True
            available[global_idx] = False
            
            # Remove neighbors from available
            neighbors = np.where(conflict[global_idx])[0]
            available[neighbors] = False
            
        return selected, np.sum(selected)

    def local_search(x, current_size, num_iterations):
        """
        Local search to improve the independent set.
        Moves: Add, Remove, Swap.
        """
        best_x = x.copy()
        best_size_local = current_size
        x = x.copy()
        
        for it in range(num_iterations):
            if time.time() > deadline - 15:
                break
            
            # Compute blocked nodes: nodes that conflict with any node in x
            if current_size > 0:
                blocked = conflict[x].any(axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            add_candidates = np.where(~x & ~blocked)[0]
            
            if len(add_candidates) > 0:
                # Try to add a node
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            else:
                # Try to swap
                selected = np.where(x)[0]
                if len(selected) == 0:
                    break
                
                v_rem = py_rng.choice(selected)
                x[v_rem] = False
                current_size -= 1
                
                # Recompute blocked after removal
                if current_size > 0:
                    blocked = conflict[x].any(axis=0)
                else:
                    blocked = np.zeros(N_S, dtype=bool)
                
                # Candidates that can be added now
                # Must not be in x (v_rem is not in x) and not blocked
                add_candidates = np.where(~x & ~blocked)[0]
                
                if len(add_candidates) > 0:
                    v_add = py_rng.choice(add_candidates)
                    x[v_add] = True
                    current_size += 1
                    if current_size > best_size_local:
                        best_size_local = current_size
                        best_x = x.copy()
                else:
                    # Revert
                    x[v_rem] = True
                    current_size += 1

        return best_x, best_size_local

    num_restarts = 100
    iterations_per_restart = 5000

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        # Use the new greedy min-degree initialization
        x, size = greedy_min_degree_init(randomize_ties=True)
        
        # Refine with local search
        best_x, best_size_local = local_search(x, size, iterations_per_restart)
        
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Generate Type D vectors
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    final_vectors = np.vstack([vectors_A, selected_vectors_D])
    return final_vectors
