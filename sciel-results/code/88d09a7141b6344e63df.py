import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # 1. Generate Candidates
    candidates = []
    norms_sq = []
    
    # Type 1: Norm sq 4 (One 2)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            norms_sq.append(4)
            
    # Type 2: Norm sq 4 (Four 1s)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(4)
            
    # Type 3: Norm sq 3 (Three 1s)
    for idxs in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(3)
            
    # Type 4: Norm sq 2 (Two 1s)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(2)
            
    # Type 5: Norm sq 1 (One 1)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
            norms_sq.append(1)

    candidates = np.array(candidates, dtype=np.int32)
    norms_sq = np.array(norms_sq, dtype=np.int32)
    N = len(candidates)
    
    # 2. Compute conflict matrix
    M = candidates @ candidates.T
    norms_sq_col = norms_sq[:, None]
    norms_sq_row = norms_sq[None, :]
    
    R_sq = 4
    allowed_dot = (norms_sq_col + norms_sq_row - R_sq) // 2
    conflict = (M > allowed_dot).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # 3. Simulated Annealing on continuous relaxation
    # x[i] in [0, 1] represents selection probability of node i
    # Objective: maximize sum(x) - lambda * sum_{i,j: conflict[i,j]} x[i]*x[j]
    # We use a smooth penalty for pairwise conflicts
    
    rng = np.random.default_rng(42)
    
    best_count = 0
    best_solution = np.zeros(N, dtype=bool)
    
    # Precompute conflict degrees for efficient evaluation
    conflict_degrees = np.sum(conflict, axis=1).astype(np.float64)
    
    # Multiple restarts with different initializations
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize x
        if restart == 0:
            # Start with high values
            x = np.full(N, 0.8, dtype=np.float64)
        elif restart % 3 == 1:
            # Random initialization
            x = rng.uniform(0.3, 0.9, size=N)
        else:
            # Start from best found so far, add noise
            x = best_solution.astype(np.float64) + rng.uniform(-0.1, 0.1, size=N)
            x = np.clip(x, 0.0, 1.0)
        
        # Temperature schedule
        T_initial = 1.0
        T_final = 0.01
        n_steps = 2000
        alpha = (T_final / T_initial) ** (1.0 / n_steps)
        T = T_initial
        
        # Penalty weight - needs to be large enough to avoid conflicts
        # But not too large to prevent exploration
        lambda_penalty = 2.0
        
        # Evaluate objective function
        def evaluate(x_vec):
            # sum(x) - lambda * sum_{conflicts} x[i]*x[j]
            # Efficient computation using matrix
            penalty = 0.0
            # Only compute for conflict pairs
            # Use the fact that conflict is symmetric
            conflict_pairs_i, conflict_pairs_j = np.where(conflict > 0)
            if len(conflict_pairs_i) > 0:
                # Only take upper triangle to avoid double counting
                mask = conflict_pairs_i < conflict_pairs_j
                ci = conflict_pairs_i[mask]
                cj = conflict_pairs_j[mask]
                penalty = np.sum(x_vec[ci] * x_vec[cj])
            return np.sum(x_vec) - lambda_penalty * penalty
        
        current_obj = evaluate(x)
        best_obj = current_obj
        best_x = x.copy()
        
        # Annealing loop
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            T *= alpha
            
            # Pick a random node to perturb
            i = rng.integers(N)
            old_val = x[i]
            
            # Perturb: move toward 0 or 1, or random step
            move = rng.choice([-1, 1]) * rng.uniform(0.05, 0.2)
            new_val = np.clip(old_val + move, 0.0, 1.0)
            
            if new_val == old_val:
                continue
            
            x[i] = new_val
            
            # Compute delta objective efficiently
            # delta = (new_val - old_val) - lambda * (new_val - old_val) * sum_{j: conflict[i,j]} x[j]
            # But we need to be careful: the penalty term for pair (i,j) is x[i]*x[j]
            # When we change x[i] from old to new, the change in penalty is:
            # (new - old) * x[j] for each j that conflicts with i
            # So delta_penalty = (new_val - old_val) * sum_{j: conflict[i,j]} x[j]
            # delta_obj = (new_val - old_val) - lambda * delta_penalty
            # Note: we should not include i in the sum (no self-conflict)
            
            conflict_neighbors = np.where(conflict[i] > 0)[0]
            if len(conflict_neighbors) > 0:
                neighbor_sum = np.sum(x[conflict_neighbors])
            else:
                neighbor_sum = 0.0
            
            delta = (new_val - old_val) - lambda_penalty * (new_val - old_val) * neighbor_sum
            
            # Accept with Metropolis criterion
            if delta > 0:
                pass  # Always accept improving moves
            else:
                # Accept with probability exp(delta / T)
                if T > 0:
                    prob = np.exp(delta / T)
                    if rng.random() > prob:
                        x[i] = old_val  # Reject
                        continue
            
            current_obj += delta
            
            if current_obj > best_obj:
                best_obj = current_obj
                best_x = x.copy()
        
        # Extract binary solution from best_x
        # Threshold at 0.5
        binary_sol = best_x > 0.5
        
        # Post-process: greedily remove conflicts from the binary solution
        # Sort by x value descending, keep if compatible
        order = np.argsort(-best_x)
        cleaned = np.zeros(N, dtype=bool)
        count = 0
        for idx in order:
            if best_x[idx] > 0.5:
                if not np.any(conflict[idx] & cleaned):
                    cleaned[idx] = True
                    count += 1
        
        if count > best_count:
            best_count = count
            best_solution = cleaned.copy()
    
    # 4. Local search on the best solution to improve further
    if time.time() < deadline - 10:
        # Try to add more vectors
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            unselected = np.where(~best_solution)[0]
            if len(unselected) > 0:
                # Shuffle to break symmetry
                rng.shuffle(unselected)
                for idx in unselected:
                    if not np.any(conflict[idx] & best_solution):
                        best_solution[idx] = True
                        best_count += 1
                        improved = True
                        break
        
        # Try 1-2 swaps
        if time.time() < deadline - 5:
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                selected_indices = np.where(best_solution)[0]
                if len(selected_indices) < 2:
                    break
                
                # Pick a random selected node to try removing
                u = selected_indices[rng.integers(len(selected_indices))]
                
                # Tentatively remove u
                best_solution[u] = False
                best_count -= 1
                
                # Try to add two compatible nodes
                unselected = np.where(~best_solution)[0]
                if len(unselected) >= 2:
                    # Try random pairs
                    found_pair = False
                    for _ in range(20):
                        v, w = rng.choice(unselected, size=2, replace=False)
                        if (not np.any(conflict[v] & best_solution) and 
                            not np.any(conflict[w] & best_solution) and
                            not conflict[v, w]):
                            best_solution[v] = True
                            best_solution[w] = True
                            best_count += 2
                            found_pair = True
                            break
                    
                    if not found_pair:
                        # Restore u
                        best_solution[u] = True
                        best_count += 1
    
    # 5. Return the result
    if best_count == 0:
        # Fallback: return Type 1 vectors
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_solution].astype(np.int64)
