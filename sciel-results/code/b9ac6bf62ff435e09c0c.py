"""
Strategy: Discrete Simulated Annealing on Half-Integer Weights
The current program uses continuous optimization which often fails to land exactly on the half-integer lattice required for exact verification, or gets stuck in local minima of the continuous loss.
This new program fixes a sparse support structure (randomly selected non-zero positions) and performs Simulated Annealing directly on the discrete set of half-integer values {-2, -1.5, ..., 2} (or similar bounded range) for the weights.
This respects the "integer/half-integer" constraint intrinsically and uses the sparsity of the tensor to keep the search space manageable.
"""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 84
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    # Non-zero indices of the multiplication tensor
    idx_i = []
    idx_j = []
    idx_k = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_i.append(i * m + j)
                idx_j.append(j * p + k)
                idx_k.append(k * n + i)
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)
    num_nz = len(idx_i)  # 84

    start_time = time.time()
    deadline = 145.0  # Safety margin

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        U_sel = U[idx_i, :]
        V_sel = V[idx_j, :]
        W_sel = W[idx_k, :]
        T = np.sum(U_sel * V_sel * W_sel, axis=1)
        # Check exactness with tight tolerance
        return np.max(np.abs(T - 1.0)) < 1e-9

    def snap_half_int(U, V, W):
        """Snap to nearest half-integer and verify."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if verify(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    def simulate_annealing(R, max_iters=20000, T_start=1.0, T_end=0.01):
        """
        Discrete Simulated Annealing on half-integer weights.
        1. Initialize with a random sparse support of size R.
        2. Assign random half-integer weights.
        3. Iterate: pick a random non-zero entry, change its value to a neighbor half-integer.
        4. Accept if energy (sum of squared residuals) decreases, or with probability exp(-dE/T).
        """
        if time.time() > deadline:
            return None
        
        # Define the discrete domain for weights
        # Matrix multiplication tensors usually have small integer coefficients.
        # Let's restrict to [-3, 3] in half-steps: -3, -2.5, ..., 3
        domain = np.arange(-3, 3.5, 0.5)
        
        best_energy = float('inf')
        best_state = None
        
        # Try multiple restarts
        for restart in range(5):
            if time.time() > deadline:
                return None
            
            # 1. Initialize Support
            # We need to choose R columns. For each column, we need to pick one index for U, V, W.
            # To ensure the tensor can potentially be 1.0, we should bias initialization towards the "diagonal" 
            # where U[i*m+j], V[j*p+k], W[k*n+i] are non-zero for some (i,j,k).
            
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            
            # Randomly select R distinct "slots" from the 84 possible (i,j,k) tuples?
            # No, the rank R can be larger than 84, or the support can overlap.
            # However, for low rank, it's likely that each term corresponds to a specific (i,j,k) contribution.
            # Let's initialize by picking R random indices from the 84, with replacement if R > 84.
            
            if R <= 84:
                # Pick R unique indices
                chosen_indices = np.random.choice(84, R, replace=False)
            else:
                chosen_indices = np.random.choice(84, R, replace=True)
                
            # Map global index 0..83 to (i, j, k)
            # Index = i * m * p + j * p + k
            # U index = i * m + j
            # V index = j * p + k
            # W index = k * n + i
            
            for r, idx in enumerate(chosen_indices):
                i = idx // (m * p)
                j = (idx // p) % m
                k = idx % p
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Assign random half-integer weights
                # Bias towards 1.0 to start closer to schoolbook
                # P(1.0) = 0.5, others distributed
                weights = np.random.choice(domain, 3, p=[0.5] + [0.5/11]*11)
                U[u_idx, r] = weights[0]
                V[v_idx, r] = weights[1]
                W[w_idx, r] = weights[2]
            
            # Initial Energy
            def calc_energy():
                U_sel = U[idx_i, :]
                V_sel = V[idx_j, :]
                W_sel = W[idx_k, :]
                T = np.sum(U_sel * V_sel * W_sel, axis=1)
                return np.sum((T - 1.0)**2)
            
            current_energy = calc_energy()
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = (U.copy(), V.copy(), W.copy())
            
            T = T_start
            cooling = (T_end / T_start) ** (1.0 / max_iters)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return best_state if best_state else None
                
                # Pick a random non-zero entry to modify
                # We maintain a list of active (matrix, row, col) tuples
                # For simplicity, pick a random column r and a random matrix (0,1,2)
                r = np.random.randint(R)
                mat = np.random.randint(3)
                
                # Find the row index for this column
                # We stored the structure in chosen_indices initially, but weights can move?
                # No, in this simple version, the SUPPORT (non-zero positions) is fixed.
                # Only the WEIGHTS change.
                # This is a strong constraint. If the optimal solution requires moving a non-zero to a different row, this method fails.
                # To allow support changes, we'd need a more complex move.
                # Let's stick to weight changes first. If it fails, we might need to allow row swaps.
                
                # Get current value
                if mat == 0:
                    # Find the row for U[:, r]
                    # We need to know which row is non-zero.
                    # Let's store the structure explicitly.
                    pass
                
                # Actually, let's re-initialize the structure storage to be explicit
                # struct[r] = (u_row, v_row, w_row)
                
                # Let's restart the loop logic with explicit structure
                break # Break out to re-implement properly below

        return None

    def simulate_annealing_v2(R, max_iters=5000, T_start=1.0, T_end=0.001):
        """
        Improved SA: Fixed Support Structure, Discrete Weights.
        """
        if time.time() > deadline:
            return None
        
        domain = np.arange(-3, 3.5, 0.5)
        best_energy = float('inf')
        best_state = None
        
        for restart in range(10):
            if time.time() > deadline:
                return best_state if best_state else None
            
            # Initialize Structure: R terms, each with a fixed (u_row, v_row, w_row)
            # We pick R random (i,j,k) tuples.
            # If R > 84, we allow duplicates.
            if R <= 84:
                chosen_indices = np.random.choice(84, R, replace=False)
            else:
                chosen_indices = np.random.choice(84, R, replace=True)
                
            struct = []
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            
            for r, idx in enumerate(chosen_indices):
                i = idx // (m * p)
                j = (idx // p) % m
                k = idx % p
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                struct.append((u_idx, v_idx, w_idx))
                
                # Initialize weights with 1.0
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
            
            def calc_energy():
                U_sel = U[idx_i, :]
                V_sel = V[idx_j, :]
                W_sel = W[idx_k, :]
                T = np.sum(U_sel * V_sel * W_sel, axis=1)
                return np.sum((T - 1.0)**2)
            
            current_energy = calc_energy()
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = (U.copy(), V.copy(), W.copy())
                
            T = T_start
            cooling = (T_end / T_start) ** (1.0 / max_iters)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return best_state if best_state else None
                
                # Pick a random term r
                r = np.random.randint(R)
                u_idx, v_idx, w_idx = struct[r]
                
                # Pick a random matrix to update (0=U, 1=V, 2=W)
                mat = np.random.randint(3)
                
                old_val = 0.0
                if mat == 0:
                    old_val = U[u_idx, r]
                    new_val = np.random.choice(domain)
                    if new_val == old_val:
                        continue
                    U[u_idx, r] = new_val
                    delta_energy = calc_energy() - current_energy
                    if delta_energy > 0 and random.random() > np.exp(-delta_energy / T):
                        U[u_idx, r] = old_val
                    else:
                        current_energy += delta_energy
                elif mat == 1:
                    old_val = V[v_idx, r]
                    new_val = np.random.choice(domain)
                    if new_val == old_val:
                        continue
                    V[v_idx, r] = new_val
                    delta_energy = calc_energy() - current_energy
                    if delta_energy > 0 and random.random() > np.exp(-delta_energy / T):
                        V[v_idx, r] = old_val
                    else:
                        current_energy += delta_energy
                else:
                    old_val = W[w_idx, r]
                    new_val = np.random.choice(domain)
                    if new_val == old_val:
                        continue
                    W[w_idx, r] = new_val
                    delta_energy = calc_energy() - current_energy
                    if delta_energy > 0 and random.random() > np.exp(-delta_energy / T):
                        W[w_idx, r] = old_val
                    else:
                        current_energy += delta_energy
                
                # Cool down
                T *= cooling
                
                # Check for exact solution
                if current_energy < 1e-9:
                    snapped = snap_half_int(U, V, W)
                    if snapped:
                        return snapped

        return best_state if best_state else None

    def try_structural_sa(R):
        """
        Allow the support structure to change as well.
        Moves:
        1. Change weight of a term.
        2. Move a term's U-index to a different row (keeping V, W fixed).
        3. Move a term's V-index to a different row.
        4. Move a term's W-index to a different row.
        """
        if time.time() > deadline:
            return None
        
        domain = np.arange(-3, 3.5, 0.5)
        best_energy = float('inf')
        best_state = None
        
        for restart in range(3):
            if time.time() > deadline:
                return best_state if best_state else None
            
            # Init: Random structure
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            
            for r in range(R):
                # Pick random indices
                u_idx = np.random.randint(nm)
                v_idx = np.random.randint(mp)
                w_idx = np.random.randint(pn)
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
            
            def calc_energy():
                U_sel = U[idx_i, :]
                V_sel = V[idx_j, :]
                W_sel = W[idx_k, :]
                T = np.sum(U_sel * V_sel * W_sel, axis=1)
                return np.sum((T - 1.0)**2)
            
            current_energy = calc_energy()
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = (U.copy(), V.copy(), W.copy())
                
            T = 1.0
            max_iters = 2000
            cooling = 0.999
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return best_state if best_state else None
                
                r = np.random.randint(R)
                move_type = np.random.randint(3) # 0: weight, 1: move U, 2: move V, 3: move W (using mod 3 for simplicity, but let's do 4 types)
                
                # Let's just do weight changes and index swaps
                # Find current non-zero indices for term r
                u_rows = np.nonzero(U[:, r])[0]
                v_rows = np.nonzero(V[:, r])[0]
                w_rows = np.nonzero(W[:, r])[0]
                
                if len(u_rows) == 0 or len(v_rows) == 0 or len(w_rows) == 0:
                    continue
                    
                u_idx = u_rows[0]
                v_idx = v_rows[0]
                w_idx = w_rows[0]
                
                if move_type == 0: # Change weight
                    mat = np.random.randint(3)
                    if mat == 0:
                        old_val = U[u_idx, r]
                        new_val = np.random.choice(domain)
                        if new_val != old_val:
                            U[u_idx, r] = new_val
                            delta = calc_energy() - current_energy
                            if delta > 0 and random.random() > np.exp(-delta / T):
                                U[u_idx, r] = old_val
                            else:
                                current_energy += delta
                    elif mat == 1:
                        old_val = V[v_idx, r]
                        new_val = np.random.choice(domain)
                        if new_val != old_val:
                            V[v_idx, r] = new_val
                            delta = calc_energy() - current_energy
                            if delta > 0 and random.random() > np.exp(-delta / T):
                                V[v_idx, r] = old_val
                            else:
                                current_energy += delta
                    else:
                        old_val = W[w_idx, r]
                        new_val = np.random.choice(domain)
                        if new_val != old_val:
                            W[w_idx, r] = new_val
                            delta = calc_energy() - current_energy
                            if delta > 0 and random.random() > np.exp(-delta / T):
                                W[w_idx, r] = old_val
                            else:
                                current_energy += delta
                else: # Move index
                    # Move U index
                    if move_type == 1:
                        old_u_idx = u_idx
                        new_u_idx = np.random.randint(nm)
                        if new_u_idx != old_u_idx:
                            U[old_u_idx, r] = 0
                            U[new_u_idx, r] = 1.0 # Keep weight 1 for simplicity in move, or copy weight?
                            # Copying weight is better
                            U[new_u_idx, r] = U[old_u_idx, r] # Wait, I zeroed it.
                            # Let's store weight
                            w_val = U[old_u_idx, r]
                            U[old_u_idx, r] = 0
                            U[new_u_idx, r] = w_val
                            
                            delta = calc_energy() - current_energy
                            if delta > 0 and random.random() > np.exp(-delta / T):
                                # Revert
                                U[new_u_idx, r] = 0
                                U[old_u_idx, r] = w_val
                            else:
                                current_energy += delta
                    elif move_type == 2:
                        old_v_idx = v_idx
                        new_v_idx = np.random.randint(mp)
                        if new_v_idx != old_v_idx:
                            w_val = V[old_v_idx, r]
                            V[old_v_idx, r] = 0
                            V[new_v_idx, r] = w_val
                            
                            delta = calc_energy() - current_energy
                            if delta > 0 and random.random() > np.exp(-delta / T):
                                V[new_v_idx, r] = 0
                                V[old_v_idx, r] = w_val
                            else:
                                current_energy += delta
                    else:
                        old_w_idx = w_idx
                        new_w_idx = np.random.randint(pn)
                        if new_w_idx != old_w_idx:
                            w_val = W[old_w_idx, r]
                            W[old_w_idx, r] = 0
                            W[new_w_idx, r] = w_val
                            
                            delta = calc_energy() - current_energy
                            if delta > 0 and random.random() > np.exp(-delta / T):
                                W[new_w_idx, r] = 0
                                W[old_w_idx, r] = w_val
                            else:
                                current_energy += delta
                
                T *= cooling
                
                if current_energy < 1e-9:
                    snapped = snap_half_int(U, V, W)
                    if snapped:
                        return snapped

        return best_state if best_state else None

    best_result = None
    
    # Try ranks from 83 down to 70
    # SA is slower, so fewer ranks or fewer iters
    for R in range(schoolbook_rank - 1, 75, -1):
        if time.time() > deadline:
            break
        
        res = try_structural_sa(R)
        if res is not None:
            best_result = (R, res)
            break
    
    if best_result is not None:
        R, (U, V, W) = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
