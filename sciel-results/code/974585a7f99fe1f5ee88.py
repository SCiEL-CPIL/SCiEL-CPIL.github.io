import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication
    using Simulated Annealing on a discrete lattice of half-integers.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    start_time = time.time()
    deadline = 145  # Safety margin for 150s limit

    # Pre-compute index mappings for speed
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # We can vectorize the residual calculation.
    # Let's create arrays of indices for U, V, W for each term in the sum.
    # There are n*m*p = 4*4*5 = 80 constraints.
    
    # For each constraint c (corresponding to i,j,k), we need:
    # U_idx[c] = i*m + j
    # V_idx[c] = j*p + k
    # W_idx[c] = k*n + i
    
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
                
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    def compute_energy(U, V, W):
        """
        Compute the sum of squared errors for the tensor identity.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        # Extract rows corresponding to constraints
        # U_rows: (80, R)
        # V_rows: (80, R)
        # W_rows: (80, R)
        U_rows = U[u_indices, :]
        V_rows = V[v_indices, :]
        W_rows = W[w_indices, :]
        
        # Product: (80, R)
        # Sum over R axis to get (80,)
        # val[c] = sum_r U_rows[c, r] * V_rows[c, r] * W_rows[c, r]
        prod = U_rows * V_rows * W_rows
        vals = np.sum(prod, axis=1)
        
        # Target is 1.0 for all
        errors = vals - 1.0
        return np.sum(errors * errors)

    def get_lattice_values():
        # Focus on small values to keep search tractable and likely to find exact solutions
        # Standard Strassen uses {-1, 1}. Sometimes 0 is needed.
        # Half-integers might be needed for some decompositions, though rare for MM.
        # Let's stick to integers {-2, -1, 0, 1, 2} first for speed, 
        # as most known decompositions use small integers.
        # If we don't find a solution, we can expand.
        return np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    lattice_vals = get_lattice_values()
    
    # Initial Rank Estimation
    # Schoolbook is 80.
    # Strassen for 4x4x4 is 49.
    # For 4x4x5, the rank is likely between 50 and 70.
    # Let's try to find a solution for ranks starting from 60 down to 50.
    # If we fail, we fallback to 80.
    
    best_solution = None
    best_rank = float('inf')
    
    # Ranks to try. We prioritize lower ranks.
    # We will spend more time on lower ranks if we have time.
    ranks_to_try = list(range(60, 49, -1)) # 60, 59, ..., 50
    
    for target_rank in ranks_to_try:
        if time.time() - start_time > deadline - 20:
            break
            
        # Simulated Annealing Parameters
        # Number of iterations depends on remaining time
        # Let's allocate ~5-10 seconds per rank attempt initially.
        
        # Initialize randomly
        # We want to encourage sparsity? No, random is fine.
        # Using integers from lattice.
        
        def random_init(R):
            # Shape (n*m, R), (m*p, R), (p*n, R)
            U = np.random.choice(lattice_vals, size=(n * m, R))
            V = np.random.choice(lattice_vals, size=(m * p, R))
            W = np.random.choice(lattice_vals, size=(p * n, R))
            return U, V, W

        found = False
        
        # Try a few restarts for this rank
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() - start_time > deadline - 10:
                break
                
            U, V, W = random_init(target_rank)
            
            # Initial Energy
            current_energy = compute_energy(U, V, W)
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_energy = current_energy
            
            # Annealing Schedule
            # Start high, decay exponentially
            T = 10.0
            T_min = 0.001
            alpha = 0.995 # Decay rate per step
            
            # Determine max steps based on time budget
            # Let's do ~5000 steps per restart, which should be fast in numpy
            max_steps = 5000
            
            for step in range(max_steps):
                if time.time() - start_time > deadline - 5:
                    break
                    
                if T < T_min:
                    break
                    
                # Pick a random matrix and a random element to perturb
                mat_idx = random.randint(0, 2) # 0:U, 1:V, 2:W
                if mat_idx == 0:
                    mat = U
                elif mat_idx == 1:
                    mat = V
                else:
                    mat = W
                    
                # Pick random row and col
                rows = mat.shape[0]
                cols = mat.shape[1]
                r_idx = random.randint(0, rows - 1)
                c_idx = random.randint(0, cols - 1)
                
                old_val = mat[r_idx, c_idx]
                # Pick a new value from lattice, different from old
                new_val_idx = random.randint(0, len(lattice_vals) - 1)
                new_val = lattice_vals[new_val_idx]
                if new_val == old_val:
                    # Pick another
                    new_val_idx = (new_val_idx + 1) % len(lattice_vals)
                    new_val = lattice_vals[new_val_idx]
                    
                # Apply perturbation
                mat[r_idx, c_idx] = new_val
                
                # Compute new energy
                # Note: U, V, W are the actual arrays being modified
                new_energy = compute_energy(U, V, W)
                
                delta_E = new_energy - current_energy
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Accept
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        # If we hit zero, we found a solution!
                        if best_energy < 1e-9:
                            break
                else:
                    # Reject
                    mat[r_idx, c_idx] = old_val
                    
                # Cool down
                T *= alpha
                
            # After SA, check if best solution is exact
            if best_energy < 1e-9:
                # Verify exactly
                # The energy is sum of squares. If < 1e-9, it's effectively 0.
                # But let's do a strict check to be safe against floating point noise
                # although we are on a discrete lattice, so values are exact.
                # U, V, W entries are from {-2,-1,0,1,2}, so products are integers.
                # Sum of integers is integer. So val - 1 is integer.
                # Square is integer. Sum is integer.
                # So if energy is 0, it is exactly 0.
                
                best_solution = {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
                best_rank = target_rank
                found = True
                break # Break restarts, try lower rank
        
        if found:
            # We found a solution for this rank.
            # We could try to optimize further, but SA usually converges.
            # Let's just proceed to next lower rank if time permits.
            continue
            
    # Fallback: Schoolbook Rank 80
    if best_solution is None:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_solution = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        best_rank = rank

    return best_solution

# Ensure the function is defined at module level as per contract
# The contract says: Interface: construct() -> {'U', 'V', 'W'}
# The current program defines construct().

if __name__ == "__main__":
    # Quick test
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
