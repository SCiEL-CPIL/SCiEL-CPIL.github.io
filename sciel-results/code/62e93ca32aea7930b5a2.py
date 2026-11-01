import time
import numpy as np
import random

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    target_arr = np.array(target_indices)
    a_idx = target_arr[:, 0]
    b_idx = target_arr[:, 1]
    c_idx = target_arr[:, 2]

    def is_valid(U_mat, V_mat, W_mat, rank):
        U_t = U_mat[a_idx, :rank]
        V_t = V_mat[b_idx, :rank]
        W_t = W_mat[c_idx, :rank]
        prod = U_t * V_t * W_t
        sums = np.sum(prod, axis=1)
        return np.all(np.abs(sums - 1.0) < 1e-9)

    def snap_to_half_integers(mat):
        # Check if mat * 2 is close to integers
        scaled = mat * 2
        rounded = np.round(scaled)
        if np.all(np.abs(scaled - rounded) < 1e-6):
            return rounded / 2.0
        return None

    # Schoolbook fallback
    R0 = n * m * p
    U0 = np.zeros((nm, R0), dtype=float)
    V0 = np.zeros((mp, R0), dtype=float)
    W0 = np.zeros((pn, R0), dtype=float)
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col_idx] = 1.0
                V0[j * p + k, col_idx] = 1.0
                W0[k * n + i, col_idx] = 1.0
                col_idx += 1
    best_U, best_V, best_W, best_R = U0.copy(), V0.copy(), W0.copy(), R0

    # Try to find lower rank using algebraic construction
    # Target rank 3p+1 = 40
    target_ranks = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    
    for R in target_ranks:
        if time.time() > deadline - 10:
            break
            
        attempts = 20
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
                
            # Generate random integer U, V
            U_rand = np.random.randint(-1, 2, (nm, R))
            V_rand = np.random.randint(-1, 2, (mp, R))
            
            # Ensure no row is all zeros
            if np.any(np.all(U_rand == 0, axis=1)):
                continue
            if np.any(np.all(V_rand == 0, axis=1)):
                continue
                
            W_sol = np.zeros((pn, R), dtype=float)
            valid_solution = True
            
            # Solve for W for each c
            # For each c, we have 2 equations (for j=0, j=1)
            # Sum_r U[a_j, r] V[b_j, r] W[c, r] = 1
            # Let M_c be a 2 x R matrix where M_c[j, r] = U[a_j, r] * V[b_j, r]
            # Then M_c @ W_c = [1, 1]^T
            
            for c in range(pn):
                # Find valid (a, b) for this c
                # c = k * n + i
                # a = i * m + j
                # b = j * p + k
                # j can be 0 or 1
                i = c % n
                k = c // n
                
                M_c = np.zeros((m, R))
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    M_c[j, :] = U_rand[a, :] * V_rand[b, :]
                
                # Solve M_c @ W_c = [1, 1]
                # Use least squares
                try:
                    W_c, residuals, rank, sv = np.linalg.lstsq(M_c, np.ones(m), rcond=None)
                    # Check if it's an exact solution
                    if np.all(np.abs(M_c @ W_c - np.ones(m)) > 1e-6):
                        valid_solution = False
                        break
                    W_sol[c, :] = W_c
                except:
                    valid_solution = False
                    break
                    
            if not valid_solution:
                continue
                
            # Snap to half-integers
            W_snapped = snap_to_half_integers(W_sol)
            if W_snapped is None:
                continue
                
            # Verify
            if is_valid(U_rand, V_rand, W_snapped, R):
                best_U, best_V, best_W, best_R = U_rand.copy(), V_rand.copy(), W_snapped.copy(), R
                break # Found a valid solution for this rank, break to try lower rank? No, R is descending?
                # We are iterating R from 40 upwards. So 40 is best.
                # If we find 40, we can stop.
                break
        
        if best_R < R0:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
