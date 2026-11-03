import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    num_eqs = n * m * p
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=int)
    V_school = np.zeros((mp, rank_school), dtype=int)
    W_school = np.zeros((pn, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    start_time = time.time()
    time_limit = 115
    
    # Precompute indices for vectorized A construction
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # eq_indices: index in A rows
    eq_indices = i_idx * m * p + j_idx * p + k_idx
    base_col = (k_idx * n + i_idx) * 0 # Placeholder, will be scaled by R
    
    # Try ranks from 30 to 59
    # Start lower to find better solutions, but 30 is a reasonable lower bound guess
    # If not found, increase.
    for R in range(30, 60):
        if time.time() - start_time > time_limit:
            break
            
        # Precompute column indices for this R
        # col_indices[r, c] = (k_idx[r]*n + i_idx[r])*R + c
        base_col_R = (k_idx * n + i_idx) * R
        col_indices = base_col_R[:, np.newaxis] + np.arange(R)[np.newaxis, :]
        
        # Restart loop
        while True:
            if time.time() - start_time > time_limit:
                break
                
            # Generate random U, V with entries in {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, size=(nm, R))
            V_rand = np.random.randint(-1, 2, size=(mp, R))
            
            # Build A using einsum for speed
            u_reshaped = U_rand.reshape(n, m, R)
            v_reshaped = V_rand.reshape(m, p, R)
            
            # coeffs_4d[i, j, k, c] = U[i, j, c] * V[j, k, c]
            coeffs_4d = np.einsum('ijc,jkc->ijkc', u_reshaped, v_reshaped)
            coeffs_flat = coeffs_4d.reshape(num_eqs, R)
            
            A = np.zeros((num_eqs, R * pn), dtype=float)
            A[eq_indices[:, np.newaxis], col_indices] = coeffs_flat
            
            b = np.ones(num_eqs)
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = W_flat.reshape((pn, R))
            
            # Check if W_sol is close to half-integers
            # Use a slightly relaxed tolerance for detection, then verify exactly
            nearest = np.round(W_sol * 2) / 2.0
            if np.max(np.abs(W_sol - nearest)) < 1e-6:
                # Candidate found, verify
                # Convert to integer if possible
                W_int = nearest.astype(int) if np.all(np.abs(nearest - np.round(nearest)) < 1e-9) else nearest
                
                # Verify decomposition using vectorized operations
                # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
                # U_rand (nm, R), V_rand (mp, R), W_int (pn, R)
                
                # Reshape for einsum
                u_v = U_rand.reshape(n, m, R)
                v_v = V_rand.reshape(m, p, R)
                w_v = W_int.reshape(p, n, R)
                
                # product[i, j, k, c] = u[i, j, c] * v[j, k, c] * w[k, i, c]
                # einsum: ijc, jkc, kic -> ijk
                try:
                    result = np.einsum('ijc,jkc,kic->ijk', u_v, v_v, w_v)
                    if np.allclose(result, 1.0, atol=1e-9):
                        return {
                            "U": U_rand.tolist(),
                            "V": V_rand.tolist(),
                            "W": W_int.tolist()
                        }
                except:
                    continue

    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
