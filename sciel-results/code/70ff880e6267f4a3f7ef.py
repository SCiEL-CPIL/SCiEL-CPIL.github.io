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
    # i, j, k ranges
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # eq_indices: index in A rows
    eq_indices = i_idx * m * p + j_idx * p + k_idx
    
    # Try ranks from 20 to 55
    for R in range(20, 56):
        if time.time() - start_time > time_limit:
            break
            
        # Restart loop
        # We don't know how many restarts we can do, so we loop until time runs out
        # But we should check time frequently
        
        while True:
            if time.time() - start_time > time_limit:
                break
                
            # Generate random U, V with entries in {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, size=(nm, R))
            V_rand = np.random.randint(-1, 2, size=(mp, R))
            
            # Build A
            A = np.zeros((num_eqs, R * pn), dtype=float)
            
            for c in range(R):
                # Extract U and V columns
                u_col = U_rand[:, c]
                v_col = V_rand[:, c]
                
                # Compute coefficients for all (i, j, k)
                # u_val for (i, j) is u_col[i*m + j]
                # v_val for (j, k) is v_col[j*p + k]
                
                # Vectorized extraction
                u_vals = u_col[i_idx * m + j_idx]
                v_vals = v_col[j_idx * p + k_idx]
                
                coeffs = u_vals * v_vals
                
                # w_indices: index in A columns
                # w_idx = (k * n + i) * R + c
                w_indices = (k_idx * n + i_idx) * R + c
                
                A[eq_indices, w_indices] = coeffs
            
            b = np.ones(num_eqs)
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = W_flat.reshape((pn, R))
            
            # Check if W_sol is close to half-integers
            nearest = np.round(W_sol * 2) / 2.0
            if np.max(np.abs(W_sol - nearest)) < 1e-9:
                # Candidate found, verify
                # Convert to integer if possible
                W_int = nearest.astype(int) if np.all(np.abs(nearest - np.round(nearest)) < 1e-9) else nearest
                
                # Verify decomposition
                valid = True
                # We can vectorize verification too, but loops are fine for small sizes
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s_val = 0
                            for c in range(R):
                                s_val += U_rand[i * m + j, c] * V_rand[j * p + k, c] * W_int[k * n + i, c]
                            if abs(s_val - 1) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                    
                if valid:
                    return {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_int.tolist()
                    }

    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
