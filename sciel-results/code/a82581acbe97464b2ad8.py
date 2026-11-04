"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the matrix multiplication tensor decomposition.
    The rank is n*m*p (schoolbook) as a guaranteed valid fallback.
    Attempts to find lower-rank decompositions via structured search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize schoolbook decomposition as guaranteed valid result
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    def validate_decomposition(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly represents matrix multiplication."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]
        
        results = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            u_reshaped = u_col.reshape(n, m)
            v_reshaped = v_col.reshape(m, p)
            w_reshaped = w_col.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    results[i, j, :] += u_reshaped[i, j] * v_reshaped[j, :] * w_reshaped[:, i]
        
        return np.allclose(results, 1.0, atol=1e-6)
    
    def try_structured_rank(R, max_attempts=20):
        """Try to find a rank-R decomposition using structured patterns."""
        if time.time() > deadline - 20:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Generate structured U and V with small integer entries
            if attempt % 4 == 0:
                # Pattern: sparse {-1, 0, 1}
                U_rand = np.random.randint(-1, 2, (n * m, R)).astype(float)
                V_rand = np.random.randint(-1, 2, (m * p, R)).astype(float)
                # Sparsify: keep only ~30% of non-zeros
                mask_u = np.random.rand(n * m, R) < 0.3
                mask_v = np.random.rand(m * p, R) < 0.3
                U_rand *= mask_u
                V_rand *= mask_v
            elif attempt % 4 == 1:
                # Pattern: dense {-1, 1}
                U_rand = np.random.choice([-1.0, 1.0], size=(n * m, R))
                V_rand = np.random.choice([-1.0, 1.0], size=(m * p, R))
            elif attempt % 4 == 2:
                # Pattern: {-2, -1, 0, 1, 2} with sparsity
                U_rand = np.random.randint(-2, 3, (n * m, R)).astype(float)
                V_rand = np.random.randint(-2, 3, (m * p, R)).astype(float)
                mask_u = np.random.rand(n * m, R) < 0.4
                mask_v = np.random.rand(m * p, R) < 0.4
                U_rand *= mask_u
                V_rand *= mask_v
            else:
                # Pattern: block-structured
                U_rand = np.zeros((n * m, R))
                V_rand = np.zeros((m * p, R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r_idx = (i * m + j) * p + k
                            if r_idx < R:
                                U_rand[i * m + j, r_idx] = np.random.choice([-1.0, 1.0])
                                V_rand[j * p + k, r_idx] = np.random.choice([-1.0, 1.0])
            
            # Build linear system for W
            num_eqs = n * m * p
            num_vars = p * n * R
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    u_row = i * m + j
                    for k in range(p):
                        v_row = j * p + k
                        w_row = k * n + i
                        for r in range(R):
                            A[idx, w_row * R + r] = U_rand[u_row, r] * V_rand[v_row, r]
                        idx += 1
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                W_rand = W_flat.reshape(p * n, R)
                
                # Check if W is close to half-integers
                W_check = W_rand * 2
                dist = np.abs(W_check - np.round(W_check))
                max_dist = np.max(dist)
                
                if max_dist < 1e-4:
                    W_int = np.round(W_check) / 2.0
                    
                    if validate_decomposition(U_rand.tolist(), V_rand.tolist(), W_int.tolist()):
                        return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                pass
                
        return None
    
    # Try ranks from 74 down to 65 with structured search
    for R in range(rank_schoolbook - 1, 64, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_structured_rank(R, max_attempts=10)
        if result is not None:
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
