"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_U, best_V, best_W = U, V, W

    # Precompute index arrays for vectorized operations
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    def validate_fast(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly represents matrix multiplication using vectorized ops."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]
        
        # Vectorized: results[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        results = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_arr[:, r].reshape(n, m)
            v_col = V_arr[:, r].reshape(m, p)
            w_col = W_arr[:, r].reshape(p, n)
            # For each i, j, k: u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    results[i, j, :] += u_col[i, j] * v_col[j, :] * w_col[:, i]
        
        if not np.allclose(results, 1.0, atol=1e-6):
            return False
        return True

    def build_A(U_rand, V_rand, R):
        """Build the linear system matrix A for solving W."""
        num_eqs = n * m * p
        num_vars = p * n * R
        
        A = np.zeros((num_eqs, num_vars))
        
        # Vectorized construction
        for r in range(R):
            uv_products = U_rand[u_rows, r] * V_rand[v_rows, r]
            # For each equation (i,j,k), the variable is W[k*n+i, r]
            var_indices = w_rows * R + r
            A[np.arange(num_eqs), var_indices] = uv_products
        
        return A

    def try_rank(R, max_attempts=50, verbose=False):
        """Try to find a rank-R decomposition with integer/half-integer entries."""
        if time.time() > deadline - 15:
            return None
        
        A_template = None  # Will be built each time since U,V change
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate diverse random integer/half-integer U, V
            pattern = attempt % 6
            
            if pattern == 0:
                U_rand = np.random.randint(-1, 2, (n * m, R)).astype(float)
                V_rand = np.random.randint(-1, 2, (m * p, R)).astype(float)
            elif pattern == 1:
                U_rand = np.random.choice([-1.0, 1.0], size=(n * m, R))
                V_rand = np.random.choice([-1.0, 1.0], size=(m * p, R))
            elif pattern == 2:
                U_rand = np.random.randint(-2, 3, (n * m, R)).astype(float)
                V_rand = np.random.randint(-2, 3, (m * p, R)).astype(float)
            elif pattern == 3:
                # Sparse: mostly zeros with some ±1
                U_rand = np.zeros((n * m, R))
                V_rand = np.zeros((m * p, R))
                nz_u = np.random.randint(0, 3, (n * m, R))
                nz_v = np.random.randint(0, 3, (m * p, R))
                U_rand[nz_u == 1] = 1.0
                U_rand[nz_u == 2] = -1.0
                V_rand[nz_v == 1] = 1.0
                V_rand[nz_v == 2] = -1.0
            elif pattern == 4:
                # Half-integers for U and V
                U_rand = np.random.randint(-2, 3, (n * m, R)).astype(float) * 0.5
                V_rand = np.random.randint(-2, 3, (m * p, R)).astype(float) * 0.5
            else:
                # Mixed: some columns pure ±1, others random small integers
                U_rand = np.random.randint(-1, 2, (n * m, R)).astype(float)
                V_rand = np.random.randint(-1, 2, (m * p, R)).astype(float)
                # Randomly flip some entries to 0
                U_rand[np.random.rand(n * m, R) < 0.3] = 0.0
                V_rand[np.random.rand(m * p, R) < 0.3] = 0.0
            
            # Build linear system
            A = build_A(U_rand, V_rand, R)
            b = np.ones(n * m * p)
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                W_rand = W_flat.reshape(p * n, R)
                
                # Check if W is close to half-integers
                W_check = W_rand * 2
                dist = np.abs(W_check - np.round(W_check))
                max_dist = np.max(dist)
                
                if max_dist < 1e-4:
                    W_int = np.round(W_check) / 2.0
                    
                    if validate_fast(U_rand.tolist(), V_rand.tolist(), W_int.tolist()):
                        if verbose:
                            print(f"Found rank {R} decomposition!")
                        return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                pass
                
        return None

    # Try ranks from 74 down to 70 with time-proportional effort
    # Allocate more time to lower ranks (more likely to be minimal)
    for R in range(rank_schoolbook - 1, 69, -1):
        if time.time() > deadline - 25:
            break
            
        remaining_time = deadline - time.time() - 10
        if remaining_time < 5:
            break
        
        # More attempts for lower ranks
        max_att = max(20, int(remaining_time * 5))
        result = try_rank(R, max_attempts=max_att)
        if result:
            return result
        
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
