"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    The decomposition is defined by three matrices U, V, W of shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Such that for all i, j, k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    num_eqs = n * m * p  # 84
    
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank), dtype=int)
        V = np.zeros((dim_V, rank), dtype=int)
        W = np.zeros((dim_W, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, r):
        """Verify exact decomposition with half-integer entries."""
        # Check all entries are half-integers
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        # Verify the tensor equation
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :r] * V[j * p + k, :r], W[k * n + i, :r])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Precompute index arrays for faster construction
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    U_rows = i_idx * m + j_idx
    V_rows = j_idx * p + k_idx
    W_rows = k_idx * n + i_idx
    
    # Search for better rank
    min_rank = max(dim_U, dim_V, dim_W)  # 28
    max_rank_to_try = 83
    
    for r in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 15:
            break
        
        # Number of random tries per rank
        # Lower ranks are harder, so try more
        if r < 40:
            num_tries = 30
        elif r < 60:
            num_tries = 15
        else:
            num_tries = 5
        
        for _ in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            # Initialize U and V with sparse random integers {-1, 0, 1}
            # Use sparsity to make the system easier
            U = np.zeros((dim_U, r), dtype=int)
            V = np.zeros((dim_V, r), dtype=int)
            
            # Fill with random entries, ensuring each column has some non-zeros
            for c in range(r):
                # Randomly choose which rows to set non-zero
                num_nonzero_U = max(1, np.random.randint(1, dim_U // 2 + 1))
                rows_U = np.random.choice(dim_U, size=num_nonzero_U, replace=False)
                U[rows_U, c] = np.random.choice([-1, 1], size=num_nonzero_U)
                
                num_nonzero_V = max(1, np.random.randint(1, dim_V // 2 + 1))
                rows_V = np.random.choice(dim_V, size=num_nonzero_V, replace=False)
                V[rows_V, c] = np.random.choice([-1, 1], size=num_nonzero_V)
            
            # Build the linear system A W = b
            # A has shape (num_eqs, dim_W * r)
            # b has shape (num_eqs,)
            # Equation for (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            A = np.zeros((num_eqs, dim_W * r))
            b = np.ones(num_eqs)
            
            for eq in range(num_eqs):
                u_row = U_rows[eq]
                v_row = V_rows[eq]
                w_row = W_rows[eq]
                for c in range(r):
                    coeff = U[u_row, c] * V[v_row, c]
                    A[eq, w_row * r + c] = coeff
            
            # Solve A W_flat = b using least squares
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
            
            # Reshape W_flat to W
            W_sol = W_flat.reshape((dim_W, r))
            
            # Check if W_sol is close to half-integers
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            
            if np.max(np.abs(W_scaled - W_rounded)) < 1e-6:
                # Candidate found! Check if it's exact
                W_int = W_rounded / 2.0
                
                if verify(U, V, W_int, r):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}
    
    # If no better rank found, return schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
