"""Tensor decomposition for 3x4 by 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    We use Alternating Least Squares (ALS) to attempt to find a decomposition
    of lower rank than the schoolbook (84). If no exact low-rank decomposition
    is found, we return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    t_start = time.time()
    deadline = t_start + 145.0
    
    # Schoolbook baseline: rank = n*m*p = 84
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Target ranks to try
    # Lower bound is 28.
    # We try a range of ranks.
    target_ranks = range(28, 84)
    
    # Pre-compute the target tensor T flattened?
    # T is a 3D tensor of shape (n*m, m*p, p*n) ? No.
    # The tensor T[i,j,k] = 1.
    # In the decomposition:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # We can view this as a linear system for W given U, V.
    # Let A_c = U[:, c] (shape n*m)
    # Let B_c = V[:, c] (shape m*p)
    # Let C_c = W[:, c] (shape p*n)
    # The equation is: sum_c (A_c \otimes B_c) * C_c^T? No.
    # The tensor T is indexed by (i,j,k).
    # The term for c is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # This is a 3-mode tensor.
    
    # For ALS:
    # Fix U, V. Solve for W.
    # The equation for a specific (i,j,k) is:
    # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1
    # This is a system of linear equations for W.
    # Let M be a matrix where rows correspond to (i,j,k) triples and columns to c.
    # M[(i,j,k), c] = U[i*m+j, c] * V[j*p+k, c]
    # Then M * W_flat = 1_vector.
    # W_flat has length (p*n * R).
    # Number of equations is n*m*p = 84.
    # If R < 84, the system is underdetermined. We can find a solution with minimum norm?
    # But we need EXACT solution.
    # If R is the true rank, a solution exists.
    # We can use np.linalg.lstsq to find a solution, then check residual.
    
    def try_als_decomposition(R, seed=42):
        np.random.seed(seed)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        
        # Construct the linear system matrix M
        # M has shape (n*m*p, R)
        # Index mapping: idx = ((i*m + j)*p + k)
        # U_idx = i*m + j
        # V_idx = j*p + k
        # W_idx = k*n + i
        
        M = np.zeros((n * m * p, R))
        # We need to fill M efficiently
        # Loop over i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    M[idx] = u_val * v_val
        
        # Solve M * W_flat = 1
        # W_flat is length R * p * n
        # Wait, the column c of W is W[:, c].
        # The equation is sum_c M[idx, c] * W_flat[c * p * n + w_idx] = 1 ?
        # No.
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Let's group by c.
        # For a fixed c, the term is W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]).
        # This couples W entries across different (i,j,k) for the same c.
        # Actually, for a fixed c, W[:, c] is a vector of length p*n.
        # The equation for (i,j,k) involves W[k*n+i, c].
        # So for a fixed c, the variable is W[k*n+i, c].
        # The coefficient is U[i*m+j, c] * V[j*p+k, c].
        
        # This is not a simple linear system for all W at once because the mapping (i,j,k) -> (c, w_idx) is complex.
        # Let's re-formulate.
        # We want to find W such that T = sum_c U_c \otimes V_c \otimes W_c.
        # This is a tensor equation.
        # We can solve for W_c one by one? No, they are coupled in the sum.
        
        # Alternative: Solve for W given U, V using least squares on the whole tensor.
        # Flatten the tensor T (shape n, m, p) to a vector of length n*m*p.
        # Flatten the decomposition term for c:
        # Term_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        # This is not linear in W if we treat W as a matrix, because W[k*n+i, c] is a specific element.
        # However, for a fixed c, the vector of values W_c = W[:, c] is linear.
        # The contribution of c to the tensor is:
        # T_c[i,j,k] = (U[i*m+j, c] * V[j*p+k, c]) * W_c[k*n+i].
        # Let A_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c].
        # Then T_c = A_c * W_c (element-wise product? No, W_c is indexed by k,n,i).
        # The index of W_c is (k*n+i).
        # So T_c[i,j,k] = A_c[i,j,k] * W_c[k*n+i].
        
        # We want sum_c T_c[i,j,k] = 1.
        # This is a linear system for the variables W_c[k*n+i].
        # Total variables: R * p * n.
        # Total equations: n * m * p.
        
        # Let's build the matrix A_system.
        # Rows: (i, j, k)
        # Cols: (c, w_idx) where w_idx = k*n+i.
        # A_system[(i,j,k), (c, w_idx)] = 1 if w_idx == k*n+i else 0.
        # Value: U[i*m+j, c] * V[j*p+k, c].
        
        # This matrix is sparse.
        # For a fixed (i,j,k), only R columns are non-zero (one for each c).
        # Specifically, column (c, k*n+i) has value U[i*m+j, c] * V[j*p+k, c].
        
        # We can use np.linalg.lstsq on this system.
        
        # Construct A_system
        num_eq = n * m * p
        num_var = R * p * n
        A_system = np.zeros((num_eq, num_var))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_idx = (i * m + j) * p + k
                    for c in range(R):
                        col_idx = c * p * n + (k * n + i)
                        A_system[row_idx, col_idx] = U[i * m + j, c] * V[j * p + k, c]
        
        b = np.ones(num_eq)
        
        # Solve
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A_system, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
            
        # Check residual
        if len(residuals) > 0 and residuals[0] > 1e-9:
            return None
            
        # Reshape W_flat to W
        W = W_flat.reshape(R, p * n).T # W is (p*n, R)
        
        # Verify exactness
        # Reconstruct and check
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return None
                    error += abs(val - 1.0)
        
        return U, V, W

    # Try ALS for various ranks
    # We use a few random seeds for each rank
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
                
            res = try_als_decomposition(R, seed)
            if res is not None:
                U, V, W = res
                best_rank = R
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break
            
    if best_rank == rank_schoolbook:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
