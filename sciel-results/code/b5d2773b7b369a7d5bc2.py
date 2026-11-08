"""Rewritten program: direct rank-1 decomposition search for 3x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0  # seconds to leave before hard kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] = 1 if (i,j,k) valid, else 0.
    Actually for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    C[i, k] = sum_j A[i, j] * B[j, k]
    The tensor T has entries T[i, j, k] = delta (1 if valid)
    We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
    where T[i, j, k] = 1 for all valid i,j,k (since it's just the identity mapping on indices)
    
    Wait, let me reconsider. The tensor for matrix multiplication is:
    T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    So the tensor is actually the identity in the sense that T[i,j,k] = 1 for all (i,j,k) in range.
    
    The decomposition requires:
    sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1  for all i in [0,n), j in [0,m), k in [0,p)
    
    And for the tensor to be exact, this must hold for all valid index combinations.
    """
    # The tensor is simply all ones for valid (i, j, k)
    # We'll work with the flattened structure
    return n, m, p

def _matmul_check(U, V, W, n, m, p):
    """Verify that the decomposition is exact.
    For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    # Efficient check: for each r, compute the outer product contribution
    # T_reconstructed[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = np.array(U).reshape(n, m, rank)
    V_reshaped = np.array(V).reshape(m, p, rank)
    W_reshaped = np.array(W).reshape(p, n, rank)
    
    # For each (i, j, k): sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
    # We can compute this efficiently
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err

def _als_step(U, V, W, n, m, p, rank, max_iter=100, tol=1e-10):
    """Alternating Least Squares to find rank-rank decomposition.
    
    We want: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Fix V and W, solve for U. Fix U and W, solve for V. Fix U and V, solve for W.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    for iteration in range(max_iter):
        # Solve for U: for each (i, j, r), U[i,j,r] appears in equations for all k
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
        # For fixed (i,j): sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1 for all k
        # This is: A_U[i,j] @ U[i,j,:] = ones(p) where A_U[i,j][k, r] = V[j,k,r] * W[k,i,r]
        
        prev_U = U_reshaped.copy()
        for i in range(n):
            for j in range(m):
                # Build A: shape (p, rank)
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V_reshaped[j, k, :] * W_reshaped[k, i, :]
                b = np.ones(p)
                # Solve A @ x = b, where x = U[i,j,:]
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    x = prev_U[i, j, :]
                U_reshaped[i, j, :] = x
        
        # Solve for V: for each (j, k, r), V[j,k,r] appears in equations for all i
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i
        # For fixed (j,k): sum_r V[j,k,r] * (U[i,j,r] * W[k,i,r]) = 1 for all i
        # A_V[j,k][i, r] = U[i,j,r] * W[k,i,r]
        
        prev_V = V_reshaped.copy()
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U_reshaped[i, j, :] * W_reshaped[k, i, :]
                b = np.ones(n)
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    x = prev_V[j, k, :]
                V_reshaped[j, k, :] = x
        
        # Solve for W: for each (k, i, r), W[k,i,r] appears in equations for all j
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all j
        # For fixed (k,i): sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r]) = 1 for all j
        # A_W[k,i][j, r] = U[i,j,r] * V[j,k,r]
        
        prev_W = W_reshaped.copy()
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U_reshaped[i, j, :] * V_reshaped[j, k, :]
                b = np.ones(m)
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    x = prev_W[k, i, :]
                W_reshaped[k, i, :] = x
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                    if abs(val - 1.0) > err:
                        err = abs(val - 1.0)
        
        if err < tol:
            break
        if err > 1e6:
            # Diverged
            break
    
    return U_reshaped, V_reshaped, W_reshaped, err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook decomposition as initial guess
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    best_err = 0.0
    
    # Try ranks from low to high
    # Theoretical lower bound: rank >= max(n*p, n*m, m*p) / something... 
    # For matrix multiplication, known bounds suggest rank around n*m*p is schoolbook
    # But there might be better decompositions
    # Let's try ranks from a reasonable minimum up to schoolbook
    
    # Minimum possible rank is at least max(n, m, p) * something
    # Actually for C = AB, the tensor rank is at least max(n*p, n*m, m*p) in some sense
    # But let's just try a range
    
    min_rank = max(n, m, p)  # Very optimistic lower bound
    # Actually, let's be more realistic. Try from around 30 up to 90
    ranks_to_try = list(range(30, rank_schoolbook + 1))
    
    for rank in ranks_to_try:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Initialize with random values
        np.random.seed(rank * 1000)  # Deterministic seed per rank
        U_init = np.random.randn(n * m, rank) * 0.1
        V_init = np.random.randn(m * p, rank) * 0.1
        W_init = np.random.randn(p * n, rank) * 0.1
        
        # Also try initialization based on schoolbook (take first 'rank' columns)
        if rank <= rank_schoolbook:
            U_init2 = U_sb[:, :rank].copy()
            V_init2 = V_sb[:, :rank].copy()
            W_init2 = W_sb[:, :rank].copy()
        else:
            U_init2 = U_init.copy()
            V_init2 = V_init.copy()
            W_init2 = W_init.copy()
        
        # Try both initializations
        for init_idx, (U_i, V_i, W_i) in enumerate([(U_init, V_init, W_init), (U_init2, V_init2, W_init2)]):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            U_r, V_r, W_r, err = _als_step(U_i, V_i, W_i, n, m, p, rank, max_iter=200, tol=1e-12)
            
            if err < 1e-8:
                # Found a valid decomposition!
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank
                best_err = err
                break
            elif err < best_err:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank
                best_err = err
        
        if best_rank == rank and best_err < 1e-8:
            break
    
    # Convert to lists
    U_list = [[float(val) for val in row] for row in best_U]
    V_list = [[float(val) for val in row] for row in best_V]
    W_list = [[float(val) for val in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
