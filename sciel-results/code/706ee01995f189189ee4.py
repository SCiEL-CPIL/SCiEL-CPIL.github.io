"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i, j, k] = 1 if (i,j,k) is a valid index
    # The decomposition must satisfy: for all i, j, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since C[i,k] = sum_j A[i,j]*B[j,k], the tensor has T[i,j,k]=1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification using numpy."""
    rank = U.shape[1]
    # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape to tensors
    U_t = U.reshape(n, m, rank)  # U[i, j, c]
    V_t = V.reshape(m, p, rank)  # V[j, k, c]
    W_t = W.reshape(p, n, rank)  # W[k, i, c]
    
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    # This is a 4-index contraction. Let's do it step by step.
    # First: A[i,j,k] = sum_c U[i,j,c] * V[j,k,c]
    # Then: sum_k A[i,j,k] * W[k,i,c]... no, that's not right.
    # Actually we need: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    # Let's just iterate over i,j,k and use numpy for the inner product
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k])
                # Wait, that's not right either. We need element-wise product then sum over c.
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _random_decomposition(n, m, p, rank, seed):
    """Generate a random decomposition with the given rank."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U, V, W

def _refine_decomposition(U, V, W, n, m, p, max_iters=500, tol=1e-10, lr=0.1):
    """Iteratively refine a decomposition to reduce the error."""
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    rank = U.shape[1]
    
    def compute_error():
        err = 0.0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    diff = s - 1.0
                    err += diff * diff
                    max_err = max(max_err, abs(diff))
        return err, max_err
    
    best_err, _ = compute_error()
    
    for it in range(max_iters):
        err, max_err = compute_error()
        if max_err < tol:
            break
        
        # Gradient descent: we want to minimize sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
        # dL/dU[i*m+j, c] = 2 * (s_ij_k - 1) * V[j*p+k, c] * W[k*n+i, c] summed over k
        # But this is complex. Let's use a simpler alternating approach.
        
        # Alternative: treat it as a rank-r approximation problem.
        # For fixed V, W: U[i*m+j, c] should satisfy sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of equations. Let's use least squares.
        
        # For each i, j: we have p equations (one for each k) and rank unknowns U[i*m+j, 0..rank-1]
        # Equation: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for k=0..p-1
        # Let M[i,j] be a p x rank matrix with M[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
        # Then we need: M[i,j] @ U[i*m+j, :]^T = ones(p)
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k] * W[k*n+i]
                # Solve M @ x = ones(p) in least squares sense
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                except:
                    sol = U[i*m+j].copy()
                U[i*m+j] = sol
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j] * W[k*n+i]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                except:
                    sol = V[j*p+k].copy()
                V[j*p+k] = sol
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j] * V[j*p+k]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    sol = W[k*n+i].copy()
                W[k*n+i] = sol
        
        err, max_err = compute_error()
        if err < best_err:
            best_err = err
    
    return U, V, W, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try ranks from lower to higher
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But for matrix multiplication, the rank is at least n*m*p / min(n,m,p) in some cases
    # Let's try from a reasonable lower bound up
    min_rank = max(n*m, m*p, p*n)
    
    seed = 42
    for rank in range(min_rank, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        attempts = 0
        max_attempts = 20 if rank <= 40 else 10
        
        while attempts < max_attempts and not found and time.time() < deadline - 10:
            U, V, W = _random_decomposition(n, m, p, rank, seed)
            seed += 1
            
            U, V, W, err = _refine_decomposition(U, V, W, n, m, p, max_iters=300, tol=1e-12)
            
            if err < 1e-10:
                # Verify
                if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
                    found = True
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            attempts += 1
        
        if found:
            break
    
    # Convert to lists of lists with integer/half-integer entries if possible
    def to_half_integer_matrix(M):
        """Convert a matrix to have half-integer entries if close enough."""
        result = []
        for row in M:
            new_row = []
            for val in row:
                # Check if close to half-integer
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_half) < 1e-8:
                    new_row.append(nearest_half)
                else:
                    # Keep as float but it should be close to rational
                    new_row.append(float(val))
            result.append(new_row)
        return result
    
    return {
        "U": to_half_integer_matrix(best_U),
        "V": to_half_integer_matrix(best_V),
        "W": to_half_integer_matrix(best_W)
    }
# EVOLVE-BLOCK-END
