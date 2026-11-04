import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Start with schoolbook as safe fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to find lower rank decompositions using alternating least squares
    # Search from rank 40 up to best_rank - 1
    for rank in range(40, best_rank):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            # Continue searching for even lower rank
            # But don't go below 40 to save time
    
    return best_result

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-rank decomposition using alternating least squares."""
    # Build the matrix multiplication tensor in flattened form
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta (correct product)
    
    # The tensor entries: T[i*m+j, j*p+k, k*n+i] = 1 for the multiplication
    # Actually, let's think of it as: for each (i,j,k), we have a term A[i,j]*B[j,k] contributing to C[i,k]
    
    # Flattened indices:
    # A_flat index: i*m + j (range 0 to n*m-1)
    # B_flat index: j*p + k (range 0 to m*p-1)  
    # C_flat index: i*p + k (range 0 to n*p-1) -- but W uses k*n+i per spec
    
    # The constraint is: for each (i,k), sum_j A[i,j]*B[j,k] = C[i,k]
    # In tensor form: sum_r U[i,j,r] * V[j,k,r] = coefficient for C[i,k]
    
    # Let's use a different approach: represent the multiplication as a linear map
    # and use ALS on the tensor
    
    np.random.seed(42)
    
    # Initialize factor matrices
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # The target tensor: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if the triple (i,j,k) is valid (which it always is for valid indices)
    # Actually, the multiplication tensor has T[i,j,k] = 1 meaning A[i,j]*B[j,k] contributes to C[i,k]
    
    # Let me reformulate: We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
    # to equal the indicator that this is a valid multiplication term.
    
    # For matrix multiplication, every triple (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
    # contributes A[i,j]*B[j,k] to C[i,k]. So the tensor is all ones on valid indices.
    
    # But we need: sum_r U[i*m+j, r] * V[j*p+k, r] = 1 for all (i,j,k)
    # And W is determined by consistency... 
    
    # Actually, the standard formulation is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_r U[i,j,r] * V[j,k,r] * (some factor)
    
    # Let me use the correct tensor decomposition formulation:
    # We need matrices U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    # for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since each term A[i,j]*B[j,k] has coefficient 1)
    
    # This is a 3-way tensor factorization where the target tensor T has:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # T[a,b,c] = 0 otherwise
    
    # Build target tensor (sparse)
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # ALS iterations
    max_iter = 50
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Update U: for each (i,j), solve least squares
        # sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c]
        # For fixed a, this is: sum_r U[a,r] * (V[b,r] * W[c,r]) = target[a,b,c]
        # Group by (b,c): for each a, we have equations over all (b,c)
        
        # Efficient update: 
        # For U: fix a, the equation is sum_r U[a,r] * S1[b,c,r] = target[a,b,c]
        # where S1[b,c,r] = V[b,r] * W[c,r]
        # This is a linear system: U[a,:] @ (V[b,:] * W[c,:])^T = target[a,b,c]
        
        # Let's do it more carefully. For each a:
        # We want to find U[a,:] such that for all (b,c):
        # sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c]
        
        # This can be written as: 
        # Let M_a be a matrix where M_a[(b,c), r] = V[b,r] * W[c,r]
        # Then U[a,:] @ M_a = target[a,:,:].ravel()
        
        # Similarly for V and W.
        
        # Update U
        for a in range(n * m):
            # Build M_a: (m*p * p*n) x rank
            M_a = np.zeros((m * p * p * n, rank))
            rhs = np.zeros(m * p * p * n)
            idx = 0
            for b in range(m * p):
                for c in range(p * n):
                    M_a[idx, :] = V[b, :] * W[c, :]
                    rhs[idx] = target[a, b, c]
                    idx += 1
            # Solve least squares: U[a,:] @ M_a^T = rhs
            # Or: M_a @ U[a,:]^T = rhs
            U[a, :], _, _, _ = np.linalg.lstsq(M_a, rhs, rcond=None)
        
        # Update V
        for b in range(m * p):
            M_b = np.zeros((n * m * p * n, rank))
            rhs = np.zeros(n * m * p * n)
            idx = 0
            for a in range(n * m):
                for c in range(p * n):
                    M_b[idx, :] = U[a, :] * W[c, :]
                    rhs[idx] = target[a, b, c]
                    idx += 1
            V[b, :], _, _, _ = np.linalg.lstsq(M_b, rhs, rcond=None)
        
        # Update W
        for c in range(p * n):
            M_c = np.zeros((n * m * m * p, rank))
            rhs = np.zeros(n * m * m * p)
            idx = 0
            for a in range(n * m):
                for b in range(m * p):
                    M_c[idx, :] = U[a, :] * V[b, :]
                    rhs[idx] = target[a, b, c]
                    idx += 1
            W[c, :], _, _, _ = np.linalg.lstsq(M_c, rhs, rcond=None)
        
        # Check convergence
        if it > 0:
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        error += (val - 1.0) ** 2
            if error < 1e-10:
                # Found a valid decomposition
                return _rationalize(U, V, W)
    
    return None

def _rationalize(U, V, W):
    """Convert float matrices to rational (integer or half-integer) if possible."""
    # Try to find rational approximations
    from fractions import Fraction
    
    def rationalize_matrix(M, max_denom=100):
        result = [[0.0] * M.shape[1] for _ in range(M.shape[0])]
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                # Try to find a/b close to val with b <= max_denom
                best = (0, 1, abs(val))
                for b in range(1, max_denom + 1):
                    a = round(val * b)
                    err = abs(val - a / b)
                    if err < best[2]:
                        best = (a, b, err)
                a, b, err = best
                if err < 1e-8:
                    result[i][j] = a / b
                else:
                    result[i][j] = val
        return result
    
    U_rat = rationalize_matrix(U)
    V_rat = rationalize_matrix(V)
    W_rat = rationalize_matrix(W)
    
    return {"U": U_rat, "V": V_rat, "W": W_rat}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
