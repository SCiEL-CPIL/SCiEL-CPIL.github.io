"""Program for 3x4 * 4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    rank = U.shape[1]
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Tensor: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)

def _als_step_U(V, W, n, m, p, tol=1e-10):
    """
    Solve for U given V and W.
    For each (i,j): sum_k sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
    Let A[i,j,r] = sum_k V[j*p+k,r] * W[k*n+i,r]
    Then sum_r U[i*m+j,r] * A[i,j,r] = 1
    This is: U[i*m+j,:] @ A[i,j,:] = 1
    """
    rank = V.shape[1]
    U = np.zeros((n*m, rank))
    for i in range(n):
        for j in range(m):
            # A[i,j,r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            # V[j*p:(j+1)*p, :] has shape (p, rank)
            # W[:, i, :] from W3 has shape (p, rank)
            Vj = V[j*p:(j+1)*p, :]  # (p, rank)
            W_i = W.reshape(p, n, rank)[:, i, :]  # (p, rank)
            A = np.sum(Vj * W_i, axis=0)  # (rank,)
            # Solve: U[i*m+j,:] @ A = 1
            # If A is nonzero, U[i*m+j,:] = A / (A@A)
            dot = np.dot(A, A)
            if abs(dot) > tol:
                U[i*m+j, :] = A / dot
            else:
                U[i*m+j, :] = 0
    return U

def _als_step_V(U, W, n, m, p, tol=1e-10):
    """
    Solve for V given U and W.
    For each (j,k): sum_i sum_r (U[i*m+j,r] * W[k*n+i,r]) * V[j*p+k,r] = 1
    Let B[j,k,r] = sum_i U[i*m+j,r] * W[k*n+i,r]
    Then sum_r V[j*p+k,r] * B[j,k,r] = 1
    """
    rank = U.shape[1]
    V = np.zeros((m*p, rank))
    U3 = U.reshape(n, m, rank)
    W3 = W.reshape(p, n, rank)
    for j in range(m):
        for k in range(p):
            # B[j,k,r] = sum_i U3[i,j,r] * W3[k,i,r]
            B = np.sum(U3[:, j, :] * W3[k, :, :], axis=0)  # (rank,)
            dot = np.dot(B, B)
            if abs(dot) > tol:
                V[j*p+k, :] = B / dot
            else:
                V[j*p+k, :] = 0
    return V

def _als_step_W(U, V, n, m, p, tol=1e-10):
    """
    Solve for W given U and V.
    For each (k,i): sum_j sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
    Let C[k,i,r] = sum_j U[i*m+j,r] * V[j*p+k,r]
    Then sum_r W[k*n+i,r] * C[k,i,r] = 1
    """
    rank = U.shape[1]
    W = np.zeros((p*n, rank))
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    for k in range(p):
        for i in range(n):
            # C[k,i,r] = sum_j U3[i,j,r] * V3[j,k,r]
            C = np.sum(U3[i, :, :] * V3[:, k, :], axis=0)  # (rank,)
            dot = np.dot(C, C)
            if abs(dot) > tol:
                W[k*n+i, :] = C / dot
            else:
                W[k*n+i, :] = 0
    return W

def _alternating_minimization(n, m, p, target_rank, max_iters=500, tol=1e-10, seed=0):
    """
    Run alternating least squares to find a decomposition of rank target_rank.
    Returns (U, V, W) if successful, else None.
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, target_rank)
    V = rng.randn(m * p, target_rank)
    W = rng.randn(p * n, target_rank)
    
    for it in range(max_iters):
        U_new = _als_step_U(V, W, n, m, p)
        V_new = _als_step_V(U_new, W, n, m, p)
        W_new = _als_step_W(U_new, V_new, n, m, p)
        
        # Check convergence
        U_diff = np.max(np.abs(U_new - U))
        V_diff = np.max(np.abs(V_new - V))
        W_diff = np.max(np.abs(W_new - W))
        
        U, V, W = U_new, V_new, W_new
        
        if max(U_diff, V_diff, W_diff) < tol:
            # Check if we achieved the target
            if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                return U, V, W
            break
    
    # Final check
    if _verify_fast(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None

def _strassen_decomposition(n, m, p):
    """
    Use Strassen-style block decomposition to get a lower-rank factorization.
    
    Strategy: Use alternating minimization with structured initialization.
    Try ranks from 83 down to 50, using multiple random seeds.
    """
    n, m, p = 3, 4, 7
    best_rank = n * m * p  # 84
    
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, 49, -1):
        for seed in range(3):
            result = _alternating_minimization(n, m, p, target_rank, 
                                                max_iters=300, tol=1e-12, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 84
    
    # Try Strassen-style decomposition
    if time.time() - t0 < DEADLINE - 10:
        strassen_result = _strassen_decomposition(n, m, p)
        if strassen_result is not None:
            rank = len(strassen_result["U"][0])
            if rank < best_rank:
                best = strassen_result
                best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
