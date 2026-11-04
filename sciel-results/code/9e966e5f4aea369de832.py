import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n * m, rank) or V_arr.shape != (m * p, rank) or W_arr.shape != (p * n, rank):
        return False
        
    # Check all (i,j,k) entries equal 1
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute tensor
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if np.allclose(T, 1.0, atol=tol):
        return True
    return False

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _local_search(n, m, p, rank, seed, deadline):
    """
    Local search using block coordinate descent with decoupled analytic updates.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize U, V, W with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Target tensor T is all ones
    T = np.ones((n, m, p))
    
    # Precompute index mappings
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Function to update U given V, W
    def update_U(V, W):
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build A of shape (p, rank)
                # A[k, r] = V[j, k, r] * W[k, i, r]
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ u = 1
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U_new[i*m+j, :] = u
        return U_new

    # Function to update V given U, W
    def update_V(U, W):
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # Build B of shape (n, rank)
                # B[i, r] = U[i, j, r] * W[k, i, r]
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                # Solve B @ v = 1
                v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                V_new[j*p+k, :] = v
        return V_new

    # Function to update W given U, V
    def update_W(U, V):
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # Build C of shape (m, rank)
                # C[j, r] = U[i, j, r] * V[j, k, r]
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                # Solve C @ w = 1
                w, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                W_new[k*n+i, :] = w
        return W_new

    # Iterate
    for _ in range(100): # Max iterations
        if time.time() > deadline - 10:
            break
        
        U = update_U(V, W)
        V = update_V(U, W)
        W = update_W(U, V)
        
        # Check convergence
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.linalg.norm(T_calc - T)
        
        if err < 1e-6:
            break
    
    # Round to half-integers
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    # Try rounding to integers
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Try to find a better rank
    # Lower bound: max(n*m, m*p, p*n) = 24
    # We know some matrix multiplication can be done with less than schoolbook.
    # Let's try ranks from best_rank - 1 down to 25.
    
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for r in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            
            result = _local_search(n, m, p, r, seed, deadline)
            if result is not None:
                U_new, V_new, W_new = result
                if _verify(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = r
                    found = True
                    break
        
        if found:
            # Found a better rank, can we go lower?
            # Yes, continue the loop to try r-1
            pass
    
    return best_result
