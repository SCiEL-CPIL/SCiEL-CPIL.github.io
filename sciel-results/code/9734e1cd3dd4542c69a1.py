"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = 145  # Leave 5s margin before 150s kill

    # The tensor T has shape (n*m, m*p, p*n)
    # Indices: a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1 if valid (i,j,k), else 0.
    
    # Full tensor construction
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0

    # Trivial rank-56 decomposition
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1
                V_trivial[b, col] = 1
                W_trivial[c, col] = 1
                col += 1

    # Try to reduce the rank by finding linear dependencies
    # We'll use a randomized approach: try to find a rank-r decomposition for r < 56
    # by solving a system of linear equations.
    
    def try_rank_reduction(r):
        """
        Attempt to find a rank-r decomposition by solving for U, V, W.
        Uses a structured initialization based on the trivial decomposition.
        """
        if time.time() > deadline:
            return None
            
        # We'll try to find a decomposition where the V factors are shared
        # between the two blocks (i=0 and i=1).
        # This means V[:, r] for r in 0..r-1 should be the same for both blocks.
        # We'll initialize V with the shared structure and solve for U and W.
        
        # Initialize V with random values, but ensure it has the right structure
        np.random.seed(42)
        V = np.random.randn(m*p, r) * 0.1
        
        # Solve for U and W given V
        # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # For fixed a: T[a, :, :] = sum_r U[a, r] * (V[:, r] \otimes W[:, r])
        # This is a linear system in U and W.
        
        # We'll use an iterative approach: alternate between solving for U and W.
        U = np.random.randn(n*m, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(50):
            if time.time() > deadline:
                return None
            
            # Solve for U given V and W
            # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # For fixed a: T[a, :, :] = sum_r U[a, r] * (V[:, r] \otimes W[:, r])
            M_U = np.zeros((r, m*p * p*n))
            for r_idx in range(r):
                M_U[r_idx, :] = np.kron(V[:, r_idx], W[:, r_idx])
            
            T_U = T_full.reshape(n*m, -1)
            try:
                U = T_U @ np.linalg.pinv(M_U.T)
            except:
                return None
            
            # Solve for W given U and V
            M_W = np.zeros((r, n*m * m*p))
            for r_idx in range(r):
                M_W[r_idx, :] = np.kron(U[:, r_idx], V[:, r_idx])
            
            T_W = T_full.transpose(2, 0, 1).reshape(p*n, -1)
            try:
                W = T_W @ np.linalg.pinv(M_W.T)
            except:
                return None
            
            # Calculate error
            T_approx = np.zeros_like(T_full)
            for r_idx in range(r):
                T_approx += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            err = np.linalg.norm(T_full - T_approx) / np.linalg.norm(T_full)
            if err < 1e-10:
                return U, V, W, err
        
        return None

    # Try ranks from 50 down to 28
    for r in range(50, 27, -1):
        if time.time() > deadline:
            break
        
        res = try_rank_reduction(r)
        if res is not None:
            U, V, W, err = res
            
            # Round to half-integers or integers if close
            def clean_matrix(M):
                M_rounded = np.round(M * 2) / 2
                if np.allclose(M, M_rounded, atol=1e-6):
                    return M_rounded
                M_int = np.round(M)
                if np.allclose(M, M_int, atol=1e-6):
                    return M_int
                return M
            
            U = clean_matrix(U)
            V = clean_matrix(V)
            W = clean_matrix(W)
            
            # Verify
            T_approx = np.zeros_like(T_full)
            for r_idx in range(r):
                T_approx += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            if np.allclose(T_full, T_approx, atol=1e-6):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback: Schoolbook rank 56
    return {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
