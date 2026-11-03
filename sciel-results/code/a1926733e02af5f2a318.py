import numpy as np
import time
import random

def _construct_small_tensor():
    """
    Construct the tensor for 3x3x2 matrix multiplication.
    U index: i*3 + k
    V index: k*2 + j
    W index: j*3 + i
    """
    n, m, p = 3, 3, 2
    U_dim = n * m # 9
    V_dim = m * p # 6
    W_dim = p * n # 6
    T = np.zeros((U_dim, V_dim, W_dim))
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _als_decompose(T, rank, max_iters=1000, tol=1e-12):
    """
    Alternating Least Squares to find a decomposition of T with given rank.
    Returns U, V, W or None if failed to converge to low residual.
    """
    u_dim, v_dim, w_dim = T.shape
    
    # Initialize with random small values
    U = np.random.randn(u_dim, rank) * 0.1
    V = np.random.randn(v_dim, rank) * 0.1
    W = np.random.randn(w_dim, rank) * 0.1
    
    # Reshape T for easier matrix operations
    # T_uvw = sum_l U_ul V_vl W_wl
    
    for it in range(max_iters):
        # Update U
        # For each l, U[:, l] is the solution to:
        # min || T - sum_l U_l (V_l outer W_l) ||
        # This is a linear system for each l? No, it's coupled.
        # We can solve for all U at once by flattening V and W.
        # Let M_l be the outer product of V[:, l] and W[:, l].
        # T = sum_l U[:, l] (M_l).T ? No.
        # T[u, v, w] = sum_l U[u, l] V[v, l] W[w, l]
        # For fixed V, W, we want to find U.
        # This is equivalent to:
        # For each u, T[u, :, :] = sum_l U[u, l] (V[:, l] outer W[:, l])
        # Let's form a matrix A of shape (u_dim, v_dim * w_dim)
        # and B of shape (rank, v_dim * w_dim)
        
        A = T.reshape(u_dim, v_dim * w_dim)
        B = np.zeros((rank, v_dim * w_dim))
        for l in range(rank):
            B[l, :] = (V[:, l] * W[:, l]).flatten() # Outer product flattened
            
        # Solve A = U @ B
        # U = A @ B^+
        U_new, _, _, _ = np.linalg.lstsq(B.T, A.T, rcond=None)
        U = U_new.T
        
        # Update V
        A = T.reshape(v_dim, u_dim * w_dim)
        B = np.zeros((rank, u_dim * w_dim))
        for l in range(rank):
            B[l, :] = (U[:, l] * W[:, l]).flatten()
            
        V_new, _, _, _ = np.linalg.lstsq(B.T, A.T, rcond=None)
        V = V_new.T
        
        # Update W
        A = T.reshape(w_dim, u_dim * v_dim)
        B = np.zeros((rank, u_dim * v_dim))
        for l in range(rank):
            B[l, :] = (U[:, l] * V[:, l]).flatten()
            
        W_new, _, _, _ = np.linalg.lstsq(B.T, A.T, rcond=None)
        W = W_new.T
        
        # Check convergence
        # Calculate residual
        T_recon = np.einsum('ul,vl,wl->uvw', U, V, W)
        residual = np.linalg.norm(T - T_recon)
        if residual < tol:
            return U, V, W
            
    # Check final residual
    T_recon = np.einsum('ul,vl,wl->uvw', U, V, W)
    residual = np.linalg.norm(T - T_recon)
    if residual < 1e-6: # Allow some tolerance for rounding
        return U, V, W
    return None

def _round_and_verify(U, V, W, T):
    """
    Round U, V, W to half-integers and verify if they exactly reconstruct T.
    """
    # Try rounding to nearest 0.5
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify
    T_recon = np.einsum('ul,vl,wl->uvw', U_rounded, V_rounded, W_rounded)
    if np.allclose(T_recon, T, atol=1e-9):
        return U_rounded, V_rounded, W_rounded
    
    # Try rounding to nearest integer
    U_rounded = np.round(U)
    V_rounded = np.round(V)
    W_rounded = np.round(W)
    T_recon = np.einsum('ul,vl,wl->uvw', U_rounded, V_rounded, W_rounded)
    if np.allclose(T_recon, T, atol=1e-9):
        return U_rounded, V_rounded, W_rounded
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 50s buffer
    
    # Build small tensor
    T_small = _construct_small_tensor()
    
    best_rank = 18 # Schoolbook
    best_U_small = None
    best_V_small = None
    best_W_small = None
    
    # Try to find a decomposition with rank < 18
    # We try ranks 15, 16, 17
    for rank in [15, 16, 17]:
        if time.time() > deadline:
            break
            
        # Multiple restarts for each rank
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Random seed for reproducibility in debug, but random for search
            np.random.seed(restart * 1000 + rank)
            
            U, V, W = _als_decompose(T_small, rank, max_iters=200)
            
            if U is not None:
                # Check if it's already close to integer/half-integer
                # If not, try to round
                result = _round_and_verify(U, V, W, T_small)
                if result is not None:
                    U_r, V_r, W_r = result
                    # Verify exactness one more time with strict tolerance
                    T_recon = np.einsum('ul,vl,wl->uvw', U_r, V_r, W_r)
                    if np.allclose(T_recon, T_small, atol=1e-9):
                        best_rank = rank
                        best_U_small = U_r
                        best_V_small = V_r
                        best_W_small = W_r
                        break # Found a valid decomposition for this rank, stop searching this rank
        
        if best_rank < 18:
            break # Found a better rank, stop searching lower ranks (we go 15->16->17, so if 15 works, we stop)
            # Note: The loop is 15, 16, 17. If 15 works, we break. If 16 works, we break.

    # Construct full decomposition
    n, m, p = 3, 3, 8
    full_rank = best_rank * 4
    
    U_full = np.zeros((n * m, full_rank))
    V_full = np.zeros((m * p, full_rank))
    W_full = np.zeros((p * n, full_rank))
    
    if best_rank < 18 and best_U_small is not None:
        # Expand small decomposition to full
        for q in range(4):
            # U is the same for all blocks
            U_full[:, q * best_rank : (q + 1) * best_rank] = best_U_small
            
            # V expansion
            # V_small index: k * 2 + j_local
            # V_full index: k * 8 + (2 * q + j_local)
            for k in range(m):
                for j_local in range(2):
                    v_small_idx = k * 2 + j_local
                    v_full_idx = k * 8 + (2 * q + j_local)
                    V_full[v_full_idx, q * best_rank : (q + 1) * best_rank] = best_V_small[v_small_idx, :]
            
            # W expansion
            # W_small index: j_local * 3 + i
            # W_full index: (2 * q + j_local) * 3 + i
            for j_local in range(2):
                for i in range(n):
                    w_small_idx = j_local * 3 + i
                    w_full_idx = (2 * q + j_local) * 3 + i
                    W_full[w_full_idx, q * best_rank : (q + 1) * best_rank] = best_W_small[w_small_idx, :]
    else:
        # Fallback to schoolbook
        full_rank = n * m * p
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    u_idx = i * m + k
                    v_idx = k * p + j
                    w_idx = j * n + i
                    U_full[u_idx, col] = 1
                    V_full[v_idx, col] = 1
                    W_full[w_idx, col] = 1
                    col += 1

    # Convert to lists of integers if possible, or keep as numpy arrays?
    # The prompt says "integer/half-integer entries".
    # The return interface expects {'U', 'V', 'W'}.
    # Usually numpy arrays are accepted.
    
    return {"U": U_full, "V": V_full, "W": W_full}
