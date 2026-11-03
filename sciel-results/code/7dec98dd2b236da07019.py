import time
import numpy as np

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140

    # Build target tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback: rank 36
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=float)
    V_fb = np.zeros((m*p, rank_fb), dtype=float)
    W_fb = np.zeros((p*n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fb

    def snap_to_half(X):
        return np.round(X * 2) / 2.0

    def check_decomposition(U, V, W, rank, tol=1e-9):
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        return np.allclose(T_decomp, T, atol=tol)

    def als_decompose(rank, U_init, V_init, W_init, max_iter=200, tol=1e-12):
        """Alternating least squares for tensor decomposition."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each a, solve V^T W^T vec(T[a,:,:]) = U[a,:]
            # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # Fix V,W: for each a, U[a,:] = (V*W)^+ T[a,:,:].reshape(-1)
            # where V*W is outer product matrix of size (m*p*p*n) x rank
            
            # More efficiently: U[a,t] = sum_{b,c} T[a,b,c] V[b,t] W[c,t] / (sum_{b,c} V[b,t]^2 W[c,t]^2)
            # But this is for orthogonal factors. For general ALS:
            
            # U[a,:] solves: (V ⊙ W)^T U[a,:]^T = T[a,:,:].reshape(-1)
            # where ⊙ is Khatri-Rao product
            
            # Precompute Khatri-Rao of V and W
            # (V ⊙ W)[(b,c), t] = V[b,t] * W[c,t]
            VW = np.einsum('bt,ct->bct', V, W)  # shape (m*p, p*n, rank)
            VW_flat = VW.reshape(m*p*p*n, rank)  # (m*p*p*n, rank)
            
            # For each a, solve least squares
            U_new = np.zeros((n*m, rank))
            for a in range(n*m):
                Ta = T[a, :, :].reshape(-1)
                # Solve VW_flat^T U[a,:]^T = Ta
                # U[a,:]^T = (VW_flat^T VW_flat)^{-1} VW_flat^T Ta
                A = VW_flat.T @ Ta  # (rank,)
                B = VW_flat.T @ VW_flat  # (rank, rank)
                try:
                    U_new[a, :] = np.linalg.solve(B + 1e-10 * np.eye(rank), A)
                except:
                    U_new[a, :] = np.linalg.lstsq(B + 1e-10 * np.eye(rank), A, rcond=None)[0]
            U = U_new
            
            # Update V
            UW = np.einsum('at,ct->act', U, W)  # (n*m, p*n, rank)
            UW_flat = UW.reshape(n*m*p*n, rank)
            
            V_new = np.zeros((m*p, rank))
            for b in range(m*p):
                Tb = T[:, b, :].reshape(-1)
                A = UW_flat.T @ Tb
                B = UW_flat.T @ UW_flat
                try:
                    V_new[b, :] = np.linalg.solve(B + 1e-10 * np.eye(rank), A)
                except:
                    V_new[b, :] = np.linalg.lstsq(B + 1e-10 * np.eye(rank), A, rcond=None)[0]
            V = V_new
            
            # Update W
            UV = np.einsum('at,bt->abt', U, V)  # (n*m, m*p, rank)
            UV_flat = UV.reshape(n*m*m*p, rank)
            
            W_new = np.zeros((p*n, rank))
            for c in range(p*n):
                Tc = T[:, :, c].reshape(-1)
                A = UV_flat.T @ Tc
                B = UV_flat.T @ UV_flat
                try:
                    W_new[c, :] = np.linalg.solve(B + 1e-10 * np.eye(rank), A)
                except:
                    W_new[c, :] = np.linalg.lstsq(B + 1e-10 * np.eye(rank), A, rcond=None)[0]
            W = W_new
            
            # Check convergence
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            err = np.linalg.norm(T - T_decomp)
            if err < tol:
                break
        
        return U, V, W, err

    def try_rank(rank, num_restarts=10):
        """Try to find a decomposition of given rank."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return
            
            # Initialize with structured random values
            # Use values from {-1, -0.5, 0, 0.5, 1}
            scale = 0.5
            U_init = scale * np.random.choice([-1, 0, 1], size=(n*m, rank))
            V_init = scale * np.random.choice([-1, 0, 1], size=(m*p, rank))
            W_init = scale * np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            # Run ALS
            U, V, W, err = als_decompose(rank, U_init, V_init, W_init, max_iter=100)
            
            # Snap to half-integers and check
            U_s = snap_to_half(U)
            V_s = snap_to_half(V)
            W_s = snap_to_half(W)
            
            if check_decomposition(U_s, V_s, W_s, rank):
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = rank
                return True
            
            # Also try without snapping (in case exact solution isn't half-integer)
            if err < 1e-10:
                # Check if close to half-integers
                U_close = np.allclose(U, snap_to_half(U), atol=1e-6)
                V_close = np.allclose(V, snap_to_half(V), atol=1e-6)
                W_close = np.allclose(W, snap_to_half(W), atol=1e-6)
                if U_close and V_close and W_close:
                    best_U, best_V, best_W = snap_to_half(U), snap_to_half(V), snap_to_half(W)
                    best_rank = rank
                    return True
        
        return False

    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # But for matrix multiplication, known bounds suggest higher
    # Try from 12 up, but prioritize likely values
    
    # For 3x3x4, known results suggest rank around 24-28
    # Let's try a range
    for rank in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
        if time.time() > deadline - 15:
            break
        if rank >= best_rank:
            continue
        
        # More restarts for lower ranks (harder to find)
        num_restarts = 20 if rank <= 20 else 15
        if try_rank(rank, num_restarts):
            # Found a solution at this rank, try to go lower
            # But since we're going up, we should have found lower first
            # If we found rank R, we can stop since we tried all lower
            break
    
    # Final snap
    U_out = snap_to_half(best_U)
    V_out = snap_to_half(best_V)
    W_out = snap_to_half(best_W)
    
    # Verify
    if not check_decomposition(U_out, V_out, W_out, best_rank):
        # Fall back to schoolbook
        U_out = U_fb
        V_out = V_fb
        W_out = W_fb
        best_rank = rank_fb
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
