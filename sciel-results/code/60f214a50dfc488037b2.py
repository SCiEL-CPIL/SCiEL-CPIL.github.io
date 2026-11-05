import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 12
    P = p * n  # 12
    
    # Build target tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Pre-compute matricizations
    T_c = T.reshape(N * M, P)          # (108, 12)
    T_a = T.reshape(N, M * P)          # (9, 144)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)  # (12, 108)
    
    start_time = time.time()
    deadline = 130.0  # Safety margin
    
    def check_half_integer(U, V, W, tol=1e-9):
        """Check if rounding to half-integers gives exact decomposition."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        return False, None, None
    
    def als_attempt(U, V, W, max_iters=200, tol=1e-10):
        """Run ALS to convergence."""
        for it in range(max_iters):
            if time.time() > deadline:
                return U, V, W, False
            
            # Update W
            E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, -1)
            try:
                W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                W = W_T.T
            except:
                return U, V, W, False
            
            # Update U
            E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, -1)
            try:
                U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a.T, rcond=None)
                U = U_T.T
            except:
                return U, V, W, False
            
            # Update V
            E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, -1)
            try:
                V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b.T, rcond=None)
                V = V_T.T
            except:
                return U, V, W, False
            
            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_calc - T)
            if error < tol:
                return U, V, W, True
        
        return U, V, W, False
    
    def local_search_half_int(U, V, W, T, max_iters=200, tol=1e-9):
        """Efficient local search on the half-integer lattice using residual updates."""
        N, M, P = T.shape
        R = U.shape[1]
        
        # Project to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Compute initial T_calc and Residual
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        Res = T - T_calc
        curr_err_sq = np.sum(Res * Res)
        best_err_sq = curr_err_sq
        
        if curr_err_sq < tol * tol:
            return U, V, W
            
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_Res = Res.copy()
        
        # Precompute B matrices for current U, V, W
        # B_U[r, b, c] = V[b, r] * W[c, r]
        # B_V[r, a, c] = U[a, r] * W[c, r]
        # B_W[r, a, b] = U[a, r] * V[b, r]
        B_U = np.einsum('br,cr->rbc', V, W)
        B_V = np.einsum('ar,cr->rac', U, W)
        B_W = np.einsum('ar,br->rab', U, V)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            
            # Try flipping a random entry in U, V, or W
            choice = np.random.rand()
            
            if choice < 0.33:
                a, r = np.random.randint(N), np.random.randint(R)
                for delta in [-0.5, 0.5]:
                    # Calculate new error squared without full tensor update
                    diff = Res[a, :, :] - delta * B_U[r]
                    new_err_sq = np.sum(diff * diff)
                    if new_err_sq < best_err_sq - 1e-12:
                        # Accept
                        U[a, r] += delta
                        Res[a, :, :] -= delta * B_U[r]
                        best_err_sq = new_err_sq
                        best_U[a, r] = U[a, r]
                        best_V, best_W = V, W
                        best_Res = Res.copy()
                        improved = True
                        # Update B_V and B_W slices
                        B_V[r, a, :] = U[a, r] * W[:, r]
                        B_W[r, a, :] = U[a, r] * V[:, r]
                        break
                        
            elif choice < 0.66:
                b, r = np.random.randint(M), np.random.randint(R)
                for delta in [-0.5, 0.5]:
                    diff = Res[:, b, :] - delta * B_V[r, :, b]
                    new_err_sq = np.sum(diff * diff)
                    if new_err_sq < best_err_sq - 1e-12:
                        # Accept
                        V[b, r] += delta
                        Res[:, b, :] -= delta * B_V[r, :, b]
                        best_err_sq = new_err_sq
                        best_V[b, r] = V[b, r]
                        best_U, best_W = U, W
                        best_Res = Res.copy()
                        improved = True
                        # Update B_U and B_W slices
                        B_U[r, b, :] = V[b, r] * W[:, r]
                        B_W[r, :, b] = U[:, r] * V[b, r]
                        break
                        
            else:
                c, r = np.random.randint(P), np.random.randint(R)
                for delta in [-0.5, 0.5]:
                    diff = Res[:, :, c] - delta * B_W[r, :, c]
                    new_err_sq = np.sum(diff * diff)
                    if new_err_sq < best_err_sq - 1e-12:
                        # Accept
                        W[c, r] += delta
                        Res[:, :, c] -= delta * B_W[r, :, c]
                        best_err_sq = new_err_sq
                        best_W[c, r] = W[c, r]
                        best_U, best_V = U, V
                        best_Res = Res.copy()
                        improved = True
                        # Update B_U and B_V slices
                        B_U[r, :, c] = V[:, r] * W[c, r]
                        B_V[r, :, c] = U[:, r] * W[c, r]
                        break
                        
            if not improved:
                break
                
            if best_err_sq < tol * tol:
                return best_U, best_V, best_W
                
        if best_err_sq < tol * tol:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from low to high
    best_result = None
    
    for R in range(23, 30):
        if time.time() > deadline:
            break
        
        # Multiple random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            seed = 12345 + R * 1000 + attempt
            np.random.seed(seed)
            
            # Random initialization
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            # Run ALS
            U, V, W, converged = als_attempt(U, V, W)
            
            if converged:
                # Try to get half-integer solution
                polished = local_search_half_int(U, V, W, T)
                if polished is not None:
                    U_r, V_r, W_r = polished
                    # Final verification
                    T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(T_calc, T, atol=1e-9):
                        best_result = (R, U_r, V_r, W_r)
                        break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        R, U_r, V_r, W_r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback: schoolbook decomposition (rank = n*m*p = 36)
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
