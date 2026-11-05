import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x6 matrix.
    
    The decomposition satisfies:
        A * B = sum_{t=0}^{rank-1} W[:,t] * outer(U[:,t], V[:,t])
    where:
        U is (n*m, rank) = (12, rank)
        V is (m*p, rank) = (24, rank)
        W is (p*n, rank) = (18, rank)
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    NM, MP, PN = n * m, m * p, p * n  # 12, 24, 18
    
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin before 120s limit
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        # U: (n*m, rank) -> (n, m, rank)
        # V: (m*p, rank) -> (m, p, rank)
        # W: (p*n, rank) -> (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T_rec[i, j, k] = sum_t U[i, j, t] * V[j, k, t] * W[k, i, t]
        T_rec = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec, T, atol=1e-9)
    
    def snap_to_lattice(U, V, W):
        """Snap entries to nearest integer or half-integer and check."""
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        
        if check_exact(U_s, V_s, W_s, U.shape[1]):
            return U_s, V_s, W_s
        
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        
        if check_exact(U_i, V_i, W_i, U.shape[1]):
            return U_i, V_i, W_i
            
        return None

    def als_decompose(rank, max_iters=100, seed=0):
        """Alternating Least Squares for 3rd order tensor decomposition"""
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n, m, rank) * 0.5
        V = np.random.randn(m, p, rank) * 0.5
        W = np.random.randn(p, n, rank) * 0.5
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute current error
            T_rec = np.einsum('ijt,jkt,kit->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if abs(prev_err - current_err) < 1e-12:
                break
            prev_err = current_err
            
            # Solve for U: fix V, W
            # For each i, j, U[i, j, :] solves: sum_k V[j, k, :] * W[k, i, :] * U[i, j, :] = T[i, j, :]
            # This is a linear system in U[i, j, :]
            for i in range(n):
                for j in range(m):
                    # A is (p, rank), B is (rank, p) effectively?
                    # We want sum_k V[j, k, t] * W[k, i, t] * U[i, j, t] = T[i, j, k]
                    # Let M[t, k] = sum_u? No.
                    # For fixed i, j, we have vector T[i, j, :] of length p.
                    # We have matrix M of shape (p, rank) where M[k, t] = V[j, k, t] * W[k, i, t]
                    # We want U[i, j, :] such that M.T @ U[i, j, :] = T[i, j, :]
                    M = V[j, :, :] * W[:, i, :]  # shape (p, rank)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V: fix U, W
            for j in range(m):
                for k in range(p):
                    # T[:, j, k] = sum_i U[i, j, t] * W[k, i, t] * V[j, k, t]
                    # M[i, t] = U[i, j, t] * W[k, i, t]
                    # We want V[j, k, :] such that M.T @ V[j, k, :] = T[:, j, k]
                    M = U[:, j, :] * W[k, :, :]  # shape (n, rank)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(M.T, T[:, j, k], rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W: fix U, V
            for k in range(p):
                for i in range(n):
                    # T[i, :, k] = sum_j U[i, j, t] * V[j, k, t] * W[k, i, t]
                    # M[j, t] = U[i, j, t] * V[j, k, t]
                    # We want W[k, i, :] such that M.T @ W[k, i, :] = T[i, :, k]
                    M = U[i, :, :] * V[:, k, :]  # shape (m, rank)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(M.T, T[i, :, k], rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
        
        # Final check and polish
        T_rec = np.einsum('ijt,jkt,kit->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-6:
            # Try rounding
            snapped = snap_to_lattice(U.reshape(NM, rank), V.reshape(MP, rank), W.reshape(PN, rank))
            if snapped:
                return snapped
            
            # Try L-BFGS-B polish
            if final_err < 1.0:
                try:
                    from scipy.optimize import minimize
                    
                    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    
                    def objective(x):
                        U_x = x[:n*m*rank].reshape(n, m, rank)
                        V_x = x[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_x = x[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        T_rec = np.einsum('ijt,jkt,kit->ijk', U_x, V_x, W_x)
                        return np.sum((T_rec - T)**2)
                    
                    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-15})
                    
                    if res.success:
                        x_opt = res.x
                        U_opt = x_opt[:n*m*rank].reshape(n, m, rank)
                        V_opt = x_opt[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_opt = x_opt[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        
                        snapped = snap_to_lattice(U_opt.reshape(NM, rank), V_opt.reshape(MP, rank), W_opt.reshape(PN, rank))
                        if snapped:
                            return snapped
                except Exception:
                    pass
        
        return None

    def try_rank(rank, num_restarts=10):
        """Try to find decomposition of given rank with multiple restarts"""
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            result = als_decompose(rank, max_iters=50, seed=seed)
            if result is not None:
                return result
        return None

    best_result = None
    
    # Search for lower rank
    # Start from a reasonable lower bound. 
    # Theoretical lower bound is max(n*m, m*p, p*n) = 24.
    # Let's try a few ranks.
    # To save time, we try ranks in increasing order of likelihood of existence?
    # Or just try a few small ranks.
    
    # Let's try ranks 30 to 71
    for rank in range(30, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, num_restarts=5)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    # Fallback to schoolbook
    if best_result is None:
        rank = rank_schoolbook
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
