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
    deadline = 140.0  # Leave 10s margin before 150s limit
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        T_rec = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec, T, atol=1e-9)
    
    def try_lattice_points(U, V, W, rank):
        """Try rounding to various nearby lattice points (integers and half-integers)"""
        # Try standard rounding
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        if check_exact(U_s, V_s, W_s, rank):
            return U_s, V_s, W_s
        
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_exact(U_i, V_i, W_i, rank):
            return U_i, V_i, W_i
        
        # Try floor and ceil variants
        for fu in [np.floor, np.ceil]:
            for fv in [np.floor, np.ceil]:
                for fw in [np.floor, np.ceil]:
                    U_t = fu(U * 2) / 2.0
                    V_t = fv(V * 2) / 2.0
                    W_t = fw(W * 2) / 2.0
                    if check_exact(U_t, V_t, W_t, rank):
                        return U_t, V_t, W_t
        return None

    def build_schoolbook(rank=None):
        """Build the schoolbook decomposition"""
        if rank is None:
            rank = rank_schoolbook
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if column < rank:
                        U[i * m + j, column] = 1
                        V[j * p + k, column] = 1
                        W[k * n + i, column] = 1
                    column += 1
        
        return U, V, W, rank

    def als_decompose(rank, max_iters=200, seed=0, init=None):
        """Alternating Least Squares for 3rd order tensor decomposition"""
        np.random.seed(seed)
        
        if init is not None:
            U, V, W = init
        else:
            # Structured initialization: start from schoolbook-like structure
            U = np.random.randn(n, m, rank) * 0.1
            V = np.random.randn(m, p, rank) * 0.1
            W = np.random.randn(p, n, rank) * 0.1
            
            # Add some structured components
            for t in range(min(rank, 10)):
                i = t % n
                j = (t // n) % m
                k = (t // (n*m)) % p
                U[i, j, t] = 1.0
                V[j, k, t] = 1.0
                W[k, i, t] = 1.0
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute current error
            T_rec = np.einsum('ijt,jkt,kit->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if abs(prev_err - current_err) < 1e-14:
                break
            prev_err = current_err
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    M = V[j, :, :] * W[:, i, :]  # shape (p, rank)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    M = U[:, j, :] * W[k, :, :]  # shape (n, rank)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(M.T, T[:, j, k], rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    M = U[i, :, :] * V[:, k, :]  # shape (m, rank)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(M.T, T[i, :, k], rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
        
        # Final check
        T_rec = np.einsum('ijt,jkt,kit->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-4:
            snapped = try_lattice_points(U.reshape(NM, rank), V.reshape(MP, rank), W.reshape(PN, rank), rank)
            if snapped:
                return snapped
            
            # Try local polish with scipy
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
                    
                    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-16})
                    
                    if res.fun < 1e-8:
                        x_opt = res.x
                        U_opt = x_opt[:n*m*rank].reshape(n, m, rank)
                        V_opt = x_opt[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_opt = x_opt[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        
                        snapped = try_lattice_points(U_opt.reshape(NM, rank), V_opt.reshape(MP, rank), W_opt.reshape(PN, rank), rank)
                        if snapped:
                            return snapped
                except Exception:
                    pass
        
        return None

    def direct_search(rank, num_trials=50):
        """Direct random search with structured perturbations of schoolbook"""
        base_U, base_V, base_W, _ = build_schoolbook(rank)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Create a perturbation: zero out some columns and try to fill with structured entries
            U_try = base_U.copy()
            V_try = base_V.copy()
            W_try = base_W.copy()
            
            # Randomly modify a few columns
            num_mods = min(3, rank)
            cols = np.random.choice(rank, num_mods, replace=False)
            for c in cols:
                # Set to random small values
                U_try[:, c] = np.random.choice([-1, 0, 1, 0.5, -0.5], size=NM)
                V_try[:, c] = np.random.choice([-1, 0, 1, 0.5, -0.5], size=MP)
                W_try[:, c] = np.random.choice([-1, 0, 1, 0.5, -0.5], size=PN)
            
            if check_exact(U_try, V_try, W_try, rank):
                return U_try, V_try, W_try
        
        return None

    def try_rank(rank, num_restarts=3):
        """Try to find decomposition of given rank"""
        # First try direct search (fast)
        result = direct_search(rank, num_trials=20)
        if result is not None:
            return result
        
        # Then try ALS with multiple seeds
        for seed in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            result = als_decompose(rank, max_iters=100, seed=seed)
            if result is not None:
                return result
        
        return None

    best_result = None
    
    # Try ranks from schoolbook downward
    # Start close to schoolbook for better chance of finding valid solutions
    for rank in range(rank_schoolbook - 1, 29, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(rank, num_restarts=2)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    # Fallback to schoolbook
    if best_result is None:
        U, V, W, rank = build_schoolbook()
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
