import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    deadline = time.time() + 115  # Safer margin (evaluator kills at 150s)
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    def is_valid_entries(mat, tol=1e-6):
        """Check if all entries are integer or half-integer."""
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        return np.all(np.abs(scaled - rounded) < tol)
    
    def snap_to_half_integer(mat):
        return np.round(2.0 * mat) / 2.0
    
    def try_solve_for_U(V, W, T, r):
        """Given V and W, solve for U exactly."""
        M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
        if np.linalg.matrix_rank(M) < r:
            return None
        
        U = np.zeros((nm, r))
        for a in range(nm):
            target = T[a, :, :].flatten()
            try:
                sol, residuals, rank_m, sv = np.linalg.lstsq(M, target, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
                U[a, :] = sol
            except:
                return None
        return U
    
    def try_find_decomposition(r, max_trials=300):
        """Try to find a rank-r decomposition."""
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(10000 + r * 1000 + trial)
            
            # Strategy 1: Sparse random with varying density
            for density in [0.3, 0.5, 0.7]:
                V = (np.random.rand(mp, r) < density).astype(int) * np.random.choice([-1, 1], size=(mp, r))
                W = (np.random.rand(pn, r) < density).astype(int) * np.random.choice([-1, 1], size=(pn, r))
                
                U = try_solve_for_U(V, W, T, r)
                if U is not None:
                    U_snapped = snap_to_half_integer(U)
                    if is_valid_entries(U_snapped) and verify_solution(U_snapped, V, W, T):
                        return (U_snapped, V, W)
            
            # Strategy 2: Dense random with {-1, 1}
            V = np.random.choice([-1, 1], size=(mp, r))
            W = np.random.choice([-1, 1], size=(pn, r))
            
            U = try_solve_for_U(V, W, T, r)
            if U is not None:
                U_snapped = snap_to_half_integer(U)
                if is_valid_entries(U_snapped) and verify_solution(U_snapped, V, W, T):
                    return (U_snapped, V, W)
        
        return None
    
    # Try ranks from lower bound to 45
    # Lower bound: max(nm, mp, pn) = 27
    for r in range(27, min(rank_schoolbook, 45) + 1):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        
        result = try_find_decomposition(r, max_trials=200)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
