import time
import sys
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    """
    Returns a rank-minimal (or best known) tensor decomposition for 
    multiplying a 2x3 matrix by a 3x6 matrix.
    
    Strategy:
    1. Find a low-rank decomposition for the 2x3x3 sub-problem.
    2. Use two copies of this sub-decomposition to handle the 2x3x6 problem.
    3. Fallback to schoolbook rank 36 if optimization fails.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank for 2x3x6 is 36
    sb_rank = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n*m, sb_rank), dtype=float)
    V_sb = np.zeros((m*p, sb_rank), dtype=float)
    W_sb = np.zeros((p*n, sb_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = sb_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Sub-problem: 2x3x3
    n2, m2, p2 = 2, 3, 3
    sb_rank_sub = n2 * m2 * p2 # 18
    
    # Define target tensor for 2x3x3
    T2_indices = []
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2_indices.append((i*m2+j, j*p2+k, k*n2+i))
                
    T2_a = np.array([t[0] for t in T2_indices])
    T2_b = np.array([t[1] for t in T2_indices])
    T2_c = np.array([t[2] for t in T2_indices])
    
    def make_residual_func_2x3x3(target_rank):
        u_size = n2*m2 * target_rank
        v_size = m2*p2 * target_rank
        w_size = p2*n2 * target_rank
        
        def res(x):
            U = x[:u_size].reshape(n2*m2, target_rank)
            V = x[u_size:u_size+v_size].reshape(m2*p2, target_rank)
            W = x[u_size+v_size:].reshape(p2*n2, target_rank)
            
            U_a = U[T2_a, :]
            V_b = V[T2_b, :]
            W_c = W[T2_c, :]
            
            vals = np.einsum('ij,ij,ij->i', U_a, V_b, W_c)
            return vals - 1.0
            
        return res
    
    def snap_and_validate_2x3x3(U, V, W, tol=1e-6):
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        U_a = U_snap[T2_a, :]
        V_b = V_snap[T2_b, :]
        W_c = W_snap[T2_c, :]
        vals = np.einsum('ij,ij,ij->i', U_a, V_b, W_c)
        
        if not np.allclose(vals, 1.0, atol=tol):
            return False
        
        # Check non-targets are approximately 0
        np.random.seed(42)
        for _ in range(500):
            a = np.random.randint(0, n2*m2)
            b = np.random.randint(0, m2*p2)
            c = np.random.randint(0, p2*n2)
            if (a, b, c) in T2_indices:
                continue
            val = np.dot(U_snap[a, :], V_snap[b, :] * W_snap[c, :])
            if abs(val) > tol:
                return False
        return True

    def define_full_decomposition_from_sub(U_sub, V_sub, W_sub, rank_sub):
        """
        Constructs the 2x3x6 decomposition from two 2x3x3 sub-decompositions.
        """
        target_rank = 2 * rank_sub
        
        U_full = np.zeros((n*m, target_rank), dtype=float)
        V_full = np.zeros((m*p, target_rank), dtype=float)
        W_full = np.zeros((p*n, target_rank), dtype=float)
        
        # First half: columns 0-2 of B (k=0,1,2)
        for r in range(rank_sub):
            U_full[:, r] = U_sub[:, r]
            # V: for j in 0..2, k in 0..2 -> V[j*6+k, r]
            for j in range(m2):
                for k in range(p2):
                    V_full[j*p + k, r] = V_sub[j*p2 + k, r]
            # W: for k in 0..2, i in 0..1 -> W[k*2+i, r]
            for k in range(p2):
                for i in range(n2):
                    W_full[k*n + i, r] = W_sub[k*n2 + i, r]
        
        # Second half: columns 3-5 of B (k=3,4,5)
        for r in range(rank_sub):
            r2 = rank_sub + r
            U_full[:, r2] = U_sub[:, r]
            # V: for j in 0..2, k in 3..5 -> V[j*6+(k-3), r2] = V[j*6+k, r2]
            for j in range(m2):
                for k in range(p2):
                    V_full[j*p + (k + p2), r2] = V_sub[j*p2 + k, r]
            # W: for k in 3..5, i in 0..1 -> W[k*2+i, r2]
            for k in range(p2):
                for i in range(n2):
                    W_full[(k + p2)*n + i, r2] = W_sub[k*n2 + i, r]
        
        return U_full, V_full, W_full

    def snap_and_validate_full(U, V, W, tol=1e-6):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check target tensor entries
        T_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_indices.append((i*m+j, j*p+k, k*n+i))
        
        T_a = np.array([t[0] for t in T_indices])
        T_b = np.array([t[1] for t in T_indices])
        T_c = np.array([t[2] for t in T_indices])
        
        U_a = U_snap[T_a, :]
        V_b = V_snap[T_b, :]
        W_c = W_snap[T_c, :]
        vals = np.einsum('ij,ij,ij->i', U_a, V_b, W_c)
        
        if not np.allclose(vals, 1.0, atol=tol):
            return False
        
        # Check non-targets
        np.random.seed(42)
        for _ in range(500):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            if (a, b, c) in T_indices:
                continue
            val = np.dot(U_snap[a, :], V_snap[b, :] * W_snap[c, :])
            if abs(val) > tol:
                return False
        return True

    np.random.seed(42)
    
    # Try to find low-rank decomposition for 2x3x3
    # Known rank is likely 12. Try 12, then 13, etc.
    found_sub = False
    U_sub_best = None
    V_sub_best = None
    W_sub_best = None
    rank_sub_best = sb_rank_sub
    
    for r_sub in range(12, sb_rank_sub + 1):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        res_func_sub = make_residual_func_2x3x3(r_sub)
        u_size_sub = n2*m2 * r_sub
        v_size_sub = m2*p2 * r_sub
        w_size_sub = p2*n2 * r_sub
        
        # Try multiple restarts
        for restart in range(10):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            if restart == 0:
                # Initialize with schoolbook truncated
                U_init = np.zeros((n2*m2, r_sub))
                V_init = np.zeros((m2*p2, r_sub))
                W_init = np.zeros((p2*n2, r_sub))
                col = 0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            if col >= r_sub:
                                break
                            U_init[i*m2+j, col] = 1.0
                            V_init[j*p2+k, col] = 1.0
                            W_init[k*n2+i, col] = 1.0
                            col += 1
                        if col >= r_sub:
                            break
                    if col >= r_sub:
                        break
                if col >= r_sub:
                    break
            else:
                U_init = np.random.randn(n2*m2, r_sub) * 0.5
                V_init = np.random.randn(m2*p2, r_sub) * 0.5
                W_init = np.random.randn(p2*n2, r_sub) * 0.5
            
            x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
            
            try:
                result = least_squares(res_func_sub, x0, method='lm', max_nfev=5000)
                
                U_opt = result.x[:u_size_sub].reshape(n2*m2, r_sub)
                V_opt = result.x[u_size_sub:u_size_sub+v_size_sub].reshape(m2*p2, r_sub)
                W_opt = result.x[u_size_sub+v_size_sub:].reshape(p2*n2, r_sub)
                
                if snap_and_validate_2x3x3(U_opt, V_opt, W_opt):
                    U_sub_best = U_opt
                    V_sub_best = V_opt
                    W_sub_best = W_opt
                    rank_sub_best = r_sub
                    found_sub = True
                    break # Break restart loop, try next r_sub (lower is better, but we start at 12)
            except Exception:
                continue
        
        if found_sub:
            break # Found minimal rank for sub-problem, stop searching higher ranks
    
    if found_sub and time.time() - start_time < DEADLINE - 5:
        # Combine two rank-r_sub decompositions for 2x3x6
        U_full, V_full, W_full = define_full_decomposition_from_sub(U_sub_best, V_sub_best, W_sub_best, rank_sub_best)
        
        if snap_and_validate_full(U_full, V_full, W_full):
            best_rank = 2 * rank_sub_best
            best_U = U_full
            best_V = V_full
            best_W = W_full
        else:
            # If full validation fails, try to polish
            target_rank = 2 * rank_sub_best
            # We might need to re-optimize the full system
            # But given the structure, if sub is valid, full should be valid.
            # If not, we stick with schoolbook.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
