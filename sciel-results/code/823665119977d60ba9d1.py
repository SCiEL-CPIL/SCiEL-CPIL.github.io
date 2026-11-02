import time
import numpy as np

DIMENSIONS = [3, 4, 4]
_START = time.time()
_TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    r0 = n * m * p  # 48 schoolbook

    # Schoolbook decomposition as fallback
    U0 = np.zeros((nm, r0))
    V0 = np.zeros((mp, r0))
    W0 = np.zeros((pn, r0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    best_rank, best_U, best_V, best_W = r0, U0, V0, W0

    # Strategy 1: M_3,4,4 = M_3,3,2 + M_3,3,2 + M_3,1,4
    # Find best rank for M_3,3,2
    best_332 = _find_best_rank_332()
    
    # Rank for M_3,1,4 is 12 (Schoolbook)
    r_314 = 12
    
    total_rank = 2 * best_332 + r_314
    
    if total_rank < best_rank:
        # Assemble the decomposition
        U_new, V_new, W_new = _assemble_decomposition(best_332, r_314)
        if U_new is not None:
            best_rank = total_rank
            best_U, best_V, best_W = U_new, V_new, W_new

    # Strategy 2: Try direct optimization for lower ranks
    for target_rank in [40, 42, 44, 46]:
        if time.time() - _START > _TIME_LIMIT - 15:
            break
        if target_rank >= best_rank:
            continue
        res = _try_direct_decomposition(n, m, p, target_rank)
        if res is not None:
            U_new, V_new, W_new = res
            if _verify_full(U_new, V_new, W_new, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _find_best_rank_332():
    # Find minimal rank for 3x3x2
    # Known: rank 15 is achievable
    for r in [15, 16, 17, 18]:
        if time.time() - _START > _TIME_LIMIT - 20:
            break
        res = _try_rank_332(r)
        if res is not None:
            return r
    return 18  # Fallback to schoolbook

def _try_rank_332(r, n_trials=20):
    n, m, p = 3, 3, 2
    nm, mp, pn = n * m, m * p, p * n
    idx = [(i * m + j, j * p + k, k * n + i)
           for i in range(n) for j in range(m) for k in range(p)]
    
    best_loss = float('inf')
    best_solution = None
    
    for t in range(n_trials):
        if time.time() - _START > _TIME_LIMIT - 20:
            break
        
        rng = np.random.RandomState(t * 7919 + r * 1000)
        
        # Initialize with small values to help convergence
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5
        
        # Try multiple learning rate schedules
        for lr_init in [0.1, 0.05, 0.02]:
            U_copy = U.copy()
            V_copy = V.copy()
            W_copy = W.copy()
            
            for it in range(2000):
                if time.time() - _START > _TIME_LIMIT - 20:
                    break
                
                # Compute gradients
                Ug = np.zeros_like(U_copy)
                Vg = np.zeros_like(V_copy)
                Wg = np.zeros_like(W_copy)
                loss = 0.0
                
                for ui, vi, wi in idx:
                    vw = V_copy[vi] * W_copy[wi]
                    uv = U_copy[ui] * V_copy[vi]
                    uw = U_copy[ui] * W_copy[wi]
                    s = np.dot(U_copy[ui], vw)
                    e = s - 1.0
                    loss += e * e
                    Ug[ui] += e * vw
                    Vg[vi] += e * uw
                    Wg[wi] += e * uv
                
                if loss < 1e-12:
                    # Found a solution, check for half-integers
                    U_r = np.round(U_copy * 2) / 2
                    V_r = np.round(V_copy * 2) / 2
                    W_r = np.round(W_copy * 2) / 2
                    if _verify_332(U_r, V_r, W_r):
                        best_solution = (U_r, V_r, W_r)
                        return best_solution
                    
                    # Try to find nearby half-integer solution
                    # Use local search around current solution
                    for _ in range(50):
                        U_pert = U_copy + rng.randn(nm, r) * 0.01
                        V_pert = V_copy + rng.randn(mp, r) * 0.01
                        W_pert = W_copy + rng.randn(pn, r) * 0.01
                        U_r2 = np.round(U_pert * 2) / 2
                        V_r2 = np.round(V_pert * 2) / 2
                        W_r2 = np.round(W_pert * 2) / 2
                        if _verify_332(U_r2, V_r2, W_r2):
                            best_solution = (U_r2, V_r2, W_r2)
                            return best_solution
                    
                    # If we have a good float solution, try to polish to half-integers
                    if _verify_332(U_copy, V_copy, W_copy):
                        # Try rounding with different offsets
                        for offset in [0, 0.25, 0.5, 0.75]:
                            U_r3 = np.round((U_copy + offset) * 2) / 2 - offset
                            V_r3 = np.round((V_copy + offset) * 2) / 2 - offset
                            W_r3 = np.round((W_copy + offset) * 2) / 2 - offset
                            if _verify_332(U_r3, V_r3, W_r3):
                                best_solution = (U_r3, V_r3, W_r3)
                                return best_solution
                    
                    break
                
                # Adaptive learning rate
                lr = lr_init / (1.0 + it * 0.001)
                
                # Gradient clipping
                max_norm = 10.0
                for mat, grad in [(U_copy, Ug), (V_copy, Vg), (W_copy, Wg)]:
                    gnorm = np.linalg.norm(grad)
                    if gnorm > max_norm:
                        grad *= max_norm / gnorm
                
                U_copy -= lr * Ug
                V_copy -= lr * Vg
                W_copy -= lr * Wg
            
            # Track best loss
            if loss < best_loss:
                best_loss = loss
                best_solution = (U_copy.copy(), V_copy.copy(), W_copy.copy())
    
    # Final check on best solution found
    if best_solution is not None:
        U_b, V_b, W_b = best_solution
        U_r = np.round(U_b * 2) / 2
        V_r = np.round(V_b * 2) / 2
        W_r = np.round(W_b * 2) / 2
        if _verify_332(U_r, V_r, W_r):
            return (U_r, V_r, W_r)
    
    return None

def _verify_332(U, V, W):
    n, m, p = 3, 3, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vi = j * p + k
                wi = k * n + i
                val = np.dot(U[ui], V[vi] * W[wi])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _verify_full(U, V, W, n, m, p):
    nm, mp, pn = n * m, m * p, p * n
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vi = j * p + k
                wi = k * n + i
                val = np.dot(U[ui], V[vi] * W[wi])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _assemble_decomposition(r_332, r_314):
    # Get the actual matrices for 3x3x2
    res = _try_rank_332(r_332)
    if res is None:
        return None, None, None
    
    U_332, V_332, W_332 = res
    
    # Assemble 3x4x4
    # U (12 x R_total), V (16 x R_total), W (12 x R_total)
    R_total = 2 * r_332 + r_314
    U_full = np.zeros((12, R_total))
    V_full = np.zeros((16, R_total))
    W_full = np.zeros((12, R_total))
    
    # Part 1: A1 (rows 0-2) * B1a (cols 0-1 of B, rows 0-2)
    U_full[0:9, 0:r_332] = U_332
    v_indices_1 = [0, 1, 4, 5, 8, 9]
    V_full[v_indices_1, 0:r_332] = V_332
    w_indices_1 = [0, 1, 2, 3, 4, 5]
    W_full[w_indices_1, 0:r_332] = W_332
    
    # Part 2: A1 (rows 0-2) * B1b (cols 2-3 of B, rows 0-2)
    U_full[0:9, r_332:2*r_332] = U_332
    v_indices_2 = [2, 3, 6, 7, 10, 11]
    V_full[v_indices_2, r_332:2*r_332] = V_332
    w_indices_2 = [6, 7, 8, 9, 10, 11]
    W_full[w_indices_2, r_332:2*r_332] = W_332
    
    # Part 3: A2 (col 3 of A) * B2 (row 3 of B)
    u_indices_3 = [3, 7, 11]
    v_indices_3 = [12, 13, 14, 15]
    
    for c in range(12):
        i = c // 4
        k = c % 4
        U_full[u_indices_3[i], 2*r_332 + c] = 1
        V_full[v_indices_3[k], 2*r_332 + c] = 1
        W_full[k*3 + i, 2*r_332 + c] = 1
    
    return U_full, V_full, W_full

def _try_direct_decomposition(n, m, p, rank, n_trials=5):
    """Try to find a direct decomposition of M_{n,m,p} at the given rank."""
    nm, mp, pn = n * m, m * p, p * n
    idx = [(i * m + j, j * p + k, k * n + i)
           for i in range(n) for j in range(m) for k in range(p)]
    
    for t in range(n_trials):
        if time.time() - _START > _TIME_LIMIT - 10:
            return None
        
        rng = np.random.RandomState(t * 104729 + rank)
        
        # Initialize
        U = rng.randn(nm, rank) * 0.3
        V = rng.randn(mp, rank) * 0.3
        W = rng.randn(pn, rank) * 0.3
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(3000):
            if time.time() - _START > _TIME_LIMIT - 10:
                break
            
            # Compute loss and gradients
            Ug = np.zeros_like(U)
            Vg = np.zeros_like(V)
            Wg = np.zeros_like(W)
            loss = 0.0
            
            for ui, vi, wi in idx:
                vw = V[vi] * W[wi]
                uv = U[ui] * V[vi]
                uw = U[ui] * W[wi]
                s = np.dot(U[ui], vw)
                e = s - 1.0
                loss += e * e
                Ug[ui] += e * vw
                Vg[vi] += e * uw
                Wg[wi] += e * uv
            
            if loss < 1e-10:
                # Check for half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_full(U_r, V_r, W_r, n, m, p):
                    return (U_r, V_r, W_r)
                
                # Try nearby half-integer solutions
                for _ in range(100):
                    U_p = U + rng.randn(nm, rank) * 0.05
                    V_p = V + rng.randn(mp, rank) * 0.05
                    W_p = W + rng.randn(pn, rank) * 0.05
                    U_r2 = np.round(U_p * 2) / 2
                    V_r2 = np.round(V_p * 2) / 2
                    W_r2 = np.round(W_p * 2) / 2
                    if _verify_full(U_r2, V_r2, W_r2, n, m, p):
                        return (U_r2, V_r2, W_r2)
                
                break
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Adaptive learning rate with decay
            lr = 0.05 / (1.0 + it * 0.0005)
            
            # Gradient clipping
            for mat, grad in [(U, Ug), (V, Vg), (W, Wg)]:
                gnorm = np.linalg.norm(grad)
                if gnorm > 5.0:
                    grad *= 5.0 / gnorm
            
            U -= lr * Ug
            V -= lr * Vg
            W -= lr * Wg
        
        # Final check on best solution
        if best_U is not None and best_loss < 0.1:
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            if _verify_full(U_r, V_r, W_r, n, m, p):
                return (U_r, V_r, W_r)
    
    return None
