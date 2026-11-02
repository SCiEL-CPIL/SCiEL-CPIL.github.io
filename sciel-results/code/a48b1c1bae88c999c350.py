import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _verify_fast(U, V, W, tol=1e-9):
    """Fast vectorized verification of tensor decomposition."""
    n, m, p = 2, 4, 5
    # Precompute index mappings
    u_rows = []
    v_rows = []
    w_rows = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_rows.append(i * m + j)
                v_rows.append(j * p + k)
                w_rows.append(k * n + i)
    
    u_rows = np.array(u_rows)
    v_rows = np.array(v_rows)
    w_rows = np.array(w_rows)
    
    # Vectorized computation: for each (i,j,k), compute dot(U[u], V[v]*W[w])
    U_sel = U[u_rows, :]  # (n*m*p, rank)
    V_sel = V[v_rows, :]  # (n*m*p, rank)
    W_sel = W[w_rows, :]  # (n*m*p, rank)
    
    # Sum over rank dimension
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    return np.all(np.abs(vals - 1.0) < tol)

def _tensor_error_vec(x, n, m, p, R):
    """Calculate squared error of tensor decomposition (vectorized)."""
    U = x[:n*m*R].reshape(n*m, R)
    V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
    W = x[n*m*R + m*p*R : ].reshape(p*n, R)
    
    # Precomputed indices
    u_rows, v_rows, w_rows = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_rows.append(i * m + j)
                v_rows.append(j * p + k)
                w_rows.append(k * n + i)
    
    u_rows = np.array(u_rows)
    v_rows = np.array(v_rows)
    w_rows = np.array(w_rows)
    
    U_sel = U[u_rows, :]
    V_sel = V[v_rows, :]
    W_sel = W[w_rows, :]
    
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    return np.sum((vals - 1.0)**2)

def _optimize_rank(R, U_init, V_init, W_init, n, m, p, time_limit=8.0, max_restarts=15):
    """Try to find a valid decomposition of rank R using optimization."""
    start_time = time.time()
    
    # Flatten initial solution
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    best_x = None
    best_err = np.inf
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb the initial solution
        if restart == 0:
            x = x0.copy()
        else:
            x = x0 + np.random.normal(0, 0.05, x0.shape)
        
        try:
            res = opt.minimize(
                _tensor_error_vec, x, args=(n, m, p, R),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x.copy()
            
            if res.fun < 1e-10:
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = res.x[n*m*R + m*p*R : ].reshape(p*n, R)
                if _verify_fast(U, V, W, tol=1e-9):
                    return U, V, W
        except Exception:
            pass
    
    # Try the best found
    if best_x is not None and best_err < 1e-8:
        U = best_x[:n*m*R].reshape(n*m, R)
        V = best_x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = best_x[n*m*R + m*p*R : ].reshape(p*n, R)
        if _verify_fast(U, V, W, tol=1e-8):
            return U, V, W
    
    return None

def _greedy_reduce(U, V, W, n, m, p, max_iterations=50, time_limit=20.0):
    """Greedy column removal with fast verification."""
    start_time = time.time()
    rank = U.shape[1]
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        progress = False
        for col in range(rank):
            if time.time() - start_time > time_limit:
                break
            
            mask = [i for i in range(rank) if i != col]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if _verify_fast(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank = U.shape[1]
                progress = True
                break
        
        if not progress:
            break
    
    return U, V, W

def _polish(U, V, W, n, m, p, time_limit=10.0):
    """Final polish pass with tighter tolerance."""
    start_time = time.time()
    R = U.shape[1]
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        res = opt.minimize(
            _tensor_error_vec, x0, args=(n, m, p, R),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14, 'disp': False}
        )
        
        U_new = res.x[:n*m*R].reshape(n*m, R)
        V_new = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W_new = res.x[n*m*R + m*p*R : ].reshape(p*n, R)
        
        if _verify_fast(U_new, V_new, W_new, tol=1e-10):
            return U_new, V_new, W_new
    except Exception:
        pass
    
    return U, V, W

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    n, m, p = 2, 4, 5
    
    # Start with rank-36 Strassen decomposition
    U, V, W = _build_strassen_36()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 36
    
    # Phase 1: Greedy column removal (fast)
    if time.time() < deadline - 30:
        U, V, W = _greedy_reduce(U, V, W, n, m, p, max_iterations=100, time_limit=15.0)
        if U.shape[1] < best_rank and _verify_fast(U, V, W):
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Phase 2: Optimization-based rank reduction
    # Try to reduce from current best rank down to theoretical lower bound
    # Theoretical lower bound for 2x4x5 is at least ceil(2*4*5/2) = 20
    for target_rank in range(best_rank - 1, max(19, best_rank - 10), -1):
        if time.time() > deadline - 15:
            break
        
        # Give more time for harder (lower) ranks
        time_limit = min(10.0, (deadline - time.time() - 10) / 2)
        if time_limit < 2.0:
            break
        
        result = _optimize_rank(target_rank, best_U[:, :target_rank], best_V[:, :target_rank], 
                                best_W[:, :target_rank], n, m, p, time_limit=time_limit, 
                                max_restarts=8)
        
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = target_rank
    
    # Phase 3: Final polish
    if time.time() < deadline - 5:
        best_U, best_V, best_W = _polish(best_U, best_V, best_W, n, m, p, time_limit=8.0)
    
    # Final verification
    if not _verify_fast(best_U, best_V, best_W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
