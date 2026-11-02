import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]
DEADLINE = 135.0  # safety margin before 150s kill


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


def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _tensor_error(x, n, m, p, R):
    """Calculate the squared error of the tensor decomposition."""
    U = x[:n*m*R].reshape(n*m, R)
    V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
    W = x[n*m*R + m*p*R : ].reshape(p*n, R)
    
    S = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                S[i,j,k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
    
    return np.sum((S - 1.0)**2)


def _tensor_error_grad(x, n, m, p, R):
    """Calculate gradient of squared tensor error."""
    U = x[:n*m*R].reshape(n*m, R)
    V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
    W = x[n*m*R + m*p*R : ].reshape(p*n, R)
    
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                err = 2.0 * (val - 1.0)
                dU[u_idx, :] += err * V[v_idx, :] * W[w_idx, :]
                dV[v_idx, :] += err * U[u_idx, :] * W[w_idx, :]
                dW[w_idx, :] += err * U[u_idx, :] * V[v_idx, :]
    
    dx = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return dx


def _try_find_rank(R, n, m, p, max_restarts=10, time_limit=8.0):
    """Try to find a decomposition of rank R using L-BFGS-B with exact verification."""
    import scipy.optimize as opt
    start_time = time.time()
    num_vars = R * (n*m + m*p + p*n)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(42 + restart * 137)
        x0 = np.random.uniform(-0.5, 0.5, num_vars)
        
        def obj(x):
            return _tensor_error(x, n, m, p, R)
        
        def obj_grad(x):
            return _tensor_error_grad(x, n, m, p, R)
        
        try:
            res = opt.minimize(obj, x0, jac=obj_grad, method='L-BFGS-B', 
                              options={'maxiter': 500, 'disp': False, 'maxls': 30})
            if res.fun < 1e-12:
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = res.x[n*m*R + m*p*R : ].reshape(p*n, R)
                if _verify_decomposition(U, V, W):
                    return U, V, W
        except Exception:
            pass
            
    return None


def _try_greedy_removal(U, V, W, max_time=20.0):
    """Try to remove columns from the decomposition by checking redundancy."""
    start_time = time.time()
    n, m, p = 2, 4, 5
    R = U.shape[1]
    
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    active = list(range(R))
    
    improved = True
    while improved:
        if time.time() - start_time > max_time:
            break
        improved = False
        R_cur = len(active)
        
        # Try removing single columns
        for idx_remove in range(R_cur):
            if time.time() - start_time > max_time:
                break
            
            test_U = np.delete(U_work[:, active], idx_remove, axis=1)
            test_V = np.delete(V_work[:, active], idx_remove, axis=1)
            test_W = np.delete(W_work[:, active], idx_remove, axis=1)
            
            if _verify_decomposition(test_U, test_V, test_W, tol=1e-10):
                U_work = test_U
                V_work = test_V
                W_work = test_W
                active.pop(idx_remove)
                improved = True
                break
        
        # If no single removal worked, try pairs
        if not improved and R_cur >= 2:
            for i in range(R_cur):
                if time.time() - start_time > max_time:
                    break
                for j in range(i+1, R_cur):
                    if time.time() - start_time > max_time:
                        break
                    
                    test_U = np.delete(U_work[:, active], [i, j], axis=1)
                    test_V = np.delete(V_work[:, active], [i, j], axis=1)
                    test_W = np.delete(W_work[:, active], [i, j], axis=1)
                    
                    if _verify_decomposition(test_U, test_V, test_W, tol=1e-10):
                        U_work = test_U
                        V_work = test_V
                        W_work = test_W
                        active.pop(j)
                        active.pop(i)
                        improved = True
                        break
                if improved:
                    break
        
        # If no pair removal worked, try triples
        if not improved and R_cur >= 3:
            for i in range(R_cur):
                if time.time() - start_time > max_time:
                    break
                for j in range(i+1, R_cur):
                    if time.time() - start_time > max_time:
                        break
                    for k in range(j+1, R_cur):
                        if time.time() - start_time > max_time:
                            break
                        
                        test_U = np.delete(U_work[:, active], [i, j, k], axis=1)
                        test_V = np.delete(V_work[:, active], [i, j, k], axis=1)
                        test_W = np.delete(W_work[:, active], [i, j, k], axis=1)
                        
                        if _verify_decomposition(test_U, test_V, test_W, tol=1e-10):
                            U_work = test_U
                            V_work = test_V
                            W_work = test_W
                            active.pop(k)
                            active.pop(j)
                            active.pop(i)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return U_work, V_work, W_work


def _polish_decomposition(U, V, W, max_time=5.0):
    """Polish a decomposition to exact values using local optimization."""
    import scipy.optimize as opt
    start_time = time.time()
    n, m, p = 2, 4, 5
    R = U.shape[1]
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def obj(x):
        return _tensor_error(x, n, m, p, R)
    
    def obj_grad(x):
        return _tensor_error_grad(x, n, m, p, R)
    
    try:
        res = opt.minimize(obj, x0, jac=obj_grad, method='L-BFGS-B', 
                          options={'maxiter': 200, 'disp': False, 'maxls': 20, 'gtol': 1e-14})
        if res.fun < 1e-14:
            U_new = res.x[:n*m*R].reshape(n*m, R)
            V_new = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W_new = res.x[n*m*R + m*p*R : ].reshape(p*n, R)
            if _verify_decomposition(U_new, V_new, W_new, tol=1e-12):
                return U_new, V_new, W_new
    except Exception:
        pass
    
    return U, V, W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    n, m, p = 2, 4, 5
    
    # Get the baseline rank-36 decomposition
    U_base, V_base, W_base = _build_strassen_36()
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    best_rank = 36
    
    # First try greedy removal on the baseline (including triples)
    if time.time() < deadline - 30:
        U_g, V_g, W_g = _try_greedy_removal(U_base, V_base, W_base, max_time=min(20.0, deadline - start_time - 30))
        if U_g.shape[1] < best_rank:
            best_U, best_V, best_W = U_g, V_g, W_g
            best_rank = U_g.shape[1]
    
    # Try to find lower ranks using randomized search
    for R in range(20, best_rank):
        if time.time() > deadline - 15:
            break
            
        time_limit = min(8.0, deadline - start_time - 15)
        if time_limit < 2.0:
            break
            
        res = _try_find_rank(R, n, m, p, max_restarts=5, time_limit=time_limit)
        if res is not None:
            U, V, W = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = R
    
    # Final greedy removal on the best found
    if time.time() < deadline - 10:
        U_f, V_f, W_f = _try_greedy_removal(best_U, best_V, best_W, max_time=min(10.0, deadline - start_time - 10))
        if U_f.shape[1] < best_rank:
            best_U, best_V, best_W = U_f, V_f, W_f
            best_rank = U_f.shape[1]
    
    # Polish the final decomposition for exactness
    if time.time() < deadline - 5:
        U_p, V_p, W_p = _polish_decomposition(best_U, best_V, best_W, max_time=min(5.0, deadline - start_time - 5))
        if U_p.shape[1] <= best_rank:
            best_U, best_V, best_W = U_p, V_p, W_p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
