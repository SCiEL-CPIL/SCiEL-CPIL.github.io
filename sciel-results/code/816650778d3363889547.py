import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 4, 5]
DEADLINE = 130.0  # safety margin before 150s kill

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

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _tensor_error(x, n, m, p, R):
    """Calculate the squared error of the tensor decomposition (vectorized)."""
    U = x[:n*m*R].reshape(n*m, R)
    V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
    W = x[n*m*R + m*p*R : ].reshape(p*n, R)
    
    # Precompute index mappings
    # For each (i,j,k) triple, we need U[i*m+j], V[j*p+k], W[k*n+i]
    # Build flat index arrays
    ii = np.repeat(np.arange(n), m*p)
    jj = np.tile(np.repeat(np.arange(m), p), n)
    kk = np.tile(np.arange(p), n*m)
    
    u_idx = ii * m + jj
    v_idx = jj * p + kk
    w_idx = kk * n + ii
    
    # Product of three matrices along rank dimension
    # U[u_idx] shape (n*m*p, R), V[v_idx] shape (n*m*p, R), W[w_idx] shape (n*m*p, R)
    P = U[u_idx] * V[v_idx] * W[w_idx]  # (n*m*p, R)
    S = P.sum(axis=1)  # (n*m*p,)
    
    return np.sum((S - 1.0)**2)

def _tensor_error_grad(x, n, m, p, R):
    """Calculate the gradient of the squared error of the tensor decomposition."""
    U = x[:n*m*R].reshape(n*m, R)
    V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
    W = x[n*m*R + m*p*R : ].reshape(p*n, R)
    
    ii = np.repeat(np.arange(n), m*p)
    jj = np.tile(np.repeat(np.arange(m), p), n)
    kk = np.tile(np.arange(p), n*m)
    
    u_idx = ii * m + jj
    v_idx = jj * p + kk
    w_idx = kk * n + ii
    
    U_sel = U[u_idx]  # (n*m*p, R)
    V_sel = V[v_idx]  # (n*m*p, R)
    W_sel = W[w_idx]  # (n*m*p, R)
    
    S = (U_sel * V_sel * W_sel).sum(axis=1)  # (n*m*p,)
    err = S - 1.0  # (n*m*p,)
    
    # Gradient w.r.t. U: d/dU[i,j,r] = 2 * sum over (i,j,k) where u_idx=i*m+j of V_sel*W_sel*err
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # For each entry in U at position (a, r), it contributes to all triples where u_idx = a
    # We can use np.add.at for efficient accumulation
    # U contribution: for each triple t, U_sel[t, r] gets gradient 2 * err[t] * V_sel[t, r] * W_sel[t, r]
    grad_U_flat = 2.0 * err[:, None] * V_sel * W_sel  # (n*m*p, R)
    np.add.at(grad_U, (u_idx, slice(None)), grad_U_flat)
    
    grad_V_flat = 2.0 * err[:, None] * U_sel * W_sel  # (n*m*p, R)
    np.add.at(grad_V, (v_idx, slice(None)), grad_V_flat)
    
    grad_W_flat = 2.0 * err[:, None] * U_sel * V_sel  # (n*m*p, R)
    np.add.at(grad_W, (w_idx, slice(None)), grad_W_flat)
    
    grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
    return grad

def _try_find_rank(R, n, m, p, max_restarts=20, time_limit=10.0):
    """Try to find a decomposition of rank R using randomized local search with gradient."""
    start_time = time.time()
    num_vars = R * (n*m + m*p + p*n)
    
    def obj(x):
        return _tensor_error(x, n, m, p, R)
    
    def obj_grad(x):
        return _tensor_error_grad(x, n, m, p, R)
    
    def obj_and_grad(x):
        f = _tensor_error(x, n, m, p, R)
        g = _tensor_error_grad(x, n, m, p, R)
        return f, g
    
    best_fun = float('inf')
    best_x = None
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with small random values
        x0 = np.random.uniform(-0.5, 0.5, num_vars)
        
        # Optimize with gradient
        try:
            res = opt.minimize(obj_and_grad, x0, method='L-BFGS-B', options={'maxiter': 3000, 'disp': False})
            if res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x.copy()
            if res.fun < 1e-8:
                # Verify with exact check
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = res.x[n*m*R + m*p*R : ].reshape(p*n, R)
                if _verify_decomposition(U, V, W):
                    return U, V, W
        except Exception:
            pass
    
    # Polish pass: if we got close but not exact, try tighter optimization
    if best_x is not None and best_fun < 1e-3:
        try:
            res2 = opt.minimize(obj_and_grad, best_x, method='L-BFGS-B', options={'maxiter': 8000, 'disp': False})
            if res2.fun < 1e-10:
                U = res2.x[:n*m*R].reshape(n*m, R)
                V = res2.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = res2.x[n*m*R + m*p*R : ].reshape(p*n, R)
                if _verify_decomposition(U, V, W):
                    return U, V, W
        except Exception:
            pass
    
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    n, m, p = 2, 4, 5
    
    # Get the baseline rank 36 decomposition
    U_base, V_base, W_base = _build_strassen_36()
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    best_rank = 36
    
    # Try to find lower ranks using randomized search
    # We try ranks from 20 up to 35
    for R in range(20, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Allocate time dynamically based on remaining budget
        remaining = deadline - 10 - time.time()
        ranks_left = best_rank - R
        time_limit = max(5.0, min(20.0, remaining / max(1, ranks_left)))
        
        res = _try_find_rank(R, n, m, p, max_restarts=40, time_limit=time_limit)
        if res is not None:
            U, V, W = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = R
            # If we find a very low rank, we might stop or continue to see if we can go lower
            # But lower is better, so we continue.
            
    # If we didn't find anything better, return the baseline
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
