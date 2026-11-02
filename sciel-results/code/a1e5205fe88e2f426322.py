import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 5, 6]
n, m, p = DIMENSIONS
nm, mp, pn = n*m, m*p, p*n

def _target_tensor_block(n_block, m_block, p_block):
    """
    Construct the target tensor for a block of dimensions n_block x m_block x p_block.
    Indices:
    a = i * m_block + j
    b = j * p_block + k
    c = k * n_block + i
    """
    T = np.zeros((n_block * m_block, m_block * p_block, p_block * n_block), dtype=float)
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                a = i * m_block + j
                b = j * p_block + k
                c = k * n_block + i
                T[a, b, c] = 1.0
    return T

def _loss_and_grad(params, T, d1, d2, d3, R):
    """
    Compute loss and gradient for tensor decomposition.
    params: flattened vector of U, V, W.
    U: (d1, R)
    V: (d2, R)
    W: (d3, R)
    """
    size_U = d1 * R
    size_V = d2 * R
    size_W = d3 * R
    
    U = params[0:size_U].reshape(d1, R)
    V = params[size_U:size_U+size_V].reshape(d2, R)
    W = params[size_U+size_V:size_U+size_V+size_W].reshape(d3, R)
    
    # Reconstruction
    # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = recon - T
    loss = np.sum(diff**2)
    
    # Gradient
    # dL/dU[a,r] = 2 * sum_{b,c} (recon[a,b,c] - T[a,b,c]) V[b,r] W[c,r]
    # dL/dV[b,r] = 2 * sum_{a,c} (recon[a,b,c] - T[a,b,c]) U[a,r] W[c,r]
    # dL/dW[c,r] = 2 * sum_{a,b} (recon[a,b,c] - T[a,b,c]) U[a,r] V[b,r]
    
    grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
    grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
    grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    return loss, grad

def _solve_block(n_b, m_b, p_b, R, max_iter=200, seed=0, time_limit=5.0):
    """
    Solve for a block decomposition of rank R.
    Returns U, V, W if successful, else None.
    """
    d1 = n_b * m_b
    d2 = m_b * p_b
    d3 = p_b * n_b
    
    T = _target_tensor_block(n_b, m_b, p_b)
    
    rng = np.random.RandomState(seed)
    
    # Initial guess
    U0 = rng.randn(d1, R) * 0.1
    V0 = rng.randn(d2, R) * 0.1
    W0 = rng.randn(d3, R) * 0.1
    params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    start_time = time.time()
    
    # Use L-BFGS-B
    try:
        res = opt.minimize(
            _loss_and_grad, 
            params0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
        )
    except Exception:
        return None
    
    if time.time() - start_time > time_limit:
        return None
        
    if res.success and res.fun < 1e-6:
        # Extract matrices
        size_U = d1 * R
        size_V = d2 * R
        size_W = d3 * R
        U = res.x[0:size_U].reshape(d1, R)
        V = res.x[size_U:size_U+size_V].reshape(d2, R)
        W = res.x[size_U+size_V:size_U+size_V+size_W].reshape(d3, R)
        
        # Try to round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify
        recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
        if np.max(np.abs(recon - T)) < 1e-5:
            return U_r, V_r, W_r
            
    return None

def _verify_full(U, V, W, tol=1e-8):
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.max(np.abs(recon - T)) < tol

def _to_list(M):
    M = np.asarray(M, dtype=float)
    out = []
    for row in M:
        out_row = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                out_row.append(int(round(val)))
            elif abs(val * 2 - round(val * 2)) < 1e-9:
                out_row.append(round(val * 2) / 2.0)
            else:
                out_row.append(val)
        out.append(out_row)
    return out

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Define blocks
    # Block 1: 3 x 2 x 6
    n1, m1, p1 = 3, 2, 6
    # Block 2: 3 x 3 x 6
    n2, m2, p2 = 3, 3, 6
    
    # Lower bounds
    R1_lb = max(n1*m1, m1*p1, p1*n1)
    R2_lb = max(n2*m2, m2*p2, p2*n2)
    
    # We will search for R1 and R2
    # We want to minimize R1 + R2
    best_R1, best_R2 = None, None
    best_U1, best_V1, best_W1 = None, None, None
    best_U2, best_V2, best_W2 = None, None, None
    
    # Try to find a solution for Block 1
    found1 = False
    for R1 in range(R1_lb, R1_lb + 20):
        if time.time() > deadline - 60: break
        for seed in range(5):
            if time.time() > deadline - 60: break
            res = _solve_block(n1, m1, p1, R1, max_iter=100, seed=seed, time_limit=3.0)
            if res:
                best_U1, best_V1, best_W1 = res
                best_R1 = R1
                found1 = True
                break
        if found1: break
    if not found1:
        # Fallback to schoolbook for block 1
        R1 = n1 * m1 * p1
        best_R1 = R1
        best_U1 = np.zeros((n1*m1, R1))
        best_V1 = np.zeros((m1*p1, R1))
        best_W1 = np.zeros((p1*n1, R1))
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    a = i*m1 + j
                    b = j*p1 + k
                    c = k*n1 + i
                    best_U1[a, col] = 1
                    best_V1[b, col] = 1
                    best_W1[c, col] = 1
                    col += 1
                    
    # Try to find a solution for Block 2
    found2 = False
    for R2 in range(R2_lb, R2_lb + 20):
        if time.time() > deadline - 30: break
        for seed in range(5):
            if time.time() > deadline - 30: break
            res = _solve_block(n2, m2, p2, R2, max_iter=100, seed=seed, time_limit=3.0)
            if res:
                best_U2, best_V2, best_W2 = res
                best_R2 = R2
                found2 = True
                break
        if found2: break
    if not found2:
        # Fallback to schoolbook for block 2
        R2 = n2 * m2 * p2
        best_R2 = R2
        best_U2 = np.zeros((n2*m2, R2))
        best_V2 = np.zeros((m2*p2, R2))
        best_W2 = np.zeros((p2*n2, R2))
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    a = i*m2 + j
                    b = j*p2 + k
                    c = k*n2 + i
                    best_U2[a, col] = 1
                    best_V2[b, col] = 1
                    best_W2[c, col] = 1
                    col += 1

    # Combine
    R_total = best_R1 + best_R2
    U_full = np.zeros((nm, R_total))
    V_full = np.zeros((mp, R_total))
    W_full = np.zeros((pn, R_total))
    
    # Block 1 indices in full space
    # U1 rows: i*2+j1 (j1 in 0,1). Full U rows: i*5+j.
    # j=0,1 correspond to j1=0,1.
    # So U1 row i*2+j1 maps to U_full row i*5+j1.
    for i in range(n1):
        for j1 in range(m1):
            a1 = i*m1 + j1
            a_full = i*m + j1
            U_full[a_full, 0:best_R1] = best_U1[a1, :]
            
    # V1 rows: j1*6+k. Full V rows: j*6+k.
    # j=0,1 correspond to j1=0,1.
    # So V1 row j1*6+k maps to V_full row j1*6+k.
    V_full[0:m1*p1, 0:best_R1] = best_V1
    
    # W1 rows: k*3+i. Full W rows: k*3+i.
    # Same mapping.
    W_full[0:p1*n1, 0:best_R1] = best_W1
    
    # Block 2 indices in full space
    # U2 rows: i*3+j2 (j2 in 0,1,2). Full U rows: i*5+j.
    # j=2,3,4 correspond to j2=0,1,2.
    for i in range(n2):
        for j2 in range(m2):
            a2 = i*m2 + j2
            a_full = i*m + (m1 + j2)
            U_full[a_full, best_R1:best_R1+best_R2] = best_U2[a2, :]
            
    # V2 rows: j2*6+k. Full V rows: j*6+k.
    # j=2,3,4 correspond to j2=0,1,2.
    # V2 row j2*6+k maps to V_full row (m1+j2)*6+k.
    for j2 in range(m2):
        for k in range(p2):
            b2 = j2*p2 + k
            b_full = (m1+j2)*p2 + k
            V_full[b_full, best_R1:best_R1+best_R2] = best_V2[b2, :]
            
    # W2 rows: k*3+i. Full W rows: k*3+i.
    # Same mapping.
    W_full[0:p2*n2, best_R1:best_R1+best_R2] = best_W2
    
    # Verify
    if _verify_full(U_full, V_full, W_full):
        return {"U": _to_list(U_full), "V": _to_list(V_full), "W": _to_list(W_full)}
        
    # Fallback to full schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
