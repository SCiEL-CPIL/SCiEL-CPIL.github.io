import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i*m+j, j*p+k, k*n+i)
                T[key] = T.get(key, 0) + 1
    return T

def _verify_exact(U, V, W, n, m, p, rank):
    """Verify decomposition using Fraction arithmetic for exactness."""
    target = _target_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                s = Fraction(0)
                for r in range(rank):
                    u_val = U[a][r]
                    v_val = V[b][r]
                    w_val = W[c][r]
                    if isinstance(u_val, float):
                        u_frac = Fraction(u_val).limit_denominator(1000000)
                    else:
                        u_frac = u_val if isinstance(u_val, Fraction) else Fraction(u_val)
                    if isinstance(v_val, float):
                        v_frac = Fraction(v_val).limit_denominator(1000000)
                    else:
                        v_frac = v_val if isinstance(v_val, Fraction) else Fraction(v_val)
                    if isinstance(w_val, float):
                        w_frac = Fraction(w_val).limit_denominator(1000000)
                    else:
                        w_frac = w_val if isinstance(w_val, Fraction) else Fraction(w_val)
                    s += u_frac * v_frac * w_frac
                if s != target.get((a, b, c), 0):
                    return False
    return True

def _verify_numeric(U, V, W, n, m, p, rank, tol=1e-9):
    """Fast numeric verification."""
    target = _target_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                if abs(s - target.get((a, b, c), 0)) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _build_strassen_rank22(n, m, p):
    """Build the known rank-22 Strassen-based decomposition for 3x3 by 3x4."""
    cols = []
    
    def strassen_block(ai, aj, bj, bk, ci, ck):
        """Apply Strassen to a 2x2 by 2x2 block."""
        def A(i, j): return (ai+i, aj+j)
        def B(j, k): return (bj+j, bk+k)
        def C(i, k): return (ci+i, ck+k)
        M = [
            ([(A(0,0),1),(A(1,1),1)], [(B(0,0),1),(B(1,1),1)], [(C(0,0),1),(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),1)], [(B(0,0),1)], [(C(1,0),1),(C(1,1),-1)]),
            ([(A(0,0),1)], [(B(1,0),1),(B(1,1),-1)], [(C(0,1),1),(C(1,1),1)]),
            ([(A(1,1),1)], [(B(0,1),1),(B(1,0),-1)], [(C(0,0),1),(C(1,0),1)]),
            ([(A(0,0),1),(A(0,1),1)], [(B(1,0),1)], [(C(0,0),-1),(C(0,1),1)]),
            ([(A(0,1),1),(A(0,0),-1)], [(B(0,0),1),(B(1,0),1)], [(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),-1)], [(B(1,1),1),(B(0,1),1)], [(C(0,0),1),(C(0,1),-1)]),
        ]
        for Ac, Bc, Cc in M:
            cols.append((dict(Ac), dict(Bc), dict(Cc)))
    
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_block(0, 0, 0, 0, 0, 0)
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_block(0, 0, 0, 2, 0, 2)
    
    # A[0:2, 2:3] * B[2:3, 0:2] -> C[0:2, 0:2]  (rank 4 block)
    cols.append((
        {(0,2):1, (1,2):1},
        {(2,0):1, (2,1):1},
        {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    ))
    # A[0:2, 2:3] * B[2:3, 2:4] -> C[0:2, 2:4]  (rank 4 block)
    cols.append((
        {(0,2):1, (1,2):1},
        {(2,2):1, (2,3):1},
        {(0,2):1, (0,3):1, (1,2):1, (1,3):1}
    ))
    
    # A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]  (rank 2)
    for k in range(2):
        cols.append((
            {(2,0):1, (2,1):1},
            {(0,k):1, (1,k):1},
            {(2,k):1}
        ))
    # A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]  (rank 2)
    for k in range(2):
        cols.append((
            {(2,0):1, (2,1):1},
            {(0,2+k):1, (1,2+k):1},
            {(2,2+k):1}
        ))
    # A[2:3, 2:3] * B[2:3, 0:2] -> C[2:3, 0:2]  (rank 1)
    cols.append((
        {(2,2):1},
        {(2,0):1, (2,1):1},
        {(2,0):1, (2,1):1}
    ))
    # A[2:3, 2:3] * B[2:3, 2:4] -> C[2:3, 2:4]  (rank 1)
    cols.append((
        {(2,2):1},
        {(2,2):1, (2,3):1},
        {(2,2):1, (2,3):1}
    ))
    
    rank = len(cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (Ac, Bc, Cc) in enumerate(cols):
        for (i, j), v in Ac.items():
            U[i*m+j][r] += v
        for (j, k), v in Bc.items():
            V[j*p+k][r] += v
        for (i, k), v in Cc.items():
            W[k*n+i][r] += v
    return U, V, W, rank

def _adam_optimize(U, V, W, T, n, m, p, target_rank, lr=0.05, max_iters=5000, deadline=0, tol=1e-12):
    """Adam optimizer for tensor decomposition refinement."""
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    
    t = 0
    cur_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = float('inf')
    
    while t < max_iters:
        if deadline > 0 and time.time() > deadline:
            break
        t += 1
        lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        
        err = np.einsum('ar,br,cr->abc', U, V, W) - T
        
        gU = np.einsum('abc,br,cr->ar', err, V, W)
        gV = np.einsum('abc,ar,cr->br', err, U, W)
        gW = np.einsum('abc,ar,br->cr', err, U, V)
        
        m_U = beta1*m_U + (1-beta1)*gU
        v_U = beta2*v_U + (1-beta2)*gU**2
        m_U_hat = m_U / (1 - beta1**t)
        v_U_hat = v_U / (1 - beta2**t)
        U -= lr_t * m_U_hat / (np.sqrt(v_U_hat) + eps)
        
        m_V = beta1*m_V + (1-beta1)*gV
        v_V = beta2*v_V + (1-beta2)*gV**2
        m_V_hat = m_V / (1 - beta1**t)
        v_V_hat = v_V / (1 - beta2**t)
        V -= lr_t * m_V_hat / (np.sqrt(v_V_hat) + eps)
        
        m_W = beta1*m_W + (1-beta1)*gW
        v_W = beta2*v_W + (1-beta2)*gW**2
        m_W_hat = m_W / (1 - beta1**t)
        v_W_hat = v_W / (1 - beta2**t)
        W -= lr_t * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
        cur_loss = np.sum(err**2)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if cur_loss < tol:
            break
    
    return best_U, best_V, best_W, best_loss

def _try_lower_rank(target_rank, n, m, p, deadline, seed=0):
    """
    Try to find a rank-target_rank decomposition by starting from the 
    Strassen rank-22 solution and optimizing, or from random init.
    """
    np.random.seed(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                T[a, b, c] = 1.0
    
    best_U, best_V, best_W = None, None, None
    best_loss = float('inf')
    
    # Strategy 1: Start from Strassen rank-22, truncate/pad to target_rank
    if target_rank <= 22:
        U22, V22, W22, r22 = _build_strassen_rank22(n, m, p)
        U = np.array(U22, dtype=float)[:, :target_rank]
        V = np.array(V22, dtype=float)[:, :target_rank]
        W = np.array(W22, dtype=float)[:, :target_rank]
        
        # Add small random perturbation to escape local minima
        U += np.random.randn(nm, target_rank) * 0.05
        V += np.random.randn(mp, target_rank) * 0.05
        W += np.random.randn(pn, target_rank) * 0.05
        
        U, V, W, loss = _adam_optimize(U, V, W, T, n, m, p, target_rank, lr=0.03, max_iters=3000, deadline=deadline)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U, V, W
        
        if best_loss < 1e-8:
            return best_U, best_V, best_W, best_loss
    
    # Strategy 2: Random initialization with multiple scales
    for attempt in range(4):
        if time.time() > deadline:
            break
        scale = [1.0, 0.5, 2.0, 0.1][attempt % 4]
        U = np.random.randn(nm, target_rank) * scale
        V = np.random.randn(mp, target_rank) * scale
        W = np.random.randn(pn, target_rank) * scale
        
        U, V, W, loss = _adam_optimize(U, V, W, T, n, m, p, target_rank, lr=0.05, max_iters=2000, deadline=deadline)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U, V, W
        
        if best_loss < 1e-8:
            break
    
    if best_U is None or best_loss > 1e-6:
        return None, None, None, best_loss
    
    # Snap to rationals and verify
    def try_snap(arr):
        snapped = []
        for row in arr:
            snapped_row = []
            for val in row:
                snapped_row.append(Fraction(val).limit_denominator(100000))
            snapped.append(snapped_row)
        return snapped
    
    U_frac = try_snap(best_U)
    V_frac = try_snap(best_V)
    W_frac = try_snap(best_W)
    
    if _verify_exact(U_frac, V_frac, W_frac, n, m, p, target_rank):
        return U_frac, V_frac, W_frac, 0.0
    
    # Try with float verification
    if _verify_numeric(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, target_rank, tol=1e-9):
        return U_frac, V_frac, W_frac, best_loss
    
    return None, None, None, best_loss

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 140.0  # Safety margin from 150s limit
    
    # Start with the known rank-22 Strassen solution
    U22, V22, W22, rank22 = _build_strassen_rank22(n, m, p)
    
    # Verify the Strassen solution
    if _verify_numeric(U22, V22, W22, n, m, p, rank22, tol=1e-9):
        best_result = (U22, V22, W22, rank22)
    else:
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p) + (n*m*p,)
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Try ranks from 20 down to 12
    for target_rank in [20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() - t0 > DEADLINE - 15:
            break
        
        found = False
        for seed in range(3):
            if time.time() - t0 > DEADLINE - 10:
                break
            
            U, V, W, loss_val = _try_lower_rank(
                target_rank, n, m, p,
                deadline=time.time() + (DEADLINE - 8 - (time.time() - t0)),
                seed=seed
            )
            
            if U is not None and loss_val < 1e-6:
                if _verify_numeric(U, V, W, n, m, p, target_rank, tol=1e-8):
                    if _verify_exact(U, V, W, n, m, p, target_rank):
                        best_result = (U, V, W, target_rank)
                        found = True
                        break
        
        if found:
            continue  # Try lower rank
        else:
            continue
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
