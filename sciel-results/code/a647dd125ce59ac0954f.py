import time
import numpy as np
import random

def _build_strassen_4x4_4x4():
    """Build Strassen-based decomposition for 4x4 * 4x4 -> 4x4 with rank 49."""
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    # Strassen 2x2
    sU = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    sV = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    sW = np.array([
        [1, 0, 1, 1, -1, 1, -1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, -1, 1, 0, 1, 1, 1]
    ])
    
    # 7 block mults
    mults = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {2:1, 0:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 3:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {2:1, 3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:-1, 3:1}),
    ]
    
    for mi, (a_c, b_c, c_c) in enumerate(mults):
        for s in range(7):
            col = mi * 7 + s
            for xi in range(4):
                u_val = sU[xi, s]
                if u_val == 0: continue
                xr = xi // 2
                xc = xi % 2
                for ab, coeff in a_c.items():
                    global_u = ab * 4 + xr * 2 + xc
                    U[global_u, col] += u_val * coeff
            for yi in range(4):
                v_val = sV[yi, s]
                if v_val == 0: continue
                yr = yi // 2
                yc = yi % 2
                for bb, coeff in b_c.items():
                    global_v = bb * 4 + yr * 2 + yc
                    V[global_v, col] += v_val * coeff
            for zi in range(4):
                w_val = sW[zi, s]
                if w_val == 0: continue
                zr = zi // 2
                zc = zi % 2
                for cb, coeff in c_c.items():
                    global_w = cb * 4 + zr * 2 + zc
                    W[global_w, col] += w_val * coeff
    return U, V, W

def _build_extra_column():
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """Compute sum of squared errors."""
    err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                diff = s - 1
                err += diff * diff
    return err

def _count_nonzero_cols(U, V, W):
    """Count columns that are not all zero in any of U, V, W."""
    # A column c is "active" if U[:,c] or V[:,c] or W[:,c] is not zero.
    # Actually, if one is zero, the term is zero.
    # So rank is the number of c where U[:,c] != 0 AND V[:,c] != 0 AND W[:,c] != 0.
    rank = 0
    for c in range(U.shape[1]):
        if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0):
            rank += 1
    return rank

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Build initial 65-rank solution
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    U_e, V_e, W_e = _build_extra_column()
    
    total_rank = 65
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    
    U[:, :49] = U_s
    for j in range(m):
        for k in range(4):
            V[j*p+k, :49] = V_s[j*4+k, :]
    for k in range(4):
        for i in range(n):
            W[k*n+i, :49] = W_s[k*4+i, :]
            
    U[:, 49:] = U_e
    for j in range(m):
        V[j*p+4, 49:] = V_e[j*1+0, :]
    for i in range(n):
        W[4*n+i, 49:] = W_e[0*n+i, :]
        
    # Verify initial
    err = _compute_error(U, V, W, n, m, p)
    if err > 1e-9:
        # Fallback
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = total_rank
    
    # 2. Local Search / Hill Climbing to reduce rank
    # We try to zero out columns by perturbing coefficients.
    # If we can make a column zero without increasing error, we reduce rank.
    
    current_err = 0
    current_rank = total_rank
    
    # To speed up, we can pre-compute contributions?
    # No, just loop.
    
    while time.time() < deadline:
        # Pick a random active column
        active_cols = [c for c in range(total_rank) if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0)]
        if not active_cols:
            break
            
        c = random.choice(active_cols)
        
        # Pick a random matrix and entry to change
        mat_idx = random.choice([0, 1, 2]) # 0:U, 1:V, 2:W
        if mat_idx == 0:
            row = random.randint(0, n*m-1)
            old_val = U[row, c]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val: continue
            U[row, c] = new_val
            new_err = _compute_error(U, V, W, n, m, p)
            U[row, c] = old_val # revert
        elif mat_idx == 1:
            row = random.randint(0, m*p-1)
            old_val = V[row, c]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val: continue
            V[row, c] = new_val
            new_err = _compute_error(U, V, W, n, m, p)
            V[row, c] = old_val
        else:
            row = random.randint(0, p*n-1)
            old_val = W[row, c]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val: continue
            W[row, c] = new_val
            new_err = _compute_error(U, V, W, n, m, p)
            W[row, c] = old_val
            
        # Accept if error is still 0 (or lower, but it's 0)
        # And if rank is reduced?
        # Actually, we want to maintain error 0.
        # If new_err == 0, we accept the change.
        # Then we check if rank reduced.
        
        if new_err == 0:
            # Apply change
            if mat_idx == 0:
                U[row, c] = new_val
            elif mat_idx == 1:
                V[row, c] = new_val
            else:
                W[row, c] = new_val
                
            # Check if this column is now zero
            if not (np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0)):
                # Column is dead, rank reduced
                current_rank -= 1
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
            # Also, maybe other columns became dead? Unlikely, but check.
            # Just update current_rank
            current_rank = _count_nonzero_cols(U, V, W)
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Final check
    if _compute_error(best_U, best_V, best_W, n, m, p) == 0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Return the initial 65-rank solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
