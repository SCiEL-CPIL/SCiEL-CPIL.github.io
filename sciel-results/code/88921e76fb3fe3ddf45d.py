import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 4]

def _verify(U, V, W):
    """Verify that U @ V @ W decomposition is correct."""
    n, m, p = DIMENSIONS
    n, m, p = 2, 2, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for jp in range(m):
                for k in range(p):
                    b = jp * p + k
                    for ip in range(n):
                        for kp in range(p):
                            c = kp * n + ip
                            target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            val = float(np.dot(np.dot(U[a, :], V[b, :]), W[c, :]))
                            max_err = max(max_err, abs(val - target))
    return max_err

def _try_rank13():
    """
    Try to find a rank-13 decomposition by exploiting block structure.
    
    The 2x2x4 tensor is block-diagonal: two independent 2x2x2 blocks.
    Block 0: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    Block 1: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    
    Each block independently needs rank >= 7 (Strassen).
    Total independent: 14.
    
    To get rank 13, we need to share one rank-1 term between blocks.
    A shared term has U[a,r], V[b,r], W[c,r] for some r.
    For it to contribute to both blocks, it must have:
    - U entries in both blocks (all U entries are shared since A is 2x2)
    - V entries in both B-blocks (V has 8 entries: 4 for block 0, 4 for block 1)
    - W entries in both C-blocks (W has 8 entries: 4 for block 0, 4 for block 1)
    
    Strategy: Take the rank-14 solution, pick one term from block 0 and one from block 1,
    and try to merge them into a single term. This means finding a new rank-1 term that
    replaces both, plus adjusting the remaining 12 terms.
    
    Actually, a cleaner approach: parameterize the rank-13 solution directly.
    Let the 13 terms be r=0..12.
    Terms 0..5: block 0 only (V entries only in first 4, W entries only in first 4)
    Terms 6..11: block 1 only (V entries only in last 4, W entries only in last 4)
    Term 12: shared (can have entries in all V and W positions)
    
    This gives us a structured parameterization. We solve the nonlinear system.
    """
    n, m, p = 2, 2, 4
    rank = 13
    
    # Variable layout:
    # U: 4 * 13 = 52
    # V: 8 * 13 = 104
    # W: 8 * 13 = 104
    # Total: 260
    
    # But with structure:
    # For terms 0..5 (block 0 only): V[4:8, r] = 0, W[4:8, r] = 0
    # For terms 6..11 (block 1 only): V[0:4, r] = 0, W[0:4, r] = 0
    # For term 12 (shared): no restrictions
    
    # Free variables:
    # U: 4*13 = 52
    # V: terms 0-5: 4*6=24, terms 6-11: 4*6=24, term 12: 8*1=8. Total V: 56
    # W: terms 0-5: 4*6=24, terms 6-11: 4*6=24, term 12: 8*1=8. Total W: 56
    # Total free: 52 + 56 + 56 = 164
    
    # Equations: 256 (all tensor entries)
    # But many are trivially 0 due to structure. Let's count non-trivial ones.
    # Actually, let's just set up the full system and use the structure to reduce variables.
    
    def residuals(x):
        # Unpack
        U = x[:52].reshape(4, 13)
        
        # V: 56 free variables
        # Terms 0-5: V[0:4, 0:6] (24 vars)
        # Terms 6-11: V[4:8, 6:12] (24 vars)  
        # Term 12: V[:, 12] (8 vars)
        V = np.zeros((8, 13))
        V[0:4, 0:6] = x[52:76].reshape(4, 6)
        V[4:8, 6:12] = x[76:100].reshape(4, 6)
        V[:, 12] = x[100:108]
        
        # W: 56 free variables
        W = np.zeros((8, 13))
        W[0:4, 0:6] = x[108:132].reshape(4, 6)
        W[4:8, 6:12] = x[132:156].reshape(4, 6)
        W[:, 12] = x[156:164]
        
        # Compute residuals for all 256 tensor entries
        res = np.zeros(256)
        idx = 0
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for jp in range(m):
                    for k in range(p):
                        b = jp * p + k
                        for ip in range(n):
                            for kp in range(p):
                                c = kp * n + ip
                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                                val = 0.0
                                for r in range(rank):
                                    val += U[a, r] * V[b, r] * W[c, r]
                                res[idx] = val - target
                                idx += 1
        return res
    
    start_time = time.time()
    deadline = start_time + 100  # 100 second budget for rank-13 search
    
    best_result = None
    best_err = float('inf')
    
    from scipy.optimize import least_squares
    
    # Multiple random restarts
    np.random.seed(42)
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Initialize with Strassen-like structure
        x0 = np.zeros(164)
        
        # For block 0 (terms 0-5), use Strassen-inspired initialization
        # Strassen for 2x2x2 has 7 terms. We use 6 here + shared term.
        # Let's just use random initialization scaled appropriately
        x0 = np.random.randn(164) * 0.5
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_result = sol.x
                if best_err < 1e-8:
                    break
        except Exception:
            pass
    
    if best_result is not None and best_err < 1e-6:
        x = best_result
        U = x[:52].reshape(4, 13)
        V = np.zeros((8, 13))
        V[0:4, 0:6] = x[52:76].reshape(4, 6)
        V[4:8, 6:12] = x[76:100].reshape(4, 6)
        V[:, 12] = x[100:108]
        W = np.zeros((8, 13))
        W[0:4, 0:6] = x[108:132].reshape(4, 6)
        W[4:8, 6:12] = x[132:156].reshape(4, 6)
        W[:, 12] = x[156:164]
        
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        err = _verify(U_r.tolist(), V_r.tolist(), W_r.tolist())
        if err < 1e-10:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Also try rounding to integers
        U_r2 = np.round(U)
        V_r2 = np.round(V)
        W_r2 = np.round(W)
        err2 = _verify(U_r2.tolist(), V_r2.tolist(), W_r2.tolist())
        if err2 < 1e-10:
            return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}
    
    return None

def _try_rank12():
    """
    Try rank 12 with structured sparsity.
    
    For rank 12, we need even more sharing. Let's try:
    Terms 0-3: block 0 only
    Terms 4-7: block 1 only  
    Terms 8-11: shared
    
    Free variables:
    U: 4*12 = 48
    V: terms 0-3: 4*4=16, terms 4-7: 4*4=16, terms 8-11: 8*4=32. Total: 64
    W: same as V: 64
    Total: 176
    
    This is still underdetermined (176 vars, 256 eqs) but the structure helps.
    """
    n, m, p = 2, 2, 4
    rank = 12
    
    def residuals(x):
        U = x[:48].reshape(4, 12)
        
        V = np.zeros((8, 12))
        V[0:4, 0:4] = x[48:64].reshape(4, 4)
        V[4:8, 4:8] = x[64:80].reshape(4, 4)
        V[:, 8:12] = x[80:112].reshape(8, 4)
        
        W = np.zeros((8, 12))
        W[0:4, 0:4] = x[112:128].reshape(4, 4)
        W[4:8, 4:8] = x[128:144].reshape(4, 4)
        W[:, 8:12] = x[144:176].reshape(8, 4)
        
        res = np.zeros(256)
        idx = 0
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for jp in range(m):
                    for k in range(p):
                        b = jp * p + k
                        for ip in range(n):
                            for kp in range(p):
                                c = kp * n + ip
                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                val = 0.0
                                for r in range(rank):
                                    val += U[a, r] * V[b, r] * W[c, r]
                                res[idx] = val - target
                                idx += 1
        return res
    
    start_time = time.time()
    deadline = start_time + 80
    
    best_result = None
    best_err = float('inf')
    
    from scipy.optimize import least_squares
    
    np.random.seed(123)
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        x0 = np.random.randn(176) * 0.5
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=400, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_result = sol.x
                if best_err < 1e-8:
                    break
        except Exception:
            pass
    
    if best_result is not None and best_err < 1e-6:
        x = best_result
        U = x[:48].reshape(4, 12)
        V = np.zeros((8, 12))
        V[0:4, 0:4] = x[48:64].reshape(4, 4)
        V[4:8, 4:8] = x[64:80].reshape(4, 4)
        V[:, 8:12] = x[80:112].reshape(8, 4)
        W = np.zeros((8, 12))
        W[0:4, 0:4] = x[112:128].reshape(4, 4)
        W[4:8, 4:8] = x[128:144].reshape(4, 4)
        W[:, 8:12] = x[144:176].reshape(8, 4)
        
        # Try rounding
        for factor in [2, 1]:
            U_r = np.round(U * factor) / factor
            V_r = np.round(V * factor) / factor
            W_r = np.round(W * factor) / factor
            err = _verify(U_r.tolist(), V_r.tolist(), W_r.tolist())
            if err < 1e-10:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None

def _rank14_solution():
    """Return the known rank-14 solution."""
    rank = 14
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(8)]
    W = [[0.0]*rank for _ in range(8)]
    
    def set_col(col, u_dict, v_dict, w_dict):
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
            
    # Left block (B[:, 0:2])
    set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    set_col(1, {2:1, 3:1}, {0:1}, {1:1, 3:-1})
    set_col(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
    set_col(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
    set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
    set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})
    
    # Right block (B[:, 2:4])
    set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
    set_col(8, {2:1, 3:1}, {2:1}, {5:1, 7:-1})
    set_col(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
    set_col(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
    set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
    set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
    set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # First try rank 12
    if time.time() < deadline - 30:
        result = _try_rank12()
        if result is not None:
            return result
    
    # Then try rank 13
    if time.time() < deadline - 10:
        result = _try_rank13()
        if result is not None:
            return result
    
    # Fallback to rank 14
    return _rank14_solution()
