import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 130s deadline, 20s safety margin
    
    # Strategy: Find minimal rank for M_{2,3,2}, then repeat p/2 = 3 times
    n_sub, m_sub, p_sub = 2, 3, 2
    repeats = p // p_sub  # = 3
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from lower bound upward
    # Lower bound for M_{2,3,2} is max(6, 6, 4) = 6
    for r in range(6, 13):
        if time.time() > deadline - 5:
            break
        found = _find_sub_decomposition(n_sub, m_sub, p_sub, r, deadline)
        if found is not None:
            best_rank = r
            best_U, best_V, best_W = found
            break
    
    if best_rank is not None:
        full_rank = best_rank * repeats
        U_full = [[0] * full_rank for _ in range(n * m)]
        V_full = [[0] * full_rank for _ in range(m * p)]
        W_full = [[0] * full_rank for _ in range(p * n)]
        
        for rep in range(repeats):
            offset = rep * best_rank
            # U: i*m+j -> same indices
            for i in range(n_sub):
                for j in range(m_sub):
                    u_idx = i * m_sub + j
                    for c in range(best_rank):
                        U_full[u_idx][offset + c] = best_U[u_idx][c]
            
            # V: j*p+k, k = rep*p_sub + k_sub
            for j in range(m_sub):
                for k_sub in range(p_sub):
                    k_full = rep * p_sub + k_sub
                    v_idx = j * p_sub + k_sub
                    v_full_idx = j * p + k_full
                    for c in range(best_rank):
                        V_full[v_full_idx][offset + c] = best_V[v_idx][c]
            
            # W: k*n+i, k = rep*p_sub + k_sub
            for k_sub in range(p_sub):
                k_full = rep * p_sub + k_sub
                for i in range(n_sub):
                    w_idx = k_sub * n_sub + i
                    w_full_idx = k_full * n + i
                    for c in range(best_rank):
                        W_full[w_full_idx][offset + c] = best_W[w_idx][c]
        
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: Schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _is_half_integer(arr, atol=1e-6):
    """Check if all entries are integer or half-integer"""
    doubled = arr * 2
    rounded = np.round(doubled)
    return np.allclose(doubled, rounded, atol=atol)


def _find_sub_decomposition(n, m, p, r, deadline):
    """Find rank-r decomposition for M_{n,m,p} with half-integer entries"""
    
    # Try structured patterns first
    result = _try_structured(n, m, p, r)
    if result is not None:
        return result
    
    # Systematic random search
    for seed in range(200):
        if time.time() > deadline - 5:
            return None
        
        random.seed(seed)
        
        # Try different value ranges
        for val_range in [1, 2, 3, 5]:
            if time.time() > deadline - 5:
                break
            
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                
                U = np.array([[random.randint(-val_range, val_range) for _ in range(r)] for _ in range(n*m)], dtype=float)
                W = np.array([[random.randint(-val_range, val_range) for _ in range(r)] for _ in range(p*n)], dtype=float)
                
                # Skip if U or W is all zeros in any row
                if np.any(np.all(U == 0, axis=1)) or np.any(np.all(W == 0, axis=1)):
                    continue
                
                V_sol = np.zeros((m*p, r), dtype=float)
                valid = True
                
                for j in range(m):
                    for k in range(p):
                        A_sys = np.zeros((n, r), dtype=float)
                        for i in range(n):
                            for c in range(r):
                                A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        
                        # Check if system is solvable
                        # A_sys @ V_col = ones(n)
                        try:
                            # Use least squares
                            V_col, residuals, rank_a, sv = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)
                        except:
                            valid = False
                            break
                        
                        # Check if V_col has half-integer entries
                        if not _is_half_integer(V_col):
                            valid = False
                            break
                        
                        V_sol[j*p+k, :] = V_col
                    
                    if not valid:
                        break
                
                if valid:
                    # Verify
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for c in range(r):
                                    val += U[i*m+j, c] * V_sol[j*p+k, c] * W[k*n+i, c]
                                if not np.isclose(val, 1, atol=1e-6):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        return (U.tolist(), V_sol.tolist(), W.tolist())
    
    return None


def _try_structured(n, m, p, r):
    """Try structured decompositions"""
    
    # Schoolbook pattern
    if r == n * m * p:
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return (U.tolist(), V.tolist(), W.tolist())
    
    # Specific patterns for M_{2,3,2} at rank 6
    if n == 2 and m == 3 and p == 2 and r == 6:
        # Try several structured patterns
        patterns = []
        
        # Pattern 1: Alternating signs
        U1 = np.zeros((6, 6))
        W1 = np.zeros((4, 6))
        for c in range(6):
            for i in range(2):
                for j in range(3):
                    U1[i*3+j, c] = 1 if (i + j + c) % 2 == 0 else -1
            for k in range(2):
                for i in range(2):
                    W1[k*2+i, c] = 1 if (k + i + c) % 2 == 0 else -1
        patterns.append((U1, W1))
        
        # Pattern 2: Different phase
        U2 = np.zeros((6, 6))
        W2 = np.zeros((4, 6))
        for c in range(6):
            for i in range(2):
                for j in range(3):
                    U2[i*3+j, c] = 1 if (j + c) % 2 == 0 else -1
            for k in range(2):
                for i in range(2):
                    W2[k*2+i, c] = 1 if (k + c) % 2 == 0 else -1
        patterns.append((U2, W2))
        
        # Pattern 3: Identity-like
        U3 = np.zeros((6, 6))
        W3 = np.zeros((4, 6))
        for c in range(6):
            for i in range(2):
                for j in range(3):
                    U3[i*3+j, c] = 1 if (j == c % 3) else 0
            for k in range(2):
                for i in range(2):
                    W3[k*2+i, c] = 1 if (i == c % 2) else 0
        patterns.append((U3, W3))
        
        # Pattern 4: Random but structured
        U4 = np.zeros((6, 6))
        W4 = np.zeros((4, 6))
        for c in range(6):
            for i in range(2):
                for j in range(3):
                    U4[i*3+j, c] = 1 if (i*j + c) % 3 == 0 else (-1 if (i*j + c) % 3 == 1 else 0)
            for k in range(2):
                for i in range(2):
                    W4[k*2+i, c] = 1 if (k*i + c) % 2 == 0 else -1
        patterns.append((U4, W4))
        
        # Pattern 5: Use specific known good structure
        # Try to make each (j,k) system have a simple solution
        U5 = np.zeros((6, 6))
        W5 = np.zeros((4, 6))
        # For each column c, assign a (j,k) "owner"
        for c in range(6):
            j_owner = c % 3
            k_owner = (c // 3) % 2
            for i in range(2):
                U5[i*3+j_owner, c] = 1
                W5[k_owner*2+i, c] = 1
        patterns.append((U5, W5))
        
        for U_pat, W_pat in patterns:
            V_sol = np.zeros((m*p, r), dtype=float)
            valid = True
            for j in range(m):
                for k in range(p):
                    A_sys = np.zeros((n, r), dtype=float)
                    for i in range(n):
                        for c in range(r):
                            A_sys[i, c] = U_pat[i*m+j, c] * W_pat[k*n+i, c]
                    
                    try:
                        V_col, _, _, _ = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)
                    except:
                        valid = False
                        break
                    
                    if not _is_half_integer(V_col):
                        valid = False
                        break
                    
                    V_sol[j*p+k, :] = V_col
                
                if not valid:
                    break
            
            if valid:
                # Verify
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_pat[i*m+j, c] * V_sol[j*p+k, c] * W_pat[k*n+i, c] for c in range(r))
                            if not np.isclose(val, 1, atol=1e-6):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    return (U_pat.tolist(), V_sol.tolist(), W_pat.tolist())
    
    return None
