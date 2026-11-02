"""Matrix multiplication 3,3,5 rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
_START = time.time()
DEADLINE = 150.0
_SAFETY = 5.0  # seconds margin before kill

def _time_left():
    return DEADLINE - _SAFETY - (time.time() - _START)

def _target_tensor(n, m, p):
    """T[i*m+j, j*p+k, k*n+i] = 1 else 0."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j, j * p + k, k * n + i)] = 1
    return T

def _verify(U, V, W, T):
    """Exact integer verification that sum_r U[:,r]⊗V[:,r]⊗W[:,r] == T."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # build candidate tensor dict
    cand = {}
    for r in range(rank):
        uc = U[:, r] if isinstance(U, np.ndarray) else U
        vc = V[:, r] if isinstance(V, np.ndarray) else V
        wc = W[:, r] if isinstance(W, np.ndarray) else W
        for a in range(n * m):
            uav = uc[a]
            if uav == 0:
                continue
            for b in range(m * p):
                ubv = uav * vc[b]
                if ubv == 0:
                    continue
                for c in range(p * n):
                    wcv = ubv * wc[c]
                    if wcv == 0:
                        continue
                    cand[(a, b, c)] = cand.get((a, b, c), 0) + wcv
    # compare
    if set(cand.keys()) != set(T.keys()):
        return False
    for key, val in T.items():
        if cand[key] != val:
            return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_222():
    """Rank-7 Strassen factors for 2x2x2, indexed i*2+j, j*2+k, k*2+i."""
    Ufac = []
    Vfac = []
    Wfac = []
    
    # M1: (A00+A11)(B00+B11)
    Ufac.append({0: 1, 3: 1})
    Vfac.append({0: 1, 3: 1})
    Wfac.append({0: 1, 3: 1})  # C00, C11
    
    # M2: (A10+A11)B00
    Ufac.append({2: 1, 3: 1})
    Vfac.append({0: 1})
    Wfac.append({2: 1, 3: -1})  # C10, -C11
    
    # M3: A00(B01-B11)
    Ufac.append({0: 1})
    Vfac.append({1: 1, 3: -1})
    Wfac.append({1: 1, 3: 1})  # C01, C11
    
    # M4: A11(B10-B00)
    Ufac.append({3: 1})
    Vfac.append({2: 1, 0: -1})
    Wfac.append({0: 1, 2: 1})  # C00, C10
    
    # M5: (A00+A01)B11
    Ufac.append({0: 1, 1: 1})
    Vfac.append({3: 1})
    Wfac.append({0: -1, 1: 1})  # -C00, C01
    
    # M6: (A10-A00)(B00+B01)
    Ufac.append({2: 1, 0: -1})
    Vfac.append({0: 1, 1: 1})
    Wfac.append({3: 1})  # C11
    
    # M7: (A01-A11)(B10+B11)
    Ufac.append({1: 1, 3: -1})
    Vfac.append({2: 1, 3: 1})
    Wfac.append({0: 1})  # C00
    
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    for r in range(7):
        for idx, val in Ufac[r].items():
            U[idx][r] += val
        for idx, val in Vfac[r].items():
            V[idx][r] += val
        for idx, val in Wfac[r].items():
            W[idx][r] += val
            
    return U, V, W

def _compress_factors(U, V, W):
    """Remove zero columns from factor matrices."""
    n_U = len(U)
    n_V = len(V)
    n_W = len(W)
    if n_U == 0:
        return [], [], []
    n_rank = len(U[0])
    
    kept = []
    for c in range(n_rank):
        is_zero = True
        if any(U[a][c] != 0 for a in range(n_U)):
            is_zero = False
        if any(V[b][c] != 0 for b in range(n_V)):
            is_zero = False
        if any(W[k][c] != 0 for k in range(n_W)):
            is_zero = False
        if not is_zero:
            kept.append(c)
            
    new_U = [[U[a][c] for c in kept] for a in range(n_U)]
    new_V = [[V[b][c] for c in kept] for b in range(n_V)]
    new_W = [[W[k][c] for c in kept] for k in range(n_W)]
    return new_U, new_V, new_W

def _block_decomposition(n, m, p):
    """
    Partition 3x3x5 into blocks and apply Strassen where possible.
    Partition:
    - i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2 Strassen (rank 7)
    - i in {0,1}, j in {2},   k in {0,1}: 2x1x2 schoolbook (rank 4)
    - i in {2},   j in {0,1}, k in {0,1}: 1x2x2 schoolbook (rank 4)
    - i in {2},   j in {2},   k in {0,1}: 1x1x2 schoolbook (rank 2)
    - i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2 Strassen (rank 7)
    - i in {0,1}, j in {2},   k in {2,3}: 2x1x2 schoolbook (rank 4)
    - i in {2},   j in {0,1}, k in {2,3}: 1x2x2 schoolbook (rank 4)
    - i in {2},   j in {2},   k in {2,3}: 1x1x2 schoolbook (rank 2)
    - i in {0,1,2}, j in {0,1,2}, k in {4}: 3x3x1 schoolbook (rank 9)
    
    Total: 7+4+4+2+7+4+4+2+9 = 43
    """
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    
    col_offset = 0
    
    def add_block(block_U, block_V, block_W, i_off, j_off, k_off):
        nonlocal col_offset
        bi = len(block_U)
        bj = len(block_V)
        bk = len(block_W)
        br = len(block_U[0]) if bi > 0 else 0
        
        for r in range(br):
            for a in range(bi):
                if block_U[a][r] != 0:
                    U[i_off * m + j_off + a][col_offset + r] = block_U[a][r]
            for b in range(bj):
                if block_V[b][r] != 0:
                    V[j_off * p + k_off + b][col_offset + r] = block_V[b][r]
            for c in range(bk):
                if block_W[c][r] != 0:
                    W[k_off * n + i_off + c][col_offset + r] = block_W[c][r]
        
        col_offset += br
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2 Strassen
    sU, sV, sW = _strassen_222()
    add_block(sU, sV, sW, 0, 0, 0)
    
    # Block 2: i in {0,1}, j in {2}, k in {0,1}: 2x1x2 schoolbook
    for i in range(2):
        for k in range(2):
            add_block([[1]], [[1]], [[1]], i, 2, k)
    
    # Block 3: i in {2}, j in {0,1}, k in {0,1}: 1x2x2 schoolbook
    for j in range(2):
        for k in range(2):
            add_block([[1]], [[1]], [[1]], 2, j, k)
    
    # Block 4: i in {2}, j in {2}, k in {0,1}: 1x1x2 schoolbook
    for k in range(2):
        add_block([[1]], [[1]], [[1]], 2, 2, k)
    
    # Block 5: i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2 Strassen
    sU2, sV2, sW2 = _strassen_222()
    add_block(sU2, sV2, sW2, 0, 0, 2)
    
    # Block 6: i in {0,1}, j in {2}, k in {2,3}: 2x1x2 schoolbook
    for i in range(2):
        for k in range(2):
            add_block([[1]], [[1]], [[1]], i, 2, k + 2)
    
    # Block 7: i in {2}, j in {0,1}, k in {2,3}: 1x2x2 schoolbook
    for j in range(2):
        for k in range(2):
            add_block([[1]], [[1]], [[1]], 2, j, k + 2)
    
    # Block 8: i in {2}, j in {2}, k in {2,3}: 1x1x2 schoolbook
    for k in range(2):
        add_block([[1]], [[1]], [[1]], 2, 2, k + 2)
    
    # Block 9: i in {0,1,2}, j in {0,1,2}, k in {4}: 3x3x1 schoolbook
    for i in range(3):
        for j in range(3):
            add_block([[1]], [[1]], [[1]], i, j, 4)
    
    new_U, new_V, new_W = _compress_factors(U, V, W)
    return {"U": new_U, "V": new_V, "W": new_W}

def _try_als_completion(n, m, p, base_rank, max_iter=100, tol=1e-10):
    """
    Try to find a lower-rank decomposition using ALS-style optimization.
    Start from a schoolbook or block decomposition and try to merge terms.
    This is a heuristic and may not find improvements.
    """
    if _time_left() < 10.0:
        return None
    
    # Start with block decomposition
    base = _block_decomposition(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    target_rank = base_rank - 1  # Try to reduce by 1
    
    if target_rank < 1:
        return None
    
    # Initialize with random projection
    np.random.seed(42)
    U = U0[:, :target_rank] + np.random.randn(n*m, target_rank) * 0.1
    V = V0[:, :target_rank] + np.random.randn(m*p, target_rank) * 0.1
    W = W0[:, :target_rank] + np.random.randn(p*n, target_rank) * 0.1
    
    # ALS iterations
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Update U
        for a in range(n*m):
            # Compute gradient for U[a]
            grad = np.zeros(target_rank)
            for r in range(target_rank):
                for b in range(m*p):
                    for c in range(p*n):
                        # T[a,b,c] * V[b,r] * W[c,r]
                        if (a, b, c) in _target_tensor(n, m, p):
                            grad[r] += V[b, r] * W[c, r]
                        # Subtract current contribution
                        grad[r] -= U[a, r] * V[b, r] * W[c, r]
            U[a, :] -= 0.01 * grad
        
        # Update V
        for b in range(m*p):
            grad = np.zeros(target_rank)
            for r in range(target_rank):
                for a in range(n*m):
                    for c in range(p*n):
                        if (a, b, c) in _target_tensor(n, m, p):
                            grad[r] += U[a, r] * W[c, r]
                        grad[r] -= V[b, r] * U[a, r] * W[c, r]
            V[b, :] -= 0.01 * grad
        
        # Update W
        for c in range(p*n):
            grad = np.zeros(target_rank)
            for r in range(target_rank):
                for a in range(n*m):
                    for b in range(m*p):
                        if (a, b, c) in _target_tensor(n, m, p):
                            grad[r] += U[a, r] * V[b, r]
                        grad[r] -= W[c, r] * U[a, r] * V[b, r]
            W[c, :] -= 0.01 * grad
    
    # Round to half-integers
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Convert to integer lists
    U_list = [[int(x) for x in row] for row in U_rounded]
    V_list = [[int(x) for x in row] for row in V_rounded]
    W_list = [[int(x) for x in row] for row in W_rounded]
    
    # Verify
    T = _target_tensor(n, m, p)
    if _verify(U_list, V_list, W_list, T):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _target_tensor(n, m, p)
    candidates = []
    
    # 1. Schoolbook (Rank 45)
    candidates.append(_schoolbook(n, m, p))
    
    # 2. Block decomposition with Strassen (Rank 43)
    try:
        if _time_left() > 2.0:
            candidates.append(_block_decomposition(n, m, p))
    except Exception:
        pass
    
    # 3. Try ALS completion to find rank 42 or lower
    try:
        if _time_left() > 30.0:
            als_result = _try_als_completion(n, m, p, 43)
            if als_result is not None:
                candidates.append(als_result)
    except Exception:
        pass
    
    # Verify candidates and pick best
    best = None
    best_rank = None
    for cand in candidates:
        if _time_left() <= 0.0:
            break
        r = len(cand["U"][0]) if cand["U"] else 0
        if best_rank is not None and r >= best_rank:
            continue
        try:
            if _verify(cand["U"], cand["V"], cand["W"], T):
                best = cand
                best_rank = r
        except Exception:
            continue
            
    if best is None:
        # fallback schoolbook (always valid)
        best = _schoolbook(n, m, p)
        
    return best
