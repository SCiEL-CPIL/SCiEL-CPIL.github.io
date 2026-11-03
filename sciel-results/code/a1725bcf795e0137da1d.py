"""
Combined: Winograd/Strassen block decomposition for 4x5x6 matrix multiplication.
Strategy:
  1. Build a 4x4x4 Strassen (rank 49) + 4x4x2 Strassen (rank 28) + 4x1x6 schoolbook (rank 24)
     -> total rank 101.
  2. Also try the 4x5x6 Winograd split (4x4@4x6 + 4x1@1x6) -> rank 108.
  3. Refine with iterative column-removal (single and pairwise) and a
     gradient-based local search that tries to merge/eliminate columns.
  4. Final polish: try to find exact integer/half-integer rounding.
"""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-4):
    """Fast vectorized check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _build_strassen_2x2():
    """2x2 Strassen decomposition. Returns (U,V,W) each shape (4,7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return np.array(U,dtype=float), np.array(V,dtype=float), np.array(W,dtype=float)

def _build_strassen_4x4():
    """4x4 Strassen decomposition, rank 49. U,V,W each (16,49)."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 49
    U = np.zeros((16, rank)); V = np.zeros((16, rank)); W = np.zeros((16, rank))
    strassen_defs = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s*7 + t
            for li in range(2):
                for lj in range(2):
                    lar = li*2+lj
                    ca = U2[lar][t]
                    if ca != 0:
                        for (bi,bj),sig in a_blocks:
                            gi = bi*2+li; gj = bj*2+lj
                            U[gi*4+gj, col] += ca*sig
            for lj in range(2):
                for lk in range(2):
                    lbr = lj*2+lk
                    cb = V2[lbr][t]
                    if cb != 0:
                        for (bj,bk),sig in b_blocks:
                            gj = bj*2+lj; gk = bk*2+lk
                            V[gj*4+gk, col] += cb*sig
            for lk in range(2):
                for li in range(2):
                    lcr = lk*2+li
                    cc = W2[lcr][t]
                    if cc != 0:
                        for (bi,bk),sig in c_blocks:
                            gk = bk*2+lk; gi = bi*2+li
                            W[gk*4+gi, col] += cc*sig
    return U, V, W

def _build_strassen_4x4x2():
    """4x4x2 decomposition, rank 28. U(16,28), V(8,28), W(8,28)."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = np.zeros((16, rank)); V = np.zeros((8, rank)); W = np.zeros((8, rank))
    def add_block(ai, aj, bj, ci, ck, base):
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    ca = U2[li*2+lj][t]
                    if ca: U[(ai+li)*4+(aj+lj), col] += ca
            for lj in range(2):
                for lk in range(2):
                    cb = V2[lj*2+lk][t]
                    if cb: V[(bj+lj)*2+(ck+lk), col] += cb
            for lk in range(2):
                for li in range(2):
                    cc = W2[lk*2+li][t]
                    if cc: W[(ck+lk)*4+(ci+li), col] += cc
    add_block(0,0,0,0,0,0)
    add_block(0,2,2,0,0,7)
    add_block(2,0,0,2,0,14)
    add_block(2,2,2,2,0,21)
    return U, V, W

def _build_strassen_4x4x4():
    """4x4x4 decomposition, rank 49. U(16,49), V(16,49), W(16,49)."""
    return _build_strassen_4x4()

def _build_4x4x4_plus_remainder():
    """
    Split 4x5x6 as:
      A (4x5) = [A1 (4x4), A2 (4x1)]
      B (5x6) = [B1 (4x6); B2 (1x6)]
      C = A1@B1 + A2@B2
    For A1@B1 (4x4 @ 4x6):
      Split B1 into B1a (4x4) and B1b (4x2)
      A1@B1a: 4x4x4 Strassen, rank 49
      A1@B1b: 4x4x2 Strassen, rank 28
    For A2@B2 (4x1 @ 1x6): schoolbook, rank 24
    Total: 49 + 28 + 24 = 101
    """
    n, m, p = 4, 5, 6
    U444, V444, W444 = _build_strassen_4x4x4()
    U442, V442, W442 = _build_strassen_4x4x2()
    
    total_rank = 49 + 28 + 24
    U = np.zeros((n*m, total_rank))  # 20
    V = np.zeros((m*p, total_rank))  # 30
    W = np.zeros((p*n, total_rank))  # 24
    
    # A1@B1a: 4x4 @ 4x4 -> C1a (4x4)
    # U: rows i*5+j for i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            U[i*5+j, 0:49] = U444[i*4+j, :]
    # V: rows j*6+k for j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            V[j*6+k, 0:49] = V444[j*4+k, :]
    # W: rows k*4+i for k in 0..3, i in 0..3
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W444[k*4+i, :]
    
    # A1@B1b: 4x4 @ 4x2 -> C1b (4x2)
    base = 49
    for i in range(4):
        for j in range(4):
            U[i*5+j, base:base+28] = U442[i*4+j, :]
    for j in range(4):
        for k in range(4, 6):
            V[j*6+k, base:base+28] = V442[j*2+(k-4), :]
    for k in range(4, 6):
        for i in range(4):
            W[k*4+i, base:base+28] = W442[(k-4)*4+i, :]
    
    # A2@B2: 4x1 @ 1x6 -> C2 (4x6)
    base2 = 77
    for i in range(4):
        for k in range(6):
            col = base2 + i*6 + k
            U[i*5+4, col] = 1.0
            V[4*6+k, col] = 1.0
            W[k*4+i, col] = 1.0
    
    return U, V, W

def _build_4x5x6_winograd():
    """
    Alternative: 4x4@4x6 + 4x1@1x6
    4x4@4x6: three 4x4x2 Strassen blocks, rank 3*28=84
    4x1@1x6: schoolbook, rank 24
    Total: 108
    """
    n, m, p = 4, 5, 6
    U442, V442, W442 = _build_strassen_4x4x2()
    total_rank = 84 + 24
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for blk in range(3):
        base = blk*28
        k_start = blk*2
        for i in range(4):
            for j in range(4):
                U[i*5+j, base:base+28] = U442[i*4+j, :]
        for j in range(4):
            for k in range(6):
                if k_start <= k < k_start+2:
                    V[j*6+k, base:base+28] = V442[j*2+(k-k_start), :]
        for k in range(6):
            if k_start <= k < k_start+2:
                for i in range(4):
                    W[k*4+i, base:base+28] = W442[(k-k_start)*4+i, :]
    
    base = 84
    for i in range(4):
        for k in range(6):
            col = base + i*6 + k
            U[i*5+4, col] = 1.0
            V[4*6+k, col] = 1.0
            W[k*4+i, col] = 1.0
    return U, V, W

def _local_reduce(U, V, W, n, m, p, max_iters=30, budget_s=20.0):
    """Iteratively remove columns that are redundant."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    R = U.shape[1]
    start = time.time()
    for _ in range(max_iters):
        if _time_left() < 3.0 or (time.time() - start) > budget_s:
            break
        if R < 2:
            break
        found = False
        for c in range(R):
            if _time_left() < 2.0:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    R -= 1
                    found = True
                    break
        if not found:
            break
    return U, V, W

def _pairwise_reduce(U, V, W, n, m, p, max_iters=5, budget_s=15.0):
    """Try removing pairs of columns."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    R = U.shape[1]
    start = time.time()
    for _ in range(max_iters):
        if _time_left() < 3.0 or (time.time() - start) > budget_s:
            break
        if R < 3:
            break
        found = False
        for c1 in range(R):
            if _time_left() < 2.0:
                break
            for c2 in range(c1+1, R):
                if _time_left() < 2.0:
                    break
                mask = np.ones(R, dtype=bool)
                mask[c1] = False; mask[c2] = False
                U2 = U[:, mask]; V2 = V[:, mask]; W2 = W[:, mask]
                if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                    if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                        U, V, W = U2, V2, W2
                        R -= 2
                        found = True
                        break
            if found:
                break
        if not found:
            break
    return U, V, W

def _iterative_refinement(n, m, p, initial_rank, max_iters=80, seed=42):
    """
    Iterative refinement: start with random init, alternate solving for each factor.
    """
    if _time_left() < 10.0:
        return None
    R = initial_rank
    np.random.seed(seed)
    U = np.random.randn(n*m, R) * 0.1
    V = np.random.randn(m*p, R) * 0.1
    W = np.random.randn(p*n, R) * 0.1
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        # Solve for W
        A = np.zeros((p, n, R))
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    A[k, i, :] += U[i*m+j, :] * V[j*p+k, :]
        W_new = np.zeros((p*n, R))
        valid = True
        for k in range(p):
            for i in range(n):
                a = A[k, i, :]
                if np.linalg.norm(a) < 1e-10:
                    valid = False; break
                W_new[k*n+i, :] = a / (a @ a)
            if not valid: break
        if not valid: break
        W = W_new
        
        # Solve for U
        B = np.zeros((n, m, R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    B[i, j, :] += V[j*p+k, :] * W[k*n+i, :]
        U_new = np.zeros((n*m, R))
        valid = True
        for i in range(n):
            for j in range(m):
                b = B[i, j, :]
                if np.linalg.norm(b) < 1e-10:
                    valid = False; break
                U_new[i*m+j, :] = b / (b @ b)
            if not valid: break
        if not valid: break
        U = U_new
        
        # Solve for V
        C = np.zeros((m, p, R))
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    C[j, k, :] += U[i*m+j, :] * W[k*n+i, :]
        V_new = np.zeros((m*p, R))
        valid = True
        for j in range(m):
            for k in range(p):
                c = C[j, k, :]
                if np.linalg.norm(c) < 1e-10:
                    valid = False; break
                V_new[j*p+k, :] = c / (c @ c)
            if not valid: break
        if not valid: break
        V = V_new
        
        if it % 10 == 0:
            if _verify_np(U, V, W, n, m, p, 1e-6):
                if _verify_decomp(U, V, W, n, m, p, 1e-8):
                    return U, V, W
    if _verify_np(U, V, W, n, m, p, 1e-6):
        if _verify_decomp(U, V, W, n, m, p, 1e-8):
            return U, V, W
    return None

def _gradient_descent_reduce(U, V, W, n, m, p, budget_s=15.0):
    """
    Try to reduce rank by finding linear combinations of columns that
    can be eliminated. Uses a smooth loss: sum over (i,j,k) of (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2.
    We try to set some columns to zero by gradient descent on a penalty.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    R = U.shape[1]
    start = time.time()
    
    # Try to zero out the last column
    for target_col in range(R-1, max(R-6, -1), -1):
        if _time_left() < 3.0 or (time.time() - start) > budget_s:
            break
        lr = 0.01
        for it in range(200):
            if _time_left() < 2.0 or (time.time() - start) > budget_s:
                break
            # Compute gradient of loss w.r.t. U, V, W
            # Loss = sum_{i,j,k} (sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] - 1)^2
            # dL/dU[i*m+j, r] = 2 * sum_{k} (residual_{i,j,k} * V[j*p+k, r] * W[k*n+i, r])
            
            # Compute residuals
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            resid = T - 1.0  # (n, m, p)
            
            # Gradient for U
            dU = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        dU[i*m+j, :] += 2 * resid[i, j, k] * V[j*p+k, :] * W[k*n+i, :]
            
            # Gradient for V
            dV = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        dV[j*p+k, :] += 2 * resid[i, j, k] * U[i*m+j, :] * W[k*n+i, :]
            
            # Gradient for W
            dW = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        dW[k*n+i, :] += 2 * resid[i, j, k] * U[i*m+j, :] * V[j*p+k, :]
            
            # Apply gradient with penalty to zero out target_col
            penalty = 10.0
            dU[:, target_col] += penalty * U[:, target_col]
            dV[:, target_col] += penalty * V[:, target_col]
            dW[:, target_col] += penalty * W[:, target_col]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if np.linalg.norm(U[:, target_col]) < 1e-6 and \
               np.linalg.norm(V[:, target_col]) < 1e-6 and \
               np.linalg.norm(W[:, target_col]) < 1e-6:
                break
        
        # Check if we can remove the column
        if np.linalg.norm(U[:, target_col]) < 1e-4 and \
           np.linalg.norm(V[:, target_col]) < 1e-4 and \
           np.linalg.norm(W[:, target_col]) < 1e-4:
            mask = np.ones(R, dtype=bool)
            mask[target_col] = False
            U2 = U[:, mask]; V2 = V[:, mask]; W2 = W[:, mask]
            if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    R -= 1
    
    return U, V, W

def _round_to_half(U, V, W, tol=1e-6):
    """Try to round entries to nearest half-integer and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if _verify_np(U_r, V_r, W_r, 4, 5, 6, 1e-5):
        if _verify_decomp(U_r, V_r, W_r, 4, 5, 6, 1e-8):
            return U_r, V_r, W_r
    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition, rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Build candidate decompositions
    candidates = []
    
    # 1. 4x4x4 Strassen + 4x4x2 Strassen + 4x1x6 schoolbook = rank 101
    try:
        U1, V1, W1 = _build_4x4x4_plus_remainder()
        if _verify_decomp(U1, V1, W1, n, m, p, 1e-9):
            candidates.append((U1, V1, W1))
    except Exception:
        pass
    
    # 2. Winograd 4x4@4x6 + 4x1@1x6 = rank 108
    try:
        U2, V2, W2 = _build_4x5x6_winograd()
        if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
            candidates.append((U2, V2, W2))
    except Exception:
        pass
    
    # 3. Schoolbook = rank 120 (fallback)
    U3, V3, W3 = _build_schoolbook(n, m, p)
    candidates.append((U3, V3, W3))
    
    # Pick the best (lowest rank)
    candidates.sort(key=lambda x: x[0].shape[1])
    best_U, best_V, best_W = candidates[0]
    best_rank = best_U.shape[1]
    
    # Refine with local column removal
    try:
        if _time_left() > 15.0:
            for U_c, V_c, W_c in candidates[:2]:
                if _time_left() < 10.0:
                    break
                U_r, V_r, W_r = _local_reduce(U_c, V_c, W_c, n, m, p,
                                              max_iters=30, budget_s=min(15.0, _time_left()-5.0))
                if _verify_decomp(U_r, V_r, W_r, n, m, p, 1e-9):
                    if U_r.shape[1] < best_rank:
                        best_rank = U_r.shape[1]
                        best_U, best_V, best_W = U_r, V_r, W_r
    except Exception:
        pass
    
    # Pairwise reduction
    try:
        if _time_left() > 10.0:
            U_p, V_p, W_p = _pairwise_reduce(best_U, best_V, best_W, n, m, p,
                                             max_iters=5, budget_s=min(10.0, _time_left()-5.0))
            if _verify_decomp(U_p, V_p, W_p, n, m, p, 1e-9):
                if U_p.shape[1] < best_rank:
                    best_rank = U_p.shape[1]
                    best_U, best_V, best_W = U_p, V_p, W_p
    except Exception:
        pass
    
    # Gradient-based reduction
    try:
        if _time_left() > 10.0:
            U_g, V_g, W_g = _gradient_descent_reduce(best_U, best_V, best_W, n, m, p,
                                                     budget_s=min(12.0, _time_left()-5.0))
            if _verify_decomp(U_g, V_g, W_g, n, m, p, 1e-9):
                if U_g.shape[1] < best_rank:
                    best_rank = U_g.shape[1]
                    best_U, best_V, best_W = U_g, V_g, W_g
    except Exception:
        pass
    
    # Iterative refinement to find lower-rank decomposition
    try:
        if _time_left() > 20.0:
            for R in range(max(60, best_rank - 15), best_rank):
                if _time_left() < 10.0:
                    break
                for seed in [42, 123, 777]:
                    if _time_left() < 5.0:
                        break
                    cand = _iterative_refinement(n, m, p, initial_rank=R, max_iters=60, seed=seed)
                    if cand is not None:
                        U_c, V_c, W_c = cand
                        if _verify_decomp(U_c, V_c, W_c, n, m, p, 1e-9):
                            if U_c.shape[1] < best_rank:
                                best_rank = U_c.shape[1]
                                best_U, best_V, best_W = U_c, V_c, W_c
                                break
    except Exception:
        pass
    
    # Final local reduction on the best
    try:
        if _time_left() > 5.0:
            U_f, V_f, W_f = _local_reduce(best_U, best_V, best_W, n, m, p,
                                          max_iters=20, budget_s=min(8.0, _time_left()-3.0))
            if _verify_decomp(U_f, V_f, W_f, n, m, p, 1e-9):
                if U_f.shape[1] < best_rank:
                    best_rank = U_f.shape[1]
                    best_U, best_V, best_W = U_f, V_f, W_f
    except Exception:
        pass
    
    # Round to half-integers
    try:
        best_U, best_V, best_W = _round_to_half(best_U, best_V, best_W)
    except Exception:
        pass
    
    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
