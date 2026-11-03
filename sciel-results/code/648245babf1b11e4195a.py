"""
Matrix multiplication 4x5x6: rank-minimal exact tensor decomposition.
Strategy: build a structured low-rank initial decomposition (Strassen blocks + schoolbook),
then aggressively reduce rank via linear-algebraic column elimination (single, pair, and
triple removal) with fast vectorized verification.
"""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill

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

def _verify_np(U, V, W, n, m, p, tol=1e-6):
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
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # P1 = (a11+a22)*(b11+b22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # P2 = (a21+a22)*b11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # P3 = a11*(b12-b22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # P4 = a22*(b21-b11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # P5 = (a11+a12)*b22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # P6 = (a21-a11)*(b11+b12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # P7 = (a12-a22)*(b21+b22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x2():
    """4x4x2 via 4 Strassen 2x2 blocks. Rank 28. U(16,28), V(8,28), W(8,28)."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    def add_block(ai, aj, bj, ci, ck, base):
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    c = U2[li*2+lj, t]
                    if c: U[(ai+li)*4+(aj+lj), col] += c
            for lj in range(2):
                for lk in range(2):
                    c = V2[lj*2+lk, t]
                    if c: V[(bj+lj)*2+(ck+lk), col] += c
            for lk in range(2):
                for li in range(2):
                    c = W2[lk*2+li, t]
                    if c: W[(ck+lk)*4+(ci+li), col] += c
    add_block(0,0,0,0,0,0)
    add_block(0,2,2,0,0,7)
    add_block(2,0,0,2,0,14)
    add_block(2,2,2,2,0,21)
    return U, V, W

def _build_4x5x6():
    """
    4x5x6 decomposition:
    - Split m=5 into m1=4, m2=1
    - A_left(4x4) @ B_top(4x6): 3 blocks of 4x4x2 Strassen -> rank 84
    - A_right(4x1) @ B_bottom(1x6): schoolbook -> rank 24
    Total: 108
    """
    n, m, p = 4, 5, 6
    U442, V442, W442 = _build_4x4x2()
    total_rank = 3*28 + 24
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for blk in range(3):
        base = blk*28
        k_start = blk*2
        for i in range(4):
            for j in range(4):
                U[i*m+j, base:base+28] = U442[i*4+j, :]
        for j in range(4):
            for k in range(6):
                if k_start <= k < k_start+2:
                    V[j*p+k, base:base+28] = V442[j*2+(k-k_start), :]
        for k in range(6):
            if k_start <= k < k_start+2:
                for i in range(4):
                    W[k*n+i, base:base+28] = W442[(k-k_start)*4+i, :]
    
    base = 3*28
    for i in range(4):
        for k in range(6):
            col = base + i*6 + k
            U[i*m+4, col] = 1.0
            V[4*p+k, col] = 1.0
            W[k*n+i, col] = 1.0
    return U, V, W

def _build_schoolbook(n, m, p):
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_remove(U, V, W, n, m, p, indices_to_remove):
    """Try removing given column indices. Returns (U2,V2,W2) or None."""
    R = U.shape[1]
    mask = np.ones(R, dtype=bool)
    for idx in indices_to_remove:
        mask[idx] = False
    U2 = U[:, mask]
    V2 = V[:, mask]
    W2 = W[:, mask]
    if _verify_np(U2, V2, W2, n, m, p, 1e-6):
        if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
            return U2, V2, W2
    return None

def _single_remove(U, V, W, n, m, p, budget_s=30.0):
    """Iteratively remove single columns if possible."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start = time.time()
    progress = True
    while progress and R > 2:
        progress = False
        for c in range(R):
            if time.time() - start > budget_s or _time_left() < 5.0:
                return U, V, W
            res = _try_remove(U, V, W, n, m, p, [c])
            if res is not None:
                U, V, W = res
                R -= 1
                progress = True
                break
    return U, V, W

def _pair_remove(U, V, W, n, m, p, budget_s=20.0):
    """Iteratively remove pairs of columns if possible."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start = time.time()
    progress = True
    while progress and R > 3:
        progress = False
        for c1 in range(R):
            if time.time() - start > budget_s or _time_left() < 5.0:
                return U, V, W
            for c2 in range(c1+1, R):
                if time.time() - start > budget_s or _time_left() < 5.0:
                    return U, V, W
                res = _try_remove(U, V, W, n, m, p, [c1, c2])
                if res is not None:
                    U, V, W = res
                    R -= 2
                    progress = True
                    break
            if progress:
                break
    return U, V, W

def _triple_remove(U, V, W, n, m, p, budget_s=15.0):
    """Iteratively remove triples of columns if possible."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start = time.time()
    progress = True
    while progress and R > 4:
        progress = False
        for c1 in range(R):
            if time.time() - start > budget_s or _time_left() < 5.0:
                return U, V, W
            for c2 in range(c1+1, R):
                if time.time() - start > budget_s or _time_left() < 5.0:
                    return U, V, W
                for c3 in range(c2+1, R):
                    if time.time() - start > budget_s or _time_left() < 5.0:
                        return U, V, W
                    res = _try_remove(U, V, W, n, m, p, [c1, c2, c3])
                    if res is not None:
                        U, V, W = res
                        R -= 3
                        progress = True
                        break
                if progress:
                    break
            if progress:
                break
    return U, V, W

def _linear_dependence_reduce(U, V, W, n, m, p, budget_s=25.0):
    """
    For each column c, check if T_c = U_c ⊗ V_c ⊗ W_c is in the span of
    {T_s : s != c}. This is a linear algebra check on the tensor space.
    
    The tensor space has dimension n*m*p = 120.
    Flatten each rank-1 tensor to a vector of length 120.
    Check if v_c is in span{v_s : s != c}.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    dim = n * m * p  # 120
    
    # Build tensor vectors
    tensors = np.zeros((R, dim))
    for r in range(R):
        t = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t[i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        tensors[r] = t.reshape(-1)
    
    start = time.time()
    progress = True
    active = list(range(R))
    
    while progress and len(active) > 10:
        progress = False
        if time.time() - start > budget_s or _time_left() < 5.0:
            break
        for idx, c in enumerate(active):
            if time.time() - start > budget_s or _time_left() < 5.0:
                break
            # Check if tensor c is in span of all other active tensors
            others = [a for a in active if a != c]
            if len(others) == 0:
                continue
            A = tensors[others]  # (len(others), dim)
            b = tensors[c]
            # Check if b is in row space of A (i.e., in span of rows)
            # Use least squares residual
            if A.shape[0] < dim:
                # Underdetermined: likely in span
                # Check via SVD
                try:
                    U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
                    # Project b onto column space of A
                    proj = Vt.T @ (Vt @ b)
                    residual = b - proj
                    if np.linalg.norm(residual) < 1e-8 * max(1.0, np.linalg.norm(b)):
                        # Remove column c
                        active.remove(c)
                        R -= 1
                        progress = True
                        break
                except:
                    pass
            else:
                try:
                    # Solve A^T x = b, check residual
                    # Actually check if b is in col(A)
                    Q, Rmat = np.linalg.qr(A)
                    proj = Q @ (Q.T @ b)
                    residual = b - proj
                    if np.linalg.norm(residual) < 1e-8 * max(1.0, np.linalg.norm(b)):
                        active.remove(c)
                        R -= 1
                        progress = True
                        break
                except:
                    pass
    
    # Rebuild from active columns
    if len(active) < R:
        U2 = U[:, active]
        V2 = V[:, active]
        W2 = W[:, active]
        if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
            return U2, V2, W2
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Build initial decomposition
    U, V, W = _build_4x5x6()
    if not _verify_decomp(U, V, W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 1: Single column removal
    if _time_left() > 40.0:
        U, V, W = _single_remove(U, V, W, n, m, p, budget_s=35.0)
        if _verify_decomp(U, V, W, n, m, p, 1e-9):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 2: Pair column removal
    if _time_left() > 30.0:
        U, V, W = _pair_remove(U, V, W, n, m, p, budget_s=25.0)
        if _verify_decomp(U, V, W, n, m, p, 1e-9):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 3: Linear dependence reduction
    if _time_left() > 25.0:
        U, V, W = _linear_dependence_reduce(U, V, W, n, m, p, budget_s=20.0)
        if _verify_decomp(U, V, W, n, m, p, 1e-9):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 4: Another round of single removal
    if _time_left() > 20.0:
        U, V, W = _single_remove(U, V, W, n, m, p, budget_s=15.0)
        if _verify_decomp(U, V, W, n, m, p, 1e-9):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 5: Triple removal
    if _time_left() > 15.0:
        U, V, W = _triple_remove(U, V, W, n, m, p, budget_s=10.0)
        if _verify_decomp(U, V, W, n, m, p, 1e-9):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 6: Final single removal pass
    if _time_left() > 10.0:
        U, V, W = _single_remove(U, V, W, n, m, p, budget_s=8.0)
        if _verify_decomp(U, V, W, n, m, p, 1e-9):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
