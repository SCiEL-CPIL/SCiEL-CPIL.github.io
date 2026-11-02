"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses Strassen block decomposition as baseline, then ALS refinement with rounding."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # Compute efficiently
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        uc = Ur[:, :, r]  # (n, m)
        vc = Vr[:, :, r]  # (m, p)
        wc = Wr[:, :, r]  # (p, n)
        for j in range(m):
            T[:, j, :] += np.outer(uc[:, j], vc[j, :]) * wc.T
    return np.allclose(T, 1.0, atol=atol)

def _round_half(M):
    """Round matrix to nearest half-integer."""
    return (2.0 * np.round(2.0 * M)) / 4.0

def _compute_error(U, V, W, n, m, p):
    """Compute L2 error of decomposition."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ur[i, j, :], Vr[j, k, :] * Wr[k, i, :])
                err += (s - 1.0) ** 2
    return np.sqrt(err)

def _strassen_block(n, m, p):
    """Construct Strassen-based block decomposition for 4x4*4x5.
    Split into 2x2 blocks. For 2x2x2 use Strassen (7 terms).
    For 2x2x3 use Strassen on first 2 cols + naive on 3rd (7+4=11 terms).
    Total: 4*(7+7) + 4*(7+4) = 4*14 + 4*11 = 56 + 44 = 100... 
    Actually let me recount:
    C00(2x2) = A00*B00(2x2x2) + A01*B10(2x2x2) => 7+7=14
    C10(2x2) = A10*B00(2x2x2) + A11*B10(2x2x2) => 7+7=14
    C01(2x3) = A00*B01(2x2x3) + A01*B11(2x2x3) => 11+11=22
    C11(2x3) = A10*B01(2x2x3) + A11*B11(2x2x3) => 11+11=22
    Total = 14+14+22+22 = 72... that's worse.
    
    Let me use a different split. Actually for 2x2x3, we can do better.
    2x2x3: treat as 2x2x2 + 2x2x1. 2x2x2 rank 7, 2x2x1 rank 4. Total 11.
    But we can share: the 2x2x1 is just dot products, rank 4.
    
    Alternative: use 4x4x4 Strassen (rank 49) + handle the extra column.
    4x4x4 via Strassen: rank 7^2 = 49.
    4x4x1: rank 16 (naive) or can we do better?
    Total: 49 + 16 = 65.
    
    Actually let me just build it properly with 2x2 blocks.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def uid(gi, gj):
        return gi * m + gj
    
    def vid(gj, gk):
        return gj * p + gk
    
    def wid(gk, gi):
        return gk * n + gi
    
    def strassen_2x2_2(ai, aj, bj, bk):
        """7 Strassen terms for 2x2 * 2x2 -> 2x2."""
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); u[uid(ai,aj)]=1; u[uid(ai+1,aj+1)]=1
        v = np.zeros(m*p); v[vid(bj,bk)]=1; v[vid(bj+1,bk+1)]=1
        w = np.zeros(p*n); w[wid(bk,ai)]=1; w[wid(bk+1,ai+1)]=1
        add_term(u,v,w)
        # M2 = (c+d)e
        u = np.zeros(n*m); u[uid(ai+1,aj)]=1; u[uid(ai+1,aj+1)]=1
        v = np.zeros(m*p); v[vid(bj,bk)]=1
        w = np.zeros(p*n); w[wid(bk,ai+1)]=1; w[wid(bk+1,ai+1)]=-1
        add_term(u,v,w)
        # M3 = a(f-h)
        u = np.zeros(n*m); u[uid(ai,aj)]=1
        v = np.zeros(m*p); v[vid(bj,bk+1)]=1; v[vid(bj+1,bk+1)]=-1
        w = np.zeros(p*n); w[wid(bk+1,ai)]=1; w[wid(bk+1,ai+1)]=1
        add_term(u,v,w)
        # M4 = d(g-e)
        u = np.zeros(n*m); u[uid(ai+1,aj+1)]=1
        v = np.zeros(m*p); v[vid(bj+1,bk)]=1; v[vid(bj,bk)]=-1
        w = np.zeros(p*n); w[wid(bk,ai)]=1; w[wid(bk,ai+1)]=1
        add_term(u,v,w)
        # M5 = (a+b)h
        u = np.zeros(n*m); u[uid(ai,aj)]=1; u[uid(ai,aj+1)]=1
        v = np.zeros(m*p); v[vid(bj+1,bk+1)]=1
        w = np.zeros(p*n); w[wid(bk,ai)]=-1; w[wid(bk+1,ai)]=1
        add_term(u,v,w)
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[uid(ai+1,aj)]=1; u[uid(ai,aj)]=-1
        v = np.zeros(m*p); v[vid(bj,bk)]=1; v[vid(bj,bk+1)]=1
        w = np.zeros(p*n); w[wid(bk+1,ai+1)]=1
        add_term(u,v,w)
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[uid(ai,aj+1)]=1; u[uid(ai+1,aj+1)]=-1
        v = np.zeros(m*p); v[vid(bj+1,bk)]=1; v[vid(bj+1,bk+1)]=1
        w = np.zeros(p*n); w[wid(bk,ai)]=1
        add_term(u,v,w)
    
    def naive_2x2_1(ai, aj, bj, bk):
        """4 naive terms for 2x2 * 2x1 -> 2x1."""
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[uid(ai+di,aj+dj)]=1
                v = np.zeros(m*p); v[vid(bj+dj,bk)]=1
                w = np.zeros(p*n); w[wid(bk,ai+di)]=1
                add_term(u,v,w)
    
    def block_2x2_3(ai, aj, bj, bk):
        """11 terms: Strassen for 2x2x2 (cols bk,bk+1) + naive for 2x2x1 (col bk+2)."""
        strassen_2x2_2(ai, aj, bj, bk)
        naive_2x2_1(ai, aj, bj, bk+2)
    
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    strassen_2x2_2(0,0,0,0)
    strassen_2x2_2(0,2,2,0)
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    strassen_2x2_2(2,0,0,0)
    strassen_2x2_2(2,2,2,0)
    # C[0:2,2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:4]*B[2:4,2:5]
    block_2x2_3(0,0,0,2)
    block_2x2_3(0,2,2,2)
    # C[2:4,2:5] = A[2:4,0:2]*B[0:2,2:5] + A[2:4,2:4]*B[2:4,2:5]
    block_2x2_3(2,0,0,2)
    block_2x2_3(2,2,2,2)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _als_rounded(n, m, p, rank, U0, V0, W0, max_iters=300, deadline=None, tol=1e-10):
    """ALS with half-integer rounding. Returns (U,V,W) or None."""
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Optimize U
        for i in range(n):
            for j in range(m):
                # sum_r U[i*m+j,r] * (V[j*p+k,r]*W[k*n+i,r]) = 1 for all k
                A = np.array([V[j*p+k,:] * W[k*n+i,:] for k in range(p)])
                b = np.ones(p)
                try:
                    U[i*m+j,:] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                A = np.array([U[i*m+j,:] * W[k*n+i,:] for i in range(n)])
                b = np.ones(n)
                try:
                    V[j*p+k,:] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Optimize W
        for i in range(n):
            for k in range(p):
                A = np.array([U[i*m+j,:] * V[j*p+k,:] for j in range(m)])
                b = np.ones(m)
                try:
                    W[k*n+i,:] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        err = _compute_error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    # Round and verify
    Ur = _round_half(best_U)
    Vr = _round_half(best_V)
    Wr = _round_half(best_W)
    if _verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p, atol=1e-9):
        return Ur, Vr, Wr
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 140
    n, m, p = DIMENSIONS
    
    # Build Strassen baseline
    U_base, V_base, W_base = _strassen_block(n, m, p)
    base_rank = U_base.shape[1]
    
    # Verify
    if not _verify(U_base.tolist(), V_base.tolist(), W_base.tolist(), n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = base_rank
    best_U, best_V, best_W = U_base, V_base, W_base
    
    # Try to reduce rank via ALS
    # First try: keep all columns but optimize, then round
    if time.time() < deadline - 30:
        result = _als_rounded(n, m, p, base_rank, U_base, V_base, W_base, 
                              max_iters=200, deadline=deadline - 30)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = base_rank
    
    # Try progressively lower ranks
    for target_rank in range(best_rank - 1, 45, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try a few different initializations
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            if attempt == 0:
                # Use first target_rank columns of Strassen
                U_init = U_base[:, :target_rank]
                V_init = V_base[:, :target_rank]
                W_init = W_base[:, :target_rank]
            elif attempt == 1:
                # Random perturbation of Strassen columns
                U_init = U_base[:, :target_rank] + np.random.randn(n*m, target_rank) * 0.01
                V_init = V_base[:, :target_rank] + np.random.randn(m*p, target_rank) * 0.01
                W_init = W_base[:, :target_rank] + np.random.randn(p*n, target_rank) * 0.01
            else:
                # Random initialization
                U_init = np.random.randn(n*m, target_rank) * 0.1
                V_init = np.random.randn(m*p, target_rank) * 0.1
                W_init = np.random.randn(p*n, target_rank) * 0.1
            
            result = _als_rounded(n, m, p, target_rank, U_init, V_init, W_init,
                                  max_iters=150, deadline=deadline - 15)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = target_rank
                found = True
                break
        
        if not found:
            # If we can't go lower, stop
            if target_rank < best_rank - 5:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
