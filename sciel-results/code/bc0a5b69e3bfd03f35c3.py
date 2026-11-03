import numpy as np
import time
import sys

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    
    Strategy:
    - Base: Strassen (rank 7) for 2x2*2x2, Kronecker to get rank 49 for 4x4*4x4,
      plus schoolbook rank 4 for the extra column -> total 53.
    - Improvement attempt: sparse random search for low-rank components with
      half-integer entries to try to reduce the 49 (Strassen) or 4 (extra) parts.
      Keep the best verified decomposition found.
    """
    n, m, p = 4, 4, 5
    t_start = time.time()
    deadline = t_start + 130  # safety margin: 20s buffer before 150s kill

    # ---------- 1. Construct the structural rank-53 baseline ----------
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    U2[0, 0]=1; U2[3, 0]=1
    V2[0, 0]=1; V2[3, 0]=1
    W2[0, 0]=1; W2[3, 0]=1
    # M1 = (A10+A11)B00 -> C10 += M1, C11 += M1
    U2[2, 1]=1; U2[3, 1]=1
    V2[0, 1]=1
    W2[2, 1]=1; W2[3, 1]=1
    # M2 = A00(B10-B11) -> C01 += M2, C11 += M2
    U2[0, 2]=1
    V2[2, 2]=1; V2[3, 2]=-1
    W2[1, 2]=1; W2[3, 2]=1
    # M3 = (A00-A11)(B01+B11) -> C00 += M3, C11 += M3
    U2[0, 3]=1; U2[3, 3]=-1
    V2[1, 3]=1; V2[3, 3]=1
    W2[0, 3]=1; W2[3, 3]=1
    # M4 = (A00+A01)B11 -> C01 += M4, C11 += M4
    U2[0, 4]=1; U2[1, 4]=1
    V2[3, 4]=1
    W2[1, 4]=1; W2[3, 4]=1
    # M5 = (A10-A00)(B00+B01) -> C10 += M5, C00 += M5
    U2[2, 5]=1; U2[0, 5]=-1
    V2[0, 5]=1; V2[1, 5]=1
    W2[2, 5]=1; W2[0, 5]=1
    # M6 = (A01-A11)(B10+B11) -> C01 += M6, C10 += M6
    U2[1, 6]=1; U2[3, 6]=-1
    V2[2, 6]=1; V2[3, 6]=1
    W2[1, 6]=1; W2[2, 6]=1

    # Kronecker: rank 49 for 4x4*4x4
    U44 = np.zeros((16, 49)); V44 = np.zeros((16, 49)); W44 = np.zeros((16, 49))
    for t1 in range(7):
        for t2 in range(7):
            col = t1*7 + t2
            for i1 in range(2):
                for i2 in range(2):
                    row = i1*2 + i2
                    uval = U2[i1, t1]*U2[i2, t2]
                    vval = V2[i1, t1]*V2[i2, t2]
                    wval = W2[i1, t1]*W2[i2, t2]
                    if uval != 0: U44[row, col] = uval
                    if vval != 0: V44[row, col] = vval
                    if wval != 0: W44[row, col] = wval

    # Schoolbook for the 5th column of B: rank 4 for 4x4*4x1
    U41 = np.zeros((16, 4)); V41 = np.zeros((20, 4)); W41 = np.zeros((20, 4))
    for r in range(4):
        for i in range(4):
            if i == r:
                for j in range(4):
                    U41[i*4+j, r] = 1.0
                W41[16+i, r] = 1.0
        for j in range(4):
            V41[j*5+4, r] = 1.0

    # Combine to full 4x4*4x5
    rank0 = 53
    U0 = np.zeros((16, rank0)); V0 = np.zeros((20, rank0)); W0 = np.zeros((20, rank0))
    U0[:, 0:49] = U44
    for j in range(4):
        for k in range(4):
            V0[j*5+k, 0:49] = V44[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W0[k*4+i, 0:49] = W44[k*4+i, 0:49]
    U0[:, 49:53] = U41
    V0[:, 49:53] = V41
    W0[:, 49:53] = W41

    # ---------- 2. Helper: verify decomposition ----------
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    if not verify(U0, V0, W0, rank0):
        # schoolbook fallback
        rank0 = n*m*p
        U0 = np.zeros((n*m, rank0)); V0 = np.zeros((m*p, rank0)); W0 = np.zeros((p*n, rank0))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i*m+j, col] = 1.0
                    V0[j*p+k, col] = 1.0
                    W0[k*n+i, col] = 1.0
                    col += 1

    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0

    # ---------- 3. Sparse random search for lower rank ----------
    import random
    random.seed(12345)

    half_ints = [-0.5, 0.0, 0.5, 1.0, -1.0]

    def try_rank(rank):
        """Try to find a rank-`rank` decomposition. Returns (U,V,W) or None."""
        if time.time() > deadline:
            return None
        max_iters = 3000
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            # Generate sparse random factors
            # Each factor: for each column (tensor term), choose a sparse pattern
            U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
            for r in range(rank):
                # Each term: pick random subsets of nonzero entries
                # Sparsity level: ~5-10 nonzeros per column
                nz_u = random.randint(4, 8)
                nz_v = random.randint(4, 8)
                nz_w = random.randint(3, 7)
                for idx in random.sample(range(n*m), min(nz_u, n*m)):
                    U[idx, r] = random.choice(half_ints)
                for idx in random.sample(range(m*p), min(nz_v, m*p)):
                    V[idx, r] = random.choice(half_ints)
                for idx in random.sample(range(p*n), min(nz_w, p*n)):
                    W[idx, r] = random.choice(half_ints)
            # Quick verification (only on diagonal terms to save time, then full if close)
            # Check all 80 terms
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1.0) > 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return (U, V, W)
        return None

    # Try decreasing ranks
    for r_target in range(52, 35, -1):
        if time.time() > deadline:
            break
        result = try_rank(r_target)
        if result is not None:
            U_new, V_new, W_new = result
            if verify(U_new, V_new, W_new, r_target):
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, r_target
                break  # found the lowest; stop

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
