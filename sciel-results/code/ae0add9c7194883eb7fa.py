import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify(U, V, W, tol=1e-8):
    """
    Verify U, V, W decompose the matrix multiplication tensor.
    Convention: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    Vectorized: build the 3-tensor via einsum and check all entries equal 1.
    """
    n, m, p = DIMENSIONS
    if not isinstance(U, np.ndarray):
        U = np.array(U, dtype=float)
    if not isinstance(V, np.ndarray):
        V = np.array(V, dtype=float)
    if not isinstance(W, np.ndarray):
        W = np.array(W, dtype=float)
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    Ut = U.reshape(n, m, -1)
    Vt = V.reshape(m, p, -1)
    Wt = W.reshape(p, n, -1)
    # T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', Ut, Vt, Wt)
    return bool(np.all(np.abs(T - 1.0) <= tol))

def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_block_schoolbook_6x6():
    """
    6x6x6 decomposition via 2x2 blocks of 3x3, each 3x3 product done schoolbook.
    Rank = 4 * 9 = 36.
    This is the clean, structurally-minimal block decomposition.
    """
    n, m, p = 6, 6, 6
    nblocks = 2
    block_size = 3
    inner_rank = block_size ** 3  # 9 schoolbook terms per 3x3 product

    total_rank = nblocks ** 3 * inner_rank  # 8 * 9 = 72? No.
    # Wait: 2x2 block matrix multiply: C_bk = sum_bj A_bi,bj * B_bj,bk.
    # Number of block products = nblocks^3 = 8 (bi, bj, bk with the middle index summed).
    # Each block product is a 3x3 x 3x3 = rank 9 schoolbook.
    # Total rank = 8 * 9 = 72.
    #
    # Hmm, that's 72, not 36. Let me reconsider.
    # For a 2x2 block matrix multiply, C has 4 blocks, each C_bk = sum_bj A_bi,bj B_bj,bk.
    # For each (bi, bk) output block, we sum over bj in {0,1}: two 3x3 products.
    # So total 3x3 products = 4 output blocks * 2 = 8.
    # Each 3x3 product schoolbook = rank 9.
    # Total rank = 8 * 9 = 72.
    #
    # So the "36" claim in the current program was wrong; it's actually 72.
    # Let me build it correctly with 72.

    total_rank = nblocks ** 3 * inner_rank  # 8 * 9 = 72
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    for bi in range(nblocks):
        for bj in range(nblocks):
            for bk in range(nblocks):
                # Block product: A[bi,bj] * B[bj,bk] -> C[bi,bk]
                for r in range(inner_rank):
                    # r indexes the schoolbook term within this 3x3 product.
                    # Schoolbook term r corresponds to (di, dj, dk) with r = di*3 + dj*3 + dk? 
                    # Actually schoolbook: C[di,dk] += A[di,dj]*B[dj,dk], one term per (di,dj,dk).
                    di = r // 9
                    rem = r % 9
                    dj = rem // 3
                    dk = rem % 3
                    # U: A[bi,bj] element (di,dj)
                    i = block_size * bi + di
                    j = block_size * bj + dj
                    U[i * m + j, col] = 1.0
                    # V: B[bj,bk] element (dj,dk)
                    jv = block_size * bj + dj
                    k = block_size * bk + dk
                    V[jv * p + k, col] = 1.0
                    # W: C[bi,bk] element (dk, di) -> W[k*n+i]
                    kw = block_size * bk + dk
                    iw = block_size * bi + di
                    W[kw * n + iw, col] = 1.0
                    col += 1
    return U, V, W

def _build_block_strassen_6x6():
    """
    6x6x6 via 3x3 blocks of 2x2 Strassen. Rank = 27 * 7 = 189.
    Kept as a cross-check / alternative.
    """
    n, m, p = 6, 6, 6
    nblocks = 3
    block_size = 2

    # Strassen 2x2 factors
    srank = 7
    Us = np.zeros((4, srank))
    Vs = np.zeros((4, srank))
    Ws = np.zeros((4, srank))
    # P1 = (A00+A11)(B00+B11)
    Us[0,0]=1; Us[3,0]=1; Vs[0,0]=1; Vs[3,0]=1
    Ws[0,0]=1; Ws[3,0]=-1  # C00=+1, C11=-1
    # P2 = (A10+A11)B00
    Us[2,1]=1; Us[3,1]=1; Vs[0,1]=1
    Ws[2,1]=1; Ws[3,1]=-1  # C10=+1, C11=-1
    # P3 = A00(B01-B11)
    Us[0,2]=1; Vs[1,2]=1; Vs[3,2]=-1
    Ws[1,2]=1; Ws[3,2]=1  # C01=+1, C11=+1
    # P4 = A11(B10-B00)
    Us[3,3]=1; Vs[2,3]=1; Vs[0,3]=-1
    Ws[0,3]=1; Ws[2,3]=1  # C00=+1, C10=+1
    # P5 = (A00+A01)B11
    Us[0,4]=1; Us[1,4]=1; Vs[3,4]=1
    Ws[0,4]=-1; Ws[1,4]=1  # C00=-1, C01=+1
    # P6 = (A10-A00)(B00+B01)
    Us[2,5]=1; Us[0,5]=-1; Vs[0,5]=1; Vs[1,5]=1
    Ws[3,5]=1  # C11=+1
    # P7 = (A01-A11)(B10+B11)
    Us[1,6]=1; Us[3,6]=-1; Vs[2,6]=1; Vs[3,6]=1
    Ws[0,6]=1  # C00=+1

    total_rank = nblocks ** 3 * srank  # 27 * 7 = 189
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    for bi in range(nblocks):
        for bj in range(nblocks):
            for bk in range(nblocks):
                for r in range(srank):
                    for di in range(block_size):
                        for dj in range(block_size):
                            i = block_size * bi + di
                            j = block_size * bj + dj
                            U[i * m + j, col] = Us[di * 2 + dj, r]
                    for dj in range(block_size):
                        for dk in range(block_size):
                            j = block_size * bj + dj
                            k = block_size * bk + dk
                            V[j * p + k, col] = Vs[dj * 2 + dk, r]
                    for dk in range(block_size):
                        for di in range(block_size):
                            k = block_size * bk + dk
                            i = block_size * bi + di
                            W[k * n + i, col] = Ws[dk * 2 + di, r]
                    col += 1
    return U, V, W

def _greedy_rank_reduction(U, V, W, deadline, tol=1e-8, max_iters=2000):
    """
    Try to remove redundant columns from a valid decomposition.
    A column r is redundant if removing it still leaves a valid decomposition,
    i.e., the residual tensor (which is 0) stays 0. But since the decomposition
    is exact, removing a column makes the sum != 1 in general. Instead, we check
    whether column r is linearly dependent in a way that the tensor contribution
    is already covered. The clean test: remove column r and re-verify; if still
    valid, keep it removed.
    This is O(rank * verify_cost). Guard by deadline.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U = U.copy(); V = V.copy(); W = W.copy()

    # Precompute the tensor for fast residual check.
    def tensor_of(Uu, Vv, Ww):
        Ut = Uu.reshape(n, m, -1)
        Vt = Vv.reshape(m, p, -1)
        Wt = Ww.reshape(p, n, -1)
        return np.einsum('ijr,jkr,kir->ijk', Ut, Vt, Wt)

    current = tensor_of(U, V, W)
    assert np.all(np.abs(current - 1.0) <= tol)

    # Contribution of each column r to the tensor:
    # T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can compute all column contributions at once.
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    # T_cols[i,j,k,r] = Ut[i,j,r]*Vt[j,k,r]*Wt[k,i,r]
    T_cols = np.einsum('ijr,jkr,kir->ijk r', Ut, Vt, Wt)
    # Sum over r should be 1 everywhere.
    # Removing column r: new tensor = current - T_cols[:,:,:,r].
    # Valid if |current - T_cols[:,:,:,r] - 1| <= tol, i.e. T_cols[:,:,:,r] ~ 0.
    # So a column is removable iff its tensor contribution is ~0 everywhere.
    for r in range(rank - 1, -1, -1):
        if time.time() > deadline:
            break
        contrib = T_cols[:, :, :, r]
        if np.all(np.abs(contrib) <= tol):
            # Remove column r
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            # Recompute T_cols for remaining columns
            rank = U.shape[1]
            Ut = U.reshape(n, m, rank)
            Vt = V.reshape(m, p, rank)
            Wt = W.reshape(p, n, rank)
            T_cols = np.einsum('ijr,jkr,kir->ijk r', Ut, Vt, Wt)
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill

    best_rank = float('inf')
    best_result = None

    candidates = []

    # Primary: block schoolbook (rank 72)
    if time.time() < deadline - 30:
        try:
            U1, V1, W1 = _build_block_schoolbook_6x6()
            if _verify(U1, V1, W1):
                candidates.append((U1.shape[1], U1, V1, W1))
        except Exception:
            pass

    # Alternative: block Strassen (rank 189)
    if time.time() < deadline - 30:
        try:
            U2, V2, W2 = _build_block_strassen_6x6()
            if _verify(U2, V2, W2):
                candidates.append((U2.shape[1], U2, V2, W2))
        except Exception:
            pass

    # Fallback: full schoolbook (rank 216)
    if time.time() < deadline - 20:
        try:
            U3, V3, W3 = _build_schoolbook()
            if _verify(U3, V3, W3):
                candidates.append((U3.shape[1], U3, V3, W3))
        except Exception:
            pass

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_rank, U, V, W = candidates[0]
    else:
        # Absolute fallback: schoolbook without verification
        U, V, W = _build_schoolbook()
        best_rank = U.shape[1]

    # Attempt greedy rank reduction on the best candidate, within remaining time.
    if time.time() < deadline - 5:
        try:
            U_r, V_r, W_r = _greedy_rank_reduction(U, V, W, deadline)
            if _verify(U_r, V_r, W_r) and U_r.shape[1] < best_rank:
                U, V, W = U_r, V_r, W_r
                best_rank = U.shape[1]
        except Exception:
            pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
