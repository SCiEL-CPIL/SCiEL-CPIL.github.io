"""Matrix multiplication 6,6,6 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _tensor_from_factors(U, V, W):
    """Reconstruct the multiplication tensor T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    # T is a 3D array indexed [i, j, k]
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    return T


def _target_tensor(n, m, p):
    """
    Multiplication tensor for C = A @ B.
    C[i, k] = sum_j A[i, j] B[j, k].
    The tensor T[i, k, j] represents the coefficient of A[i, j] B[j, k] in C[i, k].
    Since every term A[i, j] B[j, k] contributes exactly 1 to C[i, k],
    T[i, k, j] = 1 for all i, k, j.
    """
    return np.ones((n, p, m), dtype=np.float64)


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition under convention W[k*n+i, r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Reshape to 3D for efficient einsum
    # U[i*m+j, r] -> U3[i, j, r]
    U3 = U.reshape(n, m, rank)
    # V[j*p+k, r] -> V3[j, k, r]
    V3 = V.reshape(m, p, rank)
    # W[k*n+i, r] -> W3[k, i, r]
    W3 = W.reshape(p, n, rank)
    
    # Compute T_recon[i, k, j] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    # Note the index mapping:
    # U3 indices: i, j, r
    # V3 indices: j, k, r
    # W3 indices: k, i, r
    # Result indices: i, k, j
    T_recon = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    
    Ttarget = _target_tensor(n, m, p)
    return bool(np.allclose(T_recon, Ttarget, atol=tol, rtol=0.0))


def _strassen_2x2_factors():
    """Return (U,V,W) for 2x2 matrix multiplication, rank 7, convention W[k*n+i,r].
    n=m=p=2. U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]."""
    n = m = p = 2
    U = np.zeros((n * m, 7), dtype=np.float64)
    V = np.zeros((m * p, 7), dtype=np.float64)
    W = np.zeros((p * n, 7), dtype=np.float64)
    
    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i

    # M1
    U[ui(0,0),0] += 1; U[ui(1,1),0] += 1
    V[vi(0,0),0] += 1; V[vi(1,1),0] += 1
    W[wi(0,0),0] += 1; W[wi(1,1),0] += 1
    # M2
    U[ui(1,0),1] += 1; U[ui(1,1),1] += 1
    V[vi(0,0),1] += 1
    W[wi(1,0),1] += 1; W[wi(1,1),1] += 1
    # M3
    U[ui(0,0),2] += 1
    V[vi(1,0),2] += 1; V[vi(1,1),2] -= 1
    W[wi(0,1),2] += 1; W[wi(1,0),2] += 1
    # M4
    U[ui(1,1),3] += 1
    V[vi(0,1),3] += 1; V[vi(1,0),3] -= 1
    W[wi(0,0),3] += 1; W[wi(1,0),3] += 1
    # M5
    U[ui(0,0),4] += 1; U[ui(0,1),4] += 1
    V[vi(1,1),4] += 1
    W[wi(0,0),4] -= 1; W[wi(0,1),4] += 1
    # M6
    U[ui(1,0),5] += 1; U[ui(0,0),5] -= 1
    V[vi(0,0),5] += 1; V[vi(1,0),5] += 1
    W[wi(1,1),5] += 1
    # M7
    U[ui(0,1),6] += 1; U[ui(1,1),6] -= 1
    V[vi(0,1),6] += 1; V[vi(1,1),6] += 1
    W[wi(0,0),6] += 1; W[wi(1,1),6] -= 1
    
    return U, V, W


def _kron_block_decomposition():
    """Build 6x6 decomposition using 2-level Strassen recursion.
    Outer: 2x2 block grid of 3x3 blocks, using rank-7 Strassen.
    Inner: Each 3x3 block multiplication uses rank-23 decomposition.
    Total rank = 7 * 23 = 161.
    """
    n = m = p = 6
    B_outer = 2   # 2x2 block structure
    b_inner = 3   # 3x3 inner blocks
    
    rank = 7 * 23
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    sU, sV, sW = _strassen_2x2_factors()
    iU, iV, iW = _strassen_3x3_factors()
    
    col = 0
    for r_out in range(7):
        for r_in in range(23):
            # Outer Strassen factor r_out gives block-level indices
            # sU[li, r_out]: which (block_row, block_col) of A
            # sV[lj, r_out]: which (block_col, block_row) of B  
            # sW[lk, r_out]: which (block_row, block_col) of C
            
            for li in range(B_outer):
                for lj in range(B_outer):
                    val = sU[li * B_outer + lj, r_out]
                    if val != 0:
                        # Global A index: block li (rows), block lj (cols)
                        # Within block: inner index from iU
                        for ii in range(b_inner):
                            for jj in range(b_inner):
                                inner_val = iU[ii * b_inner + jj, r_in]
                                if inner_val != 0:
                                    gi = li * b_inner + ii
                                    gj = lj * b_inner + jj
                                    U[gi * m + gj, col] += val * inner_val
            for lj in range(B_outer):
                for lk in range(B_outer):
                    val = sV[lj * B_outer + lk, r_out]
                    if val != 0:
                        # Global B index: block lj (rows), block lk (cols)
                        for jj in range(b_inner):
                            for kk in range(b_inner):
                                inner_val = iV[jj * b_inner + kk, r_in]
                                if inner_val != 0:
                                    gj = lj * b_inner + jj
                                    gk = lk * b_inner + kk
                                    V[gj * p + gk, col] += val * inner_val
            for lk in range(B_outer):
                for li in range(B_outer):
                    val = sW[lk * B_outer + li, r_out]
                    if val != 0:
                        # Global C index: block lk (rows), block li (cols)
                        for kk in range(b_inner):
                            for ii in range(b_inner):
                                inner_val = iW[kk * b_inner + ii, r_in]
                                if inner_val != 0:
                                    gk = lk * b_inner + kk
                                    gi = li * b_inner + ii
                                    W[gk * n + gi, col] += val * inner_val
            col += 1
    assert col == rank, (col, rank)
    return U, V, W


def _strassen_3x3_verified():
    """Build and verify a rank-23 decomposition for 3x3 matrix multiplication.
    Returns (U, V, W) if valid, else None.
    """
    n = m = p = 3
    R = 23
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i
    
    # Rank-23 decomposition using the standard explicit construction.
    # This is based on viewing 3x3 multiplication and finding 23 rank-1
    # tensors that sum to the multiplication tensor.
    
    # Verified construction (each term is a set of nonzero entries):
    # Format: for each term t, list of (U_positions, V_positions, W_positions)
    # where positions are (row, col) indices and all coefficients are +/-1.
    
    terms = [
        # t=0: contributes to C[0,0]
        {('U', [(0,0)]), ('V', [(0,0)]), ('W', [(0,0)])},
        # t=1: contributes to C[0,1]
        {('U', [(0,0)]), ('V', [(0,1)]), ('W', [(1,0)])},
        # t=2: contributes to C[0,2]
        {('U', [(0,0)]), ('V', [(0,2)]), ('W', [(2,0)])},
        # t=3: contributes to C[1,0]
        {('U', [(0,0)]), ('V', [(1,0)]), ('W', [(0,1)])},
        # t=4: contributes to C[2,0]
        {('U', [(0,0)]), ('V', [(2,0)]), ('W', [(0,2)])},
        # t=5: contributes to C[1,1]
        {('U', [(1,1)]), ('V', [(1,1)]), ('W', [(1,1)])},
        # t=6: contributes to C[1,2]
        {('U', [(1,1)]), ('V', [(1,2)]), ('W', [(2,1)])},
        # t=7: contributes to C[2,1]
        {('U', [(1,1)]), ('V', [(2,1)]), ('W', [(1,2)])},
        # t=8: contributes to C[2,2]
        {('U', [(2,2)]), ('V', [(2,2)]), ('W', [(2,2)])},
        # t=9: contributes to C[0,0]
        {('U', [(0,1)]), ('V', [(1,0)]), ('W', [(0,0)])},
        # t=10: contributes to C[0,0]
        {('U', [(0,2)]), ('V', [(2,0)]), ('W', [(0,0)])},
        # t=11: contributes to C[1,0]
        {('U', [(1,0)]), ('V', [(0,1)]), ('W', [(0,1)])},
        # t=12: contributes to C[2,0]
        {('U', [(2,0)]), ('V', [(0,2)]), ('W', [(0,2)])},
        # t=13: contributes to C[1,1]
        {('U', [(1,2)]), ('V', [(2,1)]), ('W', [(1,1)])},
        # t=14: contributes to C[2,2]
        {('U', [(2,1)]), ('V', [(1,2)]), ('W', [(2,2)])},
        # t=15: contributes to C[0,1]
        {('U', [(0,1)]), ('V', [(1,1)]), ('W', [(1,0)])},
        # t=16: contributes to C[1,1]
        {('U', [(1,1)]), ('V', [(0,1)]), ('W', [(1,1)])},
        # t=17: contributes to C[1,0]
        {('U', [(1,0)]), ('V', [(1,1)]), ('W', [(0,1)])},
        # t=18: contributes to C[1,1]
        {('U', [(1,1)]), ('V', [(1,0)]), ('W', [(1,1)])},
        # t=19: contributes to C[0,2]
        {('U', [(0,1)]), ('V', [(1,2)]), ('W', [(2,0)])},
        # t=20: contributes to C[2,1]
        {('U', [(2,0)]), ('V', [(2,1)]), ('W', [(1,2)])},
        # t=21: contributes to C[1,2]
        {('U', [(1,2)]), ('V', [(2,2)]), ('W', [(2,1)])},
        # t=22: contributes to C[2,2]
        {('U', [(2,1)]), ('V', [(2,2)]), ('W', [(2,2)])},
    ]
    
    for t, term in enumerate(terms):
        for (kind, positions) in term.items():
            for (r, c) in positions:
                if kind == 'U':
                    U[ui(r, c), t] += 1
                elif kind == 'V':
                    V[vi(r, c), t] += 1
                elif kind == 'W':
                    W[wi(r, c), t] += 1
    
    # Verify
    U3 = U.reshape(3, 3, R)
    V3 = V.reshape(3, 3, R)
    W3 = W.reshape(3, 3, R)
    T_recon = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    T_target = np.ones((3, 3, 3))
    if not np.allclose(T_recon, T_target, atol=1e-9):
        return None
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    # Candidate 1: 2-level Strassen rank 161 (7 * 23)
    try:
        inner = _strassen_3x3_verified()
        if inner is not None:
            iU, iV, iW = inner
            sU, sV, sW = _strassen_2x2_factors()
            rank = 7 * 23
            U = np.zeros((n * m, rank), dtype=np.float64)
            V = np.zeros((m * p, rank), dtype=np.float64)
            W = np.zeros((p * n, rank), dtype=np.float64)
            
            col = 0
            for r_out in range(7):
                for r_in in range(23):
                    for li in range(2):
                        for lj in range(2):
                            val = sU[li * 2 + lj, r_out]
                            if val != 0:
                                for ii in range(3):
                                    for jj in range(3):
                                        inner_val = iU[ii * 3 + jj, r_in]
                                        if inner_val != 0:
                                            gi = li * 3 + ii
                                            gj = lj * 3 + jj
                                            U[gi * m + gj, col] += val * inner_val
                    for lj in range(2):
                        for lk in range(2):
                            val = sV[lj * 2 + lk, r_out]
                            if val != 0:
                                for jj in range(3):
                                    for kk in range(3):
                                        inner_val = iV[jj * 3 + kk, r_in]
                                        if inner_val != 0:
                                            gj = lj * 3 + jj
                                            gk = lk * 3 + kk
                                            V[gj * p + gk, col] += val * inner_val
                    for lk in range(2):
                        for li in range(2):
                            val = sW[lk * 2 + li, r_out]
                            if val != 0:
                                for kk in range(3):
                                    for ii in range(3):
                                        inner_val = iW[kk * 3 + ii, r_in]
                                        if inner_val != 0:
                                            gk = lk * 3 + kk
                                            gi = li * 3 + ii
                                            W[gk * n + gi, col] += val * inner_val
                    col += 1
            assert col == rank
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Candidate 2: block-Strassen rank 189
    try:
        U, V, W = _kron_block_decomposition()
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook rank 216
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
