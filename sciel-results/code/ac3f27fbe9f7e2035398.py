import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    # Vectorized: build 3D tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape: U as (n,m,R), V as (m,p,R), W as (p,n,R)
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: T[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) <= tol))


def _strassen_2x2x2():
    """Return (U,V,W) for rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7) with W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M5
    
    # U[i*2+j, r] corresponds to A[i,j]
    # V[j*2+k, r] corresponds to B[j,k]
    # W[k*2+i, r] corresponds to C[i,k]
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1  # C00=+1, C11=+1, C01=-1, C10=-1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10=+1
    
    # M3: A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1; V[3, 2] = -1  # B10, B11
    W[1, 2] = 1; W[3, 2] = -1  # C01=+1, C11=-1
    
    # M4: A11(B01-B00)
    U[3, 3] = 1  # A11
    V[1, 3] = 1; V[0, 3] = -1  # B01, B00
    W[0, 3] = -1; W[1, 3] = 1  # C00=-1, C01=+1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = 1; W[3, 4] = -1  # C00=+1, C11=-1
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1  # A10, A00
    V[0, 5] = 1; V[2, 5] = 1  # B00, B10
    W[3, 5] = 1  # C11=+1
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01, A11
    V[1, 6] = 1; V[3, 6] = 1  # B01, B11
    W[0, 6] = -1; W[2, 6] = 1  # C00=-1, C10=+1
    
    return U, V, W


def _strassen_4x4x4():
    """Return (U,V,W) for rank-49 Strassen decomposition of 4x4x4 matrix multiplication.
    Uses 7 blocks of 2x2x2 Strassen (rank 7 each).
    U: (16,49), V: (16,49), W: (16,49) with W[k*4+i, r].
    """
    s2U, s2V, s2W = _strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Strassen for 4x4x4: split into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    #
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1, C01 -= M1, C10 -= M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B10-B11)       -> C01 += M3, C11 -= M3
    # M4 = A11(B01-B00)       -> C00 -= M4, C01 += M4
    # M5 = (A00+A01)B11       -> C00 += M5, C11 -= M5
    # M6 = (A10-A00)(B00+B10) -> C11 += M6
    # M7 = (A01-A11)(B01+B11) -> C00 -= M7, C10 += M7
    #
    # Each Mi is a 2x2x2 multiplication, decomposed with Strassen rank 7.
    #
    # Block positions in 4x4:
    # A00: rows 0-1, cols 0-1 -> U indices 0:4 (i*4+j for i in 0-1, j in 0-1)
    # A01: rows 0-1, cols 2-3 -> U indices 2:6
    # A10: rows 2-3, cols 0-1 -> U indices 4:8
    # A11: rows 2-3, cols 2-3 -> U indices 6:10
    #
    # Wait, U has shape (16, 49) with U[i*4+j, r] for i,j in 0..3
    # A00: i=0,1 j=0,1 -> U[0:4]
    # A01: i=0,1 j=2,3 -> U[2:6]
    # A10: i=2,3 j=0,1 -> U[4:8]
    # A11: i=2,3 j=2,3 -> U[6:10]
    #
    # V[j*4+k, r]:
    # B00: j=0,1 k=0,1 -> V[0:4]
    # B01: j=0,1 k=2,3 -> V[2:6]
    # B10: j=2,3 k=0,1 -> V[4:8]
    # B11: j=2,3 k=2,3 -> V[6:10]
    #
    # W[k*4+i, r]:
    # C00: k=0,1 i=0,1 -> W[0:4]
    # C01: k=2,3 i=0,1 -> W[2:6]
    # C10: k=0,1 i=2,3 -> W[4:8]
    # C11: k=2,3 i=2,3 -> W[6:10]
    
    # For each of the 7 Strassen products, we need to:
    # 1. Form the appropriate linear combinations of A-blocks for U
    # 2. Form the appropriate linear combinations of B-blocks for V
    # 3. Use the 2x2x2 Strassen decomposition (s2U, s2V, s2W)
    # 4. Distribute the result to the appropriate C-blocks with signs
    
    # Strassen coefficients:
    # (aU, aV, signs for W distribution)
    # aU: dict mapping block name to coefficient
    # aV: dict mapping block name to coefficient
    # wSigns: dict mapping block name to coefficient
    
    strassen_ops = [
        # (U_blocks, V_blocks, W_blocks)
        # M1: (A00+A11)(B00+B11)
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1, 'C01': -1, 'C10': -1}),
        # M2: (A10+A11)B00
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1}),
        # M3: A00(B10-B11)
        ({'A00': 1}, {'B10': 1, 'B11': -1}, {'C01': 1, 'C11': -1}),
        # M4: A11(B01-B00)
        ({'A11': 1}, {'B01': 1, 'B00': -1}, {'C00': -1, 'C01': 1}),
        # M5: (A00+A01)B11
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': 1, 'C11': -1}),
        # M6: (A10-A00)(B00+B10)
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B10': 1}, {'C11': 1}),
        # M7: (A01-A11)(B01+B11)
        ({'A01': 1, 'A11': -1}, {'B01': 1, 'B11': 1}, {'C00': -1, 'C10': 1}),
    ]
    
    # Block index ranges in the 4x4 space:
    # U[i*4+j]: A block (i2, j2) where i2=0 or 1, j2=0 or 1
    #   A00: i2=0,j2=0 -> U[0:4] (i=0,1; j=0,1)
    #   A01: i2=0,j2=1 -> U[2:6] (i=0,1; j=2,3)
    #   A10: i2=1,j2=0 -> U[4:8] (i=2,3; j=0,1)
    #   A11: i2=1,j2=1 -> U[6:10] (i=2,3; j=2,3)
    
    # V[j*4+k]: B block (j2, k2)
    #   B00: j2=0,k2=0 -> V[0:4]
    #   B01: j2=0,k2=1 -> V[2:6]
    #   B10: j2=1,k2=0 -> V[4:8]
    #   B11: j2=1,k2=1 -> V[6:10]
    
    # W[k*4+i]: C block (k2, i2)
    #   C00: k2=0,i2=0 -> W[0:4]
    #   C01: k2=1,i2=0 -> W[2:6]
    #   C10: k2=0,i2=1 -> W[4:8]
    #   C11: k2=1,i2=1 -> W[6:10]
    
    block_U = {
        'A00': (0, 4),   # U[0:4]
        'A01': (2, 6),   # U[2:6]
        'A10': (4, 8),   # U[4:8]
        'A11': (6, 10),  # U[6:10]
    }
    block_V = {
        'B00': (0, 4),
        'B01': (2, 6),
        'B10': (4, 8),
        'B11': (6, 10),
    }
    block_W = {
        'C00': (0, 4),
        'C01': (2, 6),
        'C10': (4, 8),
        'C11': (6, 10),
    }
    
    # s2U, s2V, s2W have shape (4, 7) each
    # For a 2x2x2 multiplication, U2[i*2+j, r] for i,j in 0,1
    # This maps to the 4 entries of a 2x2 block
    
    for op_idx, (u_blocks, v_blocks, w_blocks) in enumerate(strassen_ops):
        base_r = op_idx * 7
        # For each of the 7 Strassen-2x2x2 terms:
        for r2 in range(7):
            r = base_r + r2
            # U contribution: for each A block with coefficient c, add c * s2U[...]
            for blk, c in u_blocks.items():
                s, e = block_U[blk]
                # s2U[i*2+j, r2] for i,j in 0,1 maps to U[s + i*2+j, r]
                for i2 in range(2):
                    for j2 in range(2):
                        U[s + i2*2 + j2, r] += c * s2U[i2*2 + j2, r2]
            
            # V contribution
            for blk, c in v_blocks.items():
                s, e = block_V[blk]
                for j2 in range(2):
                    for k2 in range(2):
                        V[s + j2*2 + k2, r] += c * s2V[j2*2 + k2, r2]
            
            # W contribution
            for blk, c in w_blocks.items():
                s, e = block_W[blk]
                for k2 in range(2):
                    for i2 in range(2):
                        W[s + k2*2 + i2, r] += c * s2W[k2*2 + i2, r2]
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """Return (U,V,W) for schoolbook decomposition of n x m * m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def _combine_decompositions(decomps):
    """Combine a list of (U,V,W) decompositions into one.
    Each (U,V,W) has the same number of rows in U, V, W as the final result.
    Returns combined (U,V,W).
    """
    if not decomps:
        return None
    
    n, m, p = N, M, P
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    c = 0
    for U_i, V_i, W_i in decomps:
        r = U_i.shape[1]
        U[:, c:c+r] = U_i
        V[:, c:c+r] = V_i
        W[:, c:c+r] = W_i
        c += r
    
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = N, M, P  # 4, 5, 5
    
    # Deadline guard
    if time.time() - t0 > DEADLINE_S:
        # Fallback to schoolbook rank 100
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy: Split the inner dimension (5) into 1+4.
    # A is 4x5, B is 5x5, C is 4x5.
    # A = [A1 (4x1) | A2 (4x4)], B = [[B11 (1x1), B12 (1x4)], [B21 (4x1), B22 (4x4)]]
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    #
    # A1*B11: 4x1 * 1x1, schoolbook rank 4
    # A1*B12: 4x1 * 1x4, schoolbook rank 16
    # A2*B21: 4x4 * 4x1, schoolbook rank 16
    # A2*B22: 4x4 * 4x4, use Strassen rank 49
    #
    # Total: 4 + 16 + 16 + 49 = 85
    
    decomps = []
    
    # Helper to create a schoolbook decomposition embedded in the full space
    def make_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off):
        """Create schoolbook for A[i_off:i_off+n_b, j_off:j_off+m_b] * B[j_off:j_off+m_b, k_off:k_off+p_b]
        contributing to C[i_off:i_off+n_b, k_off:k_off+p_b]
        """
        rank = n_b * m_b * p_b
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[(i_off+i)*m + (j_off+j), c] = 1.0
                    V[(j_off+j)*p + (k_off+k), c] = 1.0
                    W[(k_off+k)*n + (i_off+i), c] = 1.0
                    c += 1
        return U, V, W
    
    # A1*B11: A[0:4,0:1] * B[0:1,0:1] -> C[0:4,0:1]
    decomps.append(make_schoolbook(4, 1, 1, 0, 0, 0))
    
    # A1*B12: A[0:4,0:1] * B[0:1,1:5] -> C[0:4,1:5]
    decomps.append(make_schoolbook(4, 1, 4, 0, 0, 1))
    
    # A2*B21: A[0:4,1:5] * B[1:5,0:1] -> C[0:4,0:1]
    decomps.append(make_schoolbook(4, 4, 1, 0, 1, 0))
    
    # A2*B22: A[0:4,1:5] * B[1:5,1:5] -> C[0:4,1:5]
    # Use Strassen rank 49 for 4x4x4
    s4U, s4V, s4W = _strassen_4x4x4()
    # Embed into full space:
    # s4U has shape (16, 49) with s4U[i*4+j, r] for i,j in 0..3
    # This corresponds to A[0:4, 1:5] -> U[(0+i)*5 + (1+j), r]
    # s4V has shape (16, 49) with s4V[j*4+k, r]
    # This corresponds to B[1:5, 1:5] -> V[(1+j)*5 + (1+k), r]
    # s4W has shape (16, 49) with s4W[k*4+i, r]
    # This corresponds to C[0:4, 1:5] -> W[(1+k)*4 + (0+i), r]
    
    U_embed = np.zeros((n*m, 49))
    V_embed = np.zeros((m*p, 49))
    W_embed = np.zeros((p*n, 49))
    
    # Vectorized embedding
    for i in range(4):
        for j in range(4):
            U_embed[i*5 + 1 + j, :] = s4U[i*4+j, :]
    for j in range(4):
        for k in range(4):
            V_embed[(1+j)*5 + 1 + k, :] = s4V[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W_embed[(1+k)*4 + i, :] = s4W[k*4+i, :]
    
    decomps.append((U_embed, V_embed, W_embed))
    
    # Combine all decompositions
    U, V, W = _combine_decompositions(decomps)
    
    # Deadline guard: if we're running out of time, return what we have
    if time.time() - t0 > DEADLINE_S - 5:
        # Return current decomposition even if not validated
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Validate (vectorized, fast)
    if not _validate(U, V, W):
        # Fallback to schoolbook rank 100
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
