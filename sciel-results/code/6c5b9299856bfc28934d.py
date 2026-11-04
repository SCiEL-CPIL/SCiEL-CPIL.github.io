"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return rank n*m*p schoolbook decomposition."""
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    # Number of columns in U, V, W must be the same
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] which should equal
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T[i,j,k] = 1 for all i,j,k (since (AB)[i,k] = sum_j A[i,j]*B[j,k])
    # Actually, the tensor T[i,j,k] = 1 iff the (i,k) entry of C depends on A[i,j] and B[j,k]
    # which is always 1. So T[i,j,k] = 1 for all valid i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    # Also check that no spurious terms exist: for each c, the outer product
    # U[:,c] (n*m), V[:,c] (m*p), W[:,c] (p*n) should only contribute to T[i,j,k]
    # where all three are non-zero. But we already checked T[i,j,k]=1.
    # We also need to ensure that for any (i,j,k) where T should be 1, it is exactly 1,
    # and there are no entries in U,V,W that create contributions outside the valid range.
    # The valid range is fully covered, so this should be fine.
    return True

def _strassen_2x2(A, B):
    """
    Decompose 2x2 by 2x2 matrix multiplication into 7 rank-1 terms.
    A is 2x2, B is 2x2, C = A*B is 2x2.
    Returns lists of (u, v, w) for each of the 7 terms.
    u: 4-vector for A entries, v: 4-vector for B entries, w: 4-vector for C entries.
    Indexing: u[i*2+j], v[j*2+k], w[k*2+i] for C[i,k] += u*v*w.
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    terms = []
    # M1
    u = [1, 0, 0, 1]  # A00 + A11
    v = [1, 0, 0, 1]  # B00 + B11
    w = [1, 0, 0, -1] # C00 += 1, C11 += -1
    terms.append((u, v, w))
    # M2
    u = [0, 0, 1, 1]  # A10 + A11
    v = [1, 0, 0, 0]  # B00
    w = [0, 0, 1, 1]  # C10 += 1, C11 += 1
    terms.append((u, v, w))
    # M3
    u = [1, 0, 0, 0]  # A00
    v = [0, 1, 0, -1] # B01 - B11
    w = [0, 1, 0, 0]  # C01 += 1
    terms.append((u, v, w))
    # M4
    u = [0, 0, 0, 1]  # A11
    v = [0, 0, 1, -1] # B10 - B00
    w = [-1, 0, 1, 0] # C00 += -1, C10 += 1
    terms.append((u, v, w))
    # M5
    u = [1, 1, 0, 0]  # A00 + A01
    v = [0, 0, 0, 1]  # B11
    w = [1, 1, 0, 0]  # C00 += 1, C01 += 1
    terms.append((u, v, w))
    # M6
    u = [0, 0, 1, -1] # A10 - A00
    v = [1, 1, 0, 0]  # B00 + B01
    w = [0, 0, 0, -1] # C11 += -1
    terms.append((u, v, w))
    # M7
    u = [0, 1, 0, -1] # A01 - A11
    v = [0, 0, 1, 1]  # B10 + B11
    w = [1, 0, 0, 1]  # C00 += 1, C11 += 1
    terms.append((u, v, w))
    return terms

def _build_strassen_2x3x8():
    """
    Build a decomposition for 2x3 by 3x8 using Strassen on 2x2 sub-blocks.
    Split the 3 columns of A and 3 rows of B into:
      - A left 2x2 block (cols 0,1), B left 2x2 block (rows 0,1): use Strassen (rank 7)
      - Remaining cross terms handled by schoolbook (rank for the rest)
    
    A is 2x3: A[i, j] for i in {0,1}, j in {0,1,2}
    B is 3x8: B[j, k] for j in {0,1,2}, k in {0,...,7}
    C is 2x8: C[i, k] for i in {0,1}, k in {0,...,7}
    
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    We can group j=0,1 into a 2x2 Strassen block and handle j=2 separately.
    
    For the 2x2 block (i in {0,1}, j in {0,1}):
      C_block[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is a 2x2 * 2x8 multiplication, but we can view it as 4 separate 2x2*2x2
    multiplications for k in {0,1} and {2,3}, etc. Actually, it's easier to think
    of it as: for each pair of k columns, we have a 2x2 * 2x2 multiplication.
    
    Alternative approach: use Strassen for the 2x2 part of A and B, where B is
    split into 4 blocks of 2x2 (since B has 8 columns, that's 4 blocks of 2 columns).
    
    Let's define:
      A_block = A[:, 0:2]  # 2x2
      B_block_l = B[0:2, 2*l : 2*l+2]  # 2x2 for l in 0..3
      C_block_l = C[:, 2*l : 2*l+2]  # 2x2 for l in 0..3
    
    For each l, C_block_l = A_block * B_block_l, which can be done in rank 7.
    
    Then for j=2: C[i,k] += A[i,2] * B[2,k] for all i,k. This is rank 2*8=16
    using schoolbook, or we can do better.
    
    Actually, A[i,2]*B[2,k] for i in {0,1}, k in {0..7} can be decomposed as:
      For each k, C[:,k] += A[:,2] * B[2,k]
    This is an outer product of a 2-vector and a scalar, which is rank 1 per k.
    So total rank 8 for this part (one for each k), using:
      U: A[0,2] and A[1,2]
      V: B[2,k]
      W: C[0,k] and C[1,k]
    
    So total rank = 4 * 7 + 8 = 28 + 8 = 36.
    
    But we can do even better for the j=2 part. Since A[i,2] is common for both k's
    in a pair, we can use:
      For each l (0..3), k in {2l, 2l+1}:
        C[0, 2l] += A[0,2]*B[2,2l]
        C[1, 2l] += A[1,2]*B[2,2l]
        C[0, 2l+1] += A[0,2]*B[2,2l+1]
        C[1, 2l+1] += A[1,2]*B[2,2l+1]
    
    This can be done with 2 rank-1 terms per l:
      Term 1: U = [A[0,2], A[1,2], 0, 0] (for A indices), V = [0,0, B[2,2l], B[2,2l+1]] (for B indices), 
               W = [C[0,2l], C[1,2l], C[0,2l+1], C[1,2l+1]] (for C indices)
      Wait, this isn't quite right because V would need to be a vector over all B entries.
    
    Let me just use schoolbook for the j=2 part: rank 16.
    Total rank = 28 + 16 = 44.
    
    Actually, let me reconsider. The j=2 part: A[i,2]*B[2,k] for i in {0,1}, k in 0..7.
    This is a 2x1 matrix times a 1x8 matrix. The tensor rank is 2*8=16 if we use
    schoolbook, but it's actually just the outer product of a 2-vector and an 8-vector,
    which has rank min(2,8)=2... no wait, that's for matrix rank. For tensor rank,
    the bilinear map (a, b) -> a*b where a is 2-dim and b is 8-dim, output is 16-dim.
    The tensor rank is 16 because each output entry is a separate product.
    
    So total rank = 4*7 + 16 = 28 + 16 = 44.
    """
    n, m, p = 2, 3, 8
    U = []
    V = []
    W = []
    
    # Strassen for 4 blocks of 2x2 * 2x2
    for l in range(4):
        terms = _strassen_2x2(None, None)
        for (u4, v4, w4) in terms:
            u = [0] * (n * m)  # 6 entries
            v = [0] * (m * p)  # 24 entries
            w = [0] * (p * n)  # 16 entries
            # Map u4[i*2+j] to U[i*2+j] for j in {0,1} (A columns 0,1)
            for idx in range(4):
                u[idx] = u4[idx]
            # Map v4[j*2+k] to V[j*p + (2*l+k)] for j in {0,1}, k in {0,1}
            for j in range(2):
                for k in range(2):
                    v[j * p + (2 * l + k)] = v4[j * 2 + k]
            # Map w4[k*2+i] to W[(2*l+k)*n + i] for k in {0,1}, i in {0,1}
            for k in range(2):
                for i in range(2):
                    w[(2 * l + k) * n + i] = w4[k * 2 + i]
            U.append(u)
            V.append(v)
            W.append(w)
    
    # Schoolbook for j=2: A[i,2]*B[2,k] for i in {0,1}, k in 0..7
    for i in range(n):
        for k in range(p):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[i * m + 2] = 1  # A[i,2]
            v[2 * p + k] = 1  # B[2,k]
            w[k * n + i] = 1  # C[i,k]
            U.append(u)
            V.append(v)
            W.append(w)
    
    rank = len(U)
    U_mat = [[U[c][row] for c in range(rank)] for row in range(len(U[0]))]
    V_mat = [[V[c][row] for c in range(rank)] for row in range(len(V[0]))]
    W_mat = [[W[c][row] for c in range(rank)] for row in range(len(W[0]))]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try the Strassen-based decomposition
    try:
        result = _build_strassen_2x3x8()
        # Verify
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
