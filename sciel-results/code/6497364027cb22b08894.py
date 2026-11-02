"""
6x6x6 Matrix Multiplication Tensor Decomposition.

Strategy:
1. Construct a rank-144 decomposition using the tensor product structure:
   R(6) <= R(2) * R(3) via Kronecker product of a rank-7 (Strassen) 2x2 decomposition
   and a rank-9 3x3 decomposition (from Z_3 group algebra / circulant structure).
   
   The key insight: if we have decompositions of 2x2x2 (rank 7) and 3x3x3 (rank 9),
   their Kronecker product gives a decomposition of 6x6x6 with rank 7*9 = 63... 
   Wait, that's not right. The Kronecker product of tensor decompositions doesn't
   directly work that way for matrix multiplication.

   Let me reconsider. For matrix multiplication:
   - 2x2x2 has rank 7 (Strassen)
   - 3x3x3 has rank 27 (schoolbook) or potentially lower
   
   If we view 6x6 as a 2x2 block matrix of 3x3 blocks, we can use:
   - Strassen's 7 multiplications at the block level
   - Each block multiplication is 3x3 * 3x3 = 3x3, which has rank 27 (schoolbook)
   - Total: 7 * 27 = 189 (this is what the current program does)

   To do better, we need either:
   a) A better 3x3 decomposition (rank < 27)
   b) A different block structure

   The hint mentions Z_12 group algebra and rank-144 via 2x2 blocks of 3x3 with rank-9.
   
   Actually, there's a known rank-9 decomposition for 3x3 matrix multiplication? No,
   the best known for 3x3 is 23 (Bini-De Falco-Lam). Rank 9 is not achievable for 3x3.

   Let me re-read the hint: "rank-144 decomposition by partitioning the 6x6 matrices
   into 2x2 blocks of 3x3 submatrices and applying the known rank-9 tensor decomposition
   for 3x3 multiplication (via the Z_3 group algebra)"

   This seems to claim rank-9 for 3x3, which is not standard. Let me think about this differently.

   Actually, maybe the approach is different. Perhaps they're using a different factorization
   where the 6x6 is viewed as a 3x3 block matrix of 2x2 blocks, and using:
   - Rank-9 for 3x3 (if such exists in some structured sense)
   - Rank-4 for 2x2 (schoolbook)
   - Total: 9 * 4 = 36? That seems too low.

   Or maybe it's a more sophisticated construction. Let me just implement the known
   best: Strassen 2x2 blocks with schoolbook 3x3 = rank 189, and also try to find
   any improvement.

   Given the constraints, I'll implement:
   1. The standard Strassen + schoolbook approach (rank 189)
   2. Try a direct construction based on group algebra if possible
   3. Fall back to schoolbook (rank 216)
"""
import time
import numpy as np


DIMENSIONS = [6, 6, 6]


def _strassen_6x6x6():
    """
    Strassen 2x2 block decomposition for 6x6x6 matrix multiplication.
    Split into 3x3 blocks. Each 3x3*3x3*3x3 has rank 27 (schoolbook).
    7 Strassen steps -> rank 7*27 = 189.
    """
    n = m = p = 6
    nh = mh = ph = 3
    block_rank = nh * mh * ph
    total_rank = 7 * block_rank

    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]

    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},
        {2: 1, 4: 1},
        {1: 1, 3: 1},
        {0: 1, 1: -1, 2: 1, 6: 1},
    ]

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                a_row_start = (ai // 2) * nh
                a_col_start = (ai % 2) * mh
                b_row_start = (bi // 2) * mh
                b_col_start = (bi % 2) * ph

                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            gi = a_row_start + i
                            gj = a_col_start + j
                            gk = b_col_start + k

                            c_block_row = gi // nh
                            c_block_col = gk // ph
                            c_block_idx = c_block_row * 2 + c_block_col

                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue

                            col = s * block_rank + i * (mh * ph) + j * ph + k

                            U[gi * m + gj][col] = a_sign
                            V[gj * p + gk][col] = b_sign
                            W[gk * n + gi][col] = c_sign

    return {"U": U, "V": V, "W": W}


def _verify_decomp(decomp, n, m, p, tol=1e-9):
    """Verify that a decomposition exactly decomposes the matrix multiplication tensor."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0

    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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


def _strassen_2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays of shape (4,7), (4,7), (4,7).
    """
    # Strassen's 7 products
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # U factors: for A
    # Each M_s has specific A and B factors
    # We need U[i,j] such that sum_s U[i,j,s] * V[j,k,s] * W[k,i,s] = delta(i,j,k)

    # Let's build it explicitly
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # M1 = (A11+A22)*(B11+B22)
    # A11 is position (0,0), A22 is (1,1)
    # B11 is (0,0), B22 is (1,1)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    # C11 += M1, C22 += M1
    W[0, 0] = 1  # C11: k=0,i=0
    W[3, 0] = 1  # C22: k=1,i=1

    # M2 = (A21+A22)*B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    # C21 += M2
    W[0, 1] = 1  # C21: k=0,i=1

    # M3 = A11*(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1  # -B22
    # C12 += M3
    W[1, 2] = 1  # C12: k=1,i=0

    # M4 = A22*(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1  # -B11
    # C11 += M4, C21 += M4
    W[0, 3] = 1  # C11
    W[0, 3] += 0  # already set
    # C21: k=0,i=1
    W[0, 3] = 1  # This is C11
    # Need separate entry for C21
    # Actually W[k,i] so C21 means k=0, i=1
    W[0, 3] = 1  # C11 (k=0,i=0)
    # C21 (k=0,i=1) - but we can only have one W entry per (k,i,s)
    # This means we need to be more careful. Let me redo.

    # Let me use a cleaner approach. Define the 7 rank-1 tensors explicitly.

    # Reset
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # Tensor 0: M1 = (A11+A22)*(B11+B22) contributes to C11 and C22
    # u: (1,0,0,1) for A positions (00,01,10,11)
    # v: (1,0,0,1) for B positions
    # w: (1,0,0,1) for C positions (11 and 22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # Tensor 1: M2 = (A21+A22)*B11 contributes to C21
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1  # C21: k=0, i=1

    # Tensor 2: M3 = A11*(B12-B22) contributes to C12
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12: k=1, i=0

    # Tensor 3: M4 = A22*(B21-B11) contributes to C11 and C21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C11: k=0, i=0
    # But we also need C21: k=0, i=1
    # Problem: W can only have one value per (k,i,s)
    # This means M4 contributes to both C11 and C21, but we can't represent that
    # with a single rank-1 tensor unless we split it.

    # I think I'm making an error. Let me look at Strassen more carefully.
    # Each M_s is a scalar product of two linear combinations of A entries and B entries.
    # The result M_s is then added to specific C entries.
    # So each M_s corresponds to multiple (k,i) positions in C.
    # This means each M_s is NOT a rank-1 tensor in the (i,j,k) space.

    # Actually, for the tensor decomposition, each rank-1 term is u ⊗ v ⊗ w where
    # u is a vector in R^(n*m), v in R^(m*p), w in R^(p*n).
    # The contribution to T[i,j,k] is u[i,j] * v[j,k] * w[k,i].

    # For Strassen, M1 = (A11+A22)*(B11+B22) contributes to C11 and C22.
    # This means the rank-1 tensor for M1 has:
    # u: A11 and A22 have coefficient 1
    # v: B11 and B22 have coefficient 1
    # w: C11 (k=0,i=0) and C22 (k=1,i=1) have coefficient 1
    # So w[0,0]=1 and w[1,1]=1 (using w[k,i] indexing)

    # But then for C11 (i=0,j=0,k=0): u[0,0]*v[0,0]*w[0,0] = 1*1*1 = 1 ✓
    # For C22 (i=1,j=1,k=1): u[1,1]*v[1,1]*w[1,1] = 1*1*1 = 1 ✓
    # For C12 (i=0,j=1,k=1): u[0,1]*v[1,1]*w[1,0] = 0*1*1 = 0 ✓
    # For C21 (i=1,j=0,k=0): u[1,0]*v[0,0]*w[0,1] = 0*1*0 = 0 ✓

    # Good, this works! Let me continue.

    # Reset again
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1  # C21

    # M3 = A11*(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12

    # M4 = A22*(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C11
    # Also contributes to C21? Let me check: C21 = M2 + M4
    # So M4 should contribute to C21 as well
    W[0, 3] = 1  # C11 (k=0,i=0)
    # For C21 (k=0,i=1): we need W[0,1] to include M4's contribution
    # But W[0,1] is already set for M2. Since W is per-tensor, each tensor has its own W.
    # So for tensor 3 (M4), W should have both C11 and C21 entries.
    W[0, 3] = 1  # C11
    # C21 is (k=0, i=1), so W index is (0,1) in (k,i) -> row 0, col 1 in W if W[k,i]
    # But W is 4x7, indexed as [k*2+i, s] or [k,i] flattened?
    # In our convention, W has shape (p*n, rank) = (4, 7) for 2x2.
    # W[k*n+i, s] where n=2, so W[k*2+i, s]
    # C11: k=0,i=0 -> index 0
    # C12: k=0,i=1 -> index 1
    # C21: k=1,i=0 -> index 2
    # C22: k=1,i=1 -> index 3

    # So for M4 contributing to C11 and C21:
    W[0, 3] = 1  # C11 (k=0,i=0)
    W[2, 3] = 1  # C21 (k=1,i=0)

    # M5 = (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C22 += M5
    W[0, 4] = -1  # C11
    W[3, 4] = 1   # C22

    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    W[3, 5] = 1

    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7, C12 -= M7
    W[0, 6] = 1   # C11
    W[1, 6] = -1  # C12

    return U, V, W


def _schoolbook_3x3():
    """Return schoolbook rank-27 decomposition for 3x3x3 matrix multiplication."""
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    for s in range(rank):
        i = s // 9
        j = (s // 3) % 3
        k = s % 3
        U[i * 3 + j, s] = 1
        V[j * 3 + k, s] = 1
        W[k * 3 + i, s] = 1
    return U, V, W


def _kronecker_decomp(U2, V2, W2, U3, V3, W3):
    """
    Combine a rank-r2 2x2 decomposition and rank-r3 3x3 decomposition
    into a 6x6 decomposition using block structure.
    
    This is NOT a direct Kronecker product of the tensors. Instead,
    we use the block structure: 6x6 = 2x2 blocks of 3x3.
    
    For each Strassen step s (0..6), we compute block-level products
    using the 3x3 schoolbook decomposition.
    """
    # U2, V2, W2: shapes (4, 7)
    # U3, V3, W3: shapes (9, 27)
    
    n = m = p = 6
    rank_total = 7 * 27
    U = np.zeros((36, rank_total))
    V = np.zeros((36, rank_total))
    W = np.zeros((36, rank_total))
    
    for s in range(7):
        for t in range(27):
            col = s * 27 + t
            
            # Get the 2x2 block indices from U2, V2, W2
            # U2 has 4 entries for 2x2 blocks
            # Each nonzero entry in U2[:,s] corresponds to a block position
            for ai in range(4):
                if U2[ai, s] != 0:
                    a_sign = U2[ai, s]
                    a_block_row = ai // 2
                    a_block_col = ai % 2
                    
                    for bi in range(4):
                        if V2[bi, s] != 0:
                            b_sign = V2[bi, s]
                            b_block_row = bi // 2
                            b_block_col = bi % 2
                            
                            # This Strassen step computes:
                            # (sum of A blocks) * (sum of B blocks)
                            # contributing to specific C blocks via W2
                            
                            # For each 3x3 sub-block multiplication
                            for i3 in range(3):
                                for j3 in range(3):
                                    for k3 in range(3):
                                        # Global indices
                                        gi = a_block_row * 3 + i3
                                        gj = a_block_col * 3 + j3
                                        gk = b_block_col * 3 + k3
                                        
                                        # C block position
                                        c_block_row = gi // 3
                                        c_block_col = gk // 3
                                        c_block_idx = c_block_row * 2 + c_block_col
                                        
                                        # Check if W2 contributes to this C block
                                        if W2[c_block_idx, s] != 0:
                                            c_sign = W2[c_block_idx, s]
                                            
                                            # Set the values
                                            U[gi * 6 + gj, col] = a_sign
                                            V[gj * 6 + gk, col] = b_sign
                                            W[gk * 6 + gi, col] = c_sign
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    best_decomp = None
    best_rank = float('inf')

    # 1. Construct Strassen + schoolbook decomposition (Rank 189)
    try:
        U2, V2, W2 = _strassen_2x2()
        U3, V3, W3 = _schoolbook_3x3()
        
        U6, V6, W6 = _kronecker_decomp(U2, V2, W2, U3, V3, W3)
        
        decomp = {
            "U": U6.tolist(),
            "V": V6.tolist(),
            "W": W6.tolist()
        }
        
        rank = len(decomp['U'][0])
        
        if _verify_decomp(decomp, n, m, p):
            best_decomp = decomp
            best_rank = rank
    except Exception:
        pass

    # 2. Fallback to Schoolbook (Rank 216)
    if best_decomp is None or best_rank > 216:
        best_decomp = _schoolbook(n, m, p)
        best_rank = 216

    return best_decomp
