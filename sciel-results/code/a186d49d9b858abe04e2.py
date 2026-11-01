"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(len(U_f[i * m + j])):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != 1:
                    return False
    return True

def _round_to_halfint(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    r = round(2 * x) / 2
    if abs(x - r) < tol:
        return r
    return None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _build_strassen_block_2x3x2():
    """
    Construct a rank-11 decomposition for 2x3x2 matrix multiplication.
    
    Dimensions:
    n=2, m=3, p=2
    U: 6x11
    V: 6x11
    W: 4x11
    
    We use a known optimal or near-optimal decomposition for 2x3x2.
    Actually, let's derive a simple one.
    M (2x3) * A (3x2) = C (2x2)
    
    Let's try to find a rank 11 decomposition.
    Schoolbook is 2*3*2 = 12.
    
    Known: rank(2x2x2) = 7 (Strassen).
    rank(1x3x2) = 6.
    We can split M into two rows.
    Row 1 of M (1x3) * A (3x2) -> 1x2. Rank 6.
    Row 2 of M (1x3) * A (3x2) -> 1x2. Rank 6.
    Total 12.
    
    Can we do better?
    Let's use the structure of Strassen for 2x2x2 and adapt.
    Or simply use a known rank-11 decomposition if it exists.
    If not, we can use rank 12 (schoolbook) for the block, giving total rank 5*12=60.
    But the prompt suggests rank 55 is achievable via 5 blocks of rank 11.
    
    Let's verify if rank(2x3x2) is indeed 11.
    Lower bound: max(2*3, 3*2, 2*2) = 6.
    Upper bound: 12.
    
    Let's try to construct a rank-11 decomposition for 2x3x2.
    U (6x11), V (6x11), W (4x11).
    
    We can use a linear combination approach.
    Let's define the 11 terms.
    
    Actually, a simpler structural approach:
    Split the 3x10 multiplication into:
    1. One 3x2 block (rightmost 2 columns)
    2. Four 3x2 blocks (remaining 8 columns)
    
    If we can find a rank-11 decomposition for 2x3x2, total rank is 5*11 = 55.
    
    Let's search for a rank-11 decomposition of 2x3x2.
    We can use the fact that rank(2x3x2) <= rank(2x3x1) + rank(2x3x1) ? No.
    
    Let's try to construct it explicitly.
    M = [[m11, m12, m13], [m21, m22, m23]]
    A = [[a11, a12], [a21, a22], [a31, a32]]
    C = [[c11, c12], [c21, c22]]
    
    c11 = m11*a11 + m12*a21 + m13*a31
    c12 = m11*a12 + m12*a22 + m13*a32
    c21 = m21*a11 + m22*a21 + m23*a31
    c22 = m21*a12 + m22*a22 + m23*a32
    
    Let's try to share some products.
    Consider the 2x2 submatrix formed by m11, m12, m21, m22 and a11, a12, a21, a22.
    This is a 2x2x2 multiplication. Rank 7 (Strassen).
    The remaining terms involve m13, m23 and a31, a32.
    
    Let S be the Strassen decomposition for the top-left 2x2 part.
    It computes:
    d11 = m11*a11 + m12*a21
    d12 = m11*a12 + m12*a22
    d21 = m21*a11 + m22*a21
    d22 = m21*a12 + m22*a22
    
    Using Strassen (7 mults):
    w1 = (m11+m12)*a11
    w2 = (m11+m12)*(a21+a22)
    w3 = (m21+m22)*a11
    w4 = (m21+m22)*(a12+a21) -- wait, indices are tricky.
    
    Standard Strassen for 2x2:
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    p1 = (a+d)(e+h)
    p2 = (c+d)e
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)h
    p6 = (c-d)(f+g)
    p7 = (b-d)(e+g)
    
    C11 = p1 + p4 - p5 + p7
    C12 = p3 + p5
    C21 = p2 + p4
    C22 = p1 - p2 + p3 + p6
    
    Applying to top-left 2x2 of M and A:
    a=m11, b=m12, c=m21, d=m22
    e=a11, f=a12, g=a21, h=a22
    
    This gives us the contributions to c11, c12, c21, c22 from the first two rows of A.
    
    Remaining contributions:
    c11 += m13*a31
    c12 += m13*a32
    c21 += m23*a31
    c22 += m23*a32
    
    These 4 terms are a 1x3 (actually 2x1 for M part, 1x2 for A part? No, M part is column 3, A part is row 3)
    It's a 2x1 matrix (col 3 of M) times a 1x2 matrix (row 3 of A).
    This is a 2x1x2 multiplication.
    Rank(2x1x2) = 2*1*2 = 4? No, rank(2x1x2) is 2.
    Wait, M_col3 = [m13, m23]^T, A_row3 = [a31, a32].
    Product is [[m13*a31, m13*a32], [m23*a31, m23*a32]].
    This is an outer product of two vectors. Rank 1?
    No, the tensor is T[i,k] = M[i,3] * A[3,k].
    This is rank 1 in the tensor sense?
    U[i,3, c] * V[3,k, c].
    We need 2x2 entries.
    c11 += m13*a31
    c12 += m13*a32
    c21 += m23*a31
    c22 += m23*a32
    
    We can compute these with 2 multiplications:
    p8 = m13*a31
    p9 = m23*a31
    Then c11 += p8, c21 += p9.
    What about c12, c22?
    c12 += m13*a32
    c22 += m23*a32
    
    We need m13*a32 and m23*a32.
    We can compute:
    p10 = m13*a32
    p11 = m23*a32
    
    So we need 4 more multiplications for the bottom-right part?
    Total 7 (Strassen) + 4 (Schoolbook for the 2x1x2 part) = 11.
    
    Yes! Rank(2x3x2) = 11.
    
    So the strategy is:
    1. Split the 3x10 matrix A into 5 blocks of size 3x2.
    2. For each block, use the rank-11 decomposition derived above.
    3. Since the blocks are independent (no shared indices in A), we can just concatenate the factor matrices.
    4. Total rank = 5 * 11 = 55.
    
    Let's construct the factor matrices for one 2x3x2 block.
    
    Indices for one block:
    i in {0, 1} (rows of M)
    j in {0, 1, 2} (cols of M / rows of A block)
    k in {0, 1} (cols of A block)
    
    U_block: 6x11 (rows: i*m+j)
    V_block: 6x11 (rows: j*p_block+k, where p_block=2)
    W_block: 4x11 (rows: k*n+i)
    
    Strassen part (j=0,1 and k=0,1):
    Uses 7 terms.
    w1 = (M[0,0]+M[0,1]) * A[0,0]  -> c11? No, Strassen formulas are specific.
    
    Let's map the Strassen multipliers to the tensor terms.
    The tensor term is U[i,j,c] * V[j,k,c] * W[k,i,c].
    
    For the Strassen part (j=0,1, k=0,1):
    We need to express the 7 Strassen products in terms of U, V, W.
    
    Let's denote the 7 Strassen factors as s1..s7.
    
    s1 = (M[0,0]+M[0,1]) * A[0,0]
       U[0,0] += 1, U[0,1] += 1
       V[0,0] += 1
       Contribution to W?
       s1 contributes to C[0,0] and C[1,1]?
       In standard Strassen:
       p1 = (a+d)(e+h)
       Here a=M[0,0], d=M[0,1]? No, M is 2x3.
       The submatrix is M[0:2, 0:2] and A[0:2, 0:2].
       M_sub = [[M00, M01], [M10, M11]]
       A_sub = [[A00, A01], [A10, A11]]
       
       p1 = (M00+M10) * (A00+A10) ? No, standard is (a+d)(e+h).
       a=M00, b=M01, c=M10, d=M11.
       e=A00, f=A01, g=A10, h=A11.
       
       p1 = (M00+M10) * (A00+A10)
       p2 = (M10+M11) * A00
       p3 = M00 * (A01-A10)
       p4 = M11 * (A10-A00)
       p5 = (M00+M01) * A11
       p6 = (M10-M11) * (A01+A10)
       p7 = (M01-M11) * (A00+A10)
       
       C00 = p1 + p4 - p5 + p7
       C01 = p3 + p5
       C10 = p2 + p4
       C11 = p1 - p2 + p3 + p6
       
       We need to represent these in the tensor form sum_c U[i,j,c] V[j,k,c] W[k,i,c].
       
       For each Strassen term pc, we need to assign it to a column c.
       Since U, V, W are linear, we can add the contributions.
       
       Let's create 11 columns.
       Columns 0-6 for Strassen.
       Columns 7-10 for the remaining 2x1x2 part.
       
       For column c=0 (p1):
       p1 = (M00+M10)(A00+A10)
       This product contributes to C00 and C11.
       C00 = M00*A00 + M00*A10 + M10*A00 + M10*A10
       C11 = M00*A00 + M00*A10 + M10*A00 + M10*A10
       
       Wait, the tensor decomposition sums over c.
       For a specific c, the term is U[i,j,c] V[j,k,c] W[k,i,c].
       
       For c=0:
       We want to generate the terms of p1.
       p1 = (M00+M10)(A00+A10) = M00*A00 + M00*A10 + M10*A00 + M10*A10.
       
       We need:
       sum_{j,k} U[i,j,0] V[j,k,0] W[k,i,0] = p1 * (contribution to C[i,k])?
       No, p1 is a scalar. It contributes to specific C[i,k] entries.
       
       Actually, the tensor T[i,j,k] = 1 for all i,j,k? No.
       The multiplication tensor T has T[i,j,k] = 1 if the term M[i,j]A[j,k] contributes to C[i,k].
       Yes, T[i,j,k] = 1 for all valid i,j,k.
       
       So for each c, we define U[:,c], V[:,c], W[:,c] such that
       sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
       
       Let's look at p1 again.
       p1 = (M00+M10)(A00+A10).
       This is not a single term M[i,j]A[j,k]. It's a product of sums.
       To express it in the tensor decomposition, we expand it:
       p1 = M00*A00 + M00*A10 + M10*A00 + M10*A10.
       
       So the contribution of column c=0 to the tensor entries is:
       T[0,0,0] += U[0,0,0]V[0,0,0]W[0,0,0]
       T[0,1,0] += U[0,1,0]V[1,0,0]W[0,0,0] ... wait.
       
       The index of W is k*n+i.
       For C[0,0], i=0, k=0. W[0,0,0].
       For C[0,1], i=0, k=1. W[1,0,0].
       For C[1,0], i=1, k=0. W[0,1,0].
       For C[1,1], i=1, k=1. W[1,1,0].
       
       We want the sum over c to be 1 for all i,j,k.
       
       Let's assign the Strassen factors to columns.
       
       Column 0: p1 = (M00+M10)(A00+A10)
       U[0,0,0] = 1, U[1,0,0] = 1  (M00, M10)
       V[0,0,0] = 1, V[1,0,0] = 1  (A00, A10)
       W[0,0,0] = 1 (C00)
       W[1,1,0] = 1 (C11)
       
       Check:
       c=0 contributes to:
       i=0, j=0, k=0: U[0,0,0]V[0,0,0]W[0,0,0] = 1*1*1 = 1. (M00*A00 part of p1)
       i=0, j=1, k=0: U[0,1,0]V[1,0,0]W[0,0,0] = 0*1*1 = 0.
       Wait, U[0,1,0] is M[0,1]. In p1, M[0,1] is not involved.
       So U[0,1,0] should be 0.
       
       i=1, j=0, k=0: U[1,0,0]V[0,0,0]W[0,0,0] = 1*1*1 = 1. (M10*A00 part of p1)
       i=1, j=1, k=0: U[1,1,0]V[1,0,0]W[0,0,0] = 0*1*1 = 0.
       
       i=0, j=0, k=1: U[0,0,0]V[0,1,0]W[1,0,0] = 1*0*1 = 0. (A[0,1] is not in p1? p1 uses A00, A10. So V[0,1,0] should be 0. V[1,0,0]=1 is A[1,0].)
       
       Let's re-evaluate V indices.
       V row index is j*p+k.
       j=0, k=0 -> V[0,0].
       j=0, k=1 -> V[0,1].
       j=1, k=0 -> V[1,0].
       j=1, k=1 -> V[1,1].
       
       p1 = (M00+M10)(A00+A10).
       A00 is j=0, k=0.
       A10 is j=1, k=0.
       
       So V[0,0,0] = 1, V[1,0,0] = 1. Others 0.
       U[0,0,0] = 1, U[1,0,0] = 1. Others 0.
       
       W indices: k*n+i.
       C00: k=0, i=0 -> W[0,0].
       C11: k=1, i=1 -> W[1,1].
       
       p1 contributes to C00 and C11.
       So W[0,0,0] = 1, W[1,1,0] = 1.
       
       Now check tensor entries for c=0:
       T[0,0,0] (M00, A00, C00): U[0,0,0]V[0,0,0]W[0,0,0] = 1*1*1 = 1.
       T[0,1,0] (M01, A10, C00): U[0,1,0]V[1,0,0]W[0,0,0] = 0*1*1 = 0.
       T[1,0,0] (M10, A00, C00): U[1,0,0]V[0,0,0]W[0,0,0] = 1*1*1 = 1.
       T[1,1,0] (M11, A10, C00): U[1,1,0]V[1,0,0]W[0,0,0] = 0*1*1 = 0.
       
       T[0,0,1] (M00, A01, C01): U[0,0,0]V[0,1,0]W[1,0,0] = 1*0*1 = 0.
       T[0,1,1] (M01, A11, C01): U[0,1,0]V[1,1,0]W[1,0,0] = 0*0*1 = 0.
       T[1,0,1] (M10, A01, C11): U[1,0,0]V[0,1,0]W[1,1,0] = 1*0*1 = 0.
       T[1,1,1] (M11, A11, C11): U[1,1,0]V[1,1,0]W[1,1,0] = 0*0*1 = 0.
       
       This looks correct for p1.
       
       We need to do this for all 7 Strassen terms and 4 Schoolbook terms.
       
       Let's code this construction.
       
    """
    n, m, p_block = 2, 3, 2
    rank_block = 11
    U = np.zeros((n*m, rank_block))
    V = np.zeros((m*p_block, rank_block))
    W = np.zeros((p_block*n, rank_block))
    
    # Strassen terms
    # p1 = (M00+M10)(A00+A10) -> C00, C11
    U[0,0] = 1; U[1,0] = 1
    V[0,0] = 1; V[1,0] = 1
    W[0,0] = 1; W[1,1] = 1
    
    # p2 = (M10+M11)A00 -> C10
    U[1,1] = 1; U[2,1] = 1  # M10, M11
    V[0,1] = 1              # A00
    W[0,1] = 1              # C10 (i=1, k=0)
    
    # p3 = M00(A01-A10) -> C01
    U[0,2] = 1
    V[0,2] = 1; V[1,2] = -1 # A01, -A10
    W[1,0] = 1              # C01 (i=0, k=1)
    
    # p4 = M11(A10-A00) -> C00, C10
    U[2,3] = 1
    V[0,3] = -1; V[1,3] = 1 # -A00, A10
    W[0,0] += 1             # C00
    W[0,1] += 1             # C10
    
    # p5 = (M00+M01)A11 -> C00, C01
    U[0,4] = 1; U[1,4] = 1  # M00, M01
    V[1,1] = 1              # A11
    W[0,0] += 1             # C00
    W[1,0] += 1             # C01
    
    # p6 = (M10-M11)(A01+A10) -> C11
    U[1,5] = 1; U[2,5] = -1 # M10, -M11
    V[0,5] = 1; V[1,5] = 1  # A01, A10
    W[1,1] += 1             # C11
    
    # p7 = (M01-M11)(A00+A10) -> C00
    U[1,6] = 1; U[2,6] = -1 # M01, -M11
    V[0,6] = 1; V[1,6] = 1  # A00, A10
    W[0,0] += 1             # C00
    
    # Schoolbook for j=2 (M02, M12) and k=0,1 (A20, A21)
    # c7 = M02*A20 -> C00
    U[0,7] = 1
    V[2,0] = 1
    W[0,0] += 1
    
    # c8 = M02*A21 -> C01
    U[0,8] = 1
    V[2,1] = 1
    W[1,0] += 1
    
    # c9 = M12*A20 -> C10
    U[1,9] = 1
    V[2,0] = 1
    W[0,1] += 1
    
    # c10 = M12*A21 -> C11
    U[1,10] = 1
    V[2,1] = 1
    W[1,1] += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Build rank-55 decomposition
    # 5 blocks of 2x3x2
    num_blocks = p // 2  # 5
    rank_block = 11
    total_rank = num_blocks * rank_block
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    U_block, V_block, W_block = _build_strassen_block_2x3x2()
    
    for b in range(num_blocks):
        # U is same for all blocks (M is shared)
        U_full[:, b*rank_block:(b+1)*rank_block] = U_block
        
        # V is block diagonal
        # Block b corresponds to columns 2b, 2b+1 of A
        # V rows: j*p+k
        for j in range(m):
            for k in range(2):
                v_idx = j*p + (2*b + k)
                v_block_idx = j*2 + k
                V_full[v_idx, b*rank_block:(b+1)*rank_block] = V_block[v_block_idx]
        
        # W is block diagonal
        # W rows: k*n+i
        for k in range(2):
            for i in range(n):
                w_idx = (2*b + k)*n + i
                w_block_idx = k*n + i
                W_full[w_idx, b*rank_block:(b+1)*rank_block] = W_block[w_block_idx]
    
    # Convert to list of lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Verify
    if _verify_exact(U_list, V_list, W_list, n, m, p):
        best_result = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = total_rank
    
    # Try to improve?
    # Given time constraints and the structural nature, we stick to 55.
    # If we had more time, we could try to merge blocks.
    
    return best_result
# EVOLVE-BLOCK-END
