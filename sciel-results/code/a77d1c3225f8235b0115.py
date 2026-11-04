"""Rewritten program for matrix multiplication 2x3 by 3x8 rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return U, V, W for Strassen's 2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
    # A00: U[0], A11: U[3]
    # B00: V[0], B11: V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C00: W[0], C11: W[3]
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00 -> contributes to C10
    # A10: U[2], A11: U[3]
    # B00: V[0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10: W[2]
    W[2][1] = 1
    
    # M3: A00(B01-B11) -> contributes to C01, C11
    # A00: U[0]
    # B01: V[1], B11: V[3]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C01: W[1], C11: W[3]
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11(B10-B00) -> contributes to C00, C10
    # A11: U[3]
    # B10: V[2], B00: V[0]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C00: W[0], C10: W[2]
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)B11 -> contributes to C00, C01
    # A00: U[0], A01: U[1]
    # B11: V[3]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C00: W[0], C01: W[1]
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01) -> contributes to C11
    # A10: U[2], A00: U[0]
    # B00: V[0], B01: V[1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C11: W[3]
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11) -> contributes to C00
    # A01: U[1], A11: U[3]
    # B10: V[2], B11: V[3]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C00: W[0]
    W[0][6] = 1
    
    return U, V, W


def _trivial_2x1x8():
    """
    Return U, V, W for trivial 2x1 by 1x8 multiplication.
    A is 2x1, B is 1x8, C is 2x8.
    U shape (2, 8), V shape (8, 8), W shape (16, 8).
    Indexing: U[i*1+j, c], V[j*8+k, c], W[k*2+i, c].
    """
    n, m, p = 2, 1, 8
    rank = n * m * p  # 16
    U = [[0]*rank for _ in range(n*m)]  # 2 rows
    V = [[0]*rank for _ in range(m*p)]  # 8 rows
    W = [[0]*rank for _ in range(p*n)]  # 16 rows
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _combine_blocks(blocks):
    """
    Combine a list of (U, V, W) blocks into a single decomposition.
    Each block has its own indexing scheme. We need to map them to the global
    2x3 by 3x8 problem.
    
    The 2x3 by 3x8 matrix multiplication can be decomposed as:
    - The 3x3 "middle" part can be seen as:
      We have A (2x3) and B (3x8).
      C[i,k] = sum_j A[i,j] B[j,k]
      
    One way to decompose: split the 3 columns/rows into a 2+1 split.
    A = [A_left (2x2), A_right (2x1)]
    B = [B_top (2x8); B_bottom (1x8)]
    
    C = A_left * B_top + A_right * B_bottom
    
    The first term: 2x2 * 2x8 -> use Strassen for the 2x2 part, but B is 2x8.
    Actually, for 2x2 * 2x8, we can think of B as two 2x4 blocks or...
    
    Better approach: 
    For 2x2 * 2x8:
    C_left = A_left * B_top where A_left is 2x2, B_top is 2x8.
    We can decompose this as: for each pair of rows in B_top, use Strassen.
    Actually, B_top is 2x8, so C_left[i,k] = A_left[i,0]*B_top[0,k] + A_left[i,1]*B_top[1,k]
    
    This is a 2x2 * 2x8 multiplication. We can view this as 8 separate 2x2 * 2x1 problems,
    but that doesn't help with Strassen directly.
    
    Alternative: View 2x2 * 2x8 as computing 8 outputs, each requiring 2x2*2x1.
    But 2x2*2x1 has rank at most 2 (it's just a linear combination).
    
    Let me think differently. The total rank should be:
    - For 2x2 * 2x8: We can use Strassen's 7 products for the 2x2 part, but each product
      gives a 1x8 vector. So we need 7 * 8 = 56? No, that's not right.
    
    Actually, the standard approach: 
    2x2 * 2x8 can be done with rank 7*8/... no.
    
    Let me reconsider. The tensor for 2x2 * 2x8:
    Input: A (2x2), B (2x8). Output: C (2x8).
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This is equivalent to 8 independent 2x2 * 2x1 multiplications (one for each k),
    but they share the same A. So we can use Strassen's 7 products for A, and each
    product gives a scalar, which is then multiplied by B[j,k] for each k.
    
    So for the 2x2 * 2x8 part:
    - Compute 7 Strassen products M1...M7 from A.
    - Each M_r is a scalar.
    - For each output C[i,k], it's a linear combination of M_r with coefficients from B.
    
    The rank contribution: 7 (for the Strassen products) times... no.
    
    Actually, the decomposition is:
    U: 4x7 (Strassen U for A)
    V: for each k, we need to combine B[0,k] and B[1,k] with the Strassen coefficients.
    
    Let me just build it explicitly.
    
    For 2x2 * 2x8:
    U (4x7): Strassen U
    V (16x7): V[j*8+k, c] - for each Strassen product c, V has entries for all 16 (j,k) pairs.
    W (16x7): W[k*2+i, c] - for each Strassen product c, W has entries for all 16 (k,i) pairs.
    
    Wait, this is getting complicated. Let me use a simpler approach.
    
    The total problem 2x3 * 3x8 can be decomposed as:
    Split the middle dimension (3) into 2+1.
    
    Term 1: A[:, 0:2] (2x2) * B[0:2, :] (2x8) -> C1 (2x8)
    Term 2: A[:, 2:3] (2x1) * B[2:3, :] (1x8) -> C2 (2x8)
    
    C = C1 + C2
    
    For Term 2 (2x1 * 1x8): rank = 2*1*8 = 16 (trivial, can't do better since it's just outer products)
    Actually, 2x1 * 1x8: C[i,k] = A[i,0] * B[0,k]. This is rank 16 (each (i,k) pair needs its own term).
    
    For Term 1 (2x2 * 2x8): 
    C1[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is a 2x2 * 2x8 multiplication.
    
    We can use Strassen for the 2x2 part. The 7 Strassen products give linear combinations
    of A entries. Then each output C1[i,k] is a linear combination of these 7 products
    with coefficients depending on B[0,k] and B[1,k].
    
    So the rank for Term 1 is 7 (the Strassen products) but each product needs to be
    combined with B values. In the tensor decomposition:
    - U (4x7): Strassen U
    - V (16x7): For each Strassen product c, V[j*8+k, c] gives the coefficient for B[j,k]
    - W (16x7): For each Strassen product c, W[k*2+i, c] gives the coefficient for C1[i,k]
    
    The total rank = 7 (Term 1) + 16 (Term 2) = 23? That seems too low.
    
    Wait, I think I'm confusing myself. In the tensor decomposition framework:
    Each "term" c corresponds to one rank-1 component.
    
    For Term 1 (2x2 * 2x8):
    The tensor T1 has indices (i,j,k) where i in {0,1}, j in {0,1}, k in {0,...,7}.
    T1[i,j,k] = delta_{i,j} ... no, T1[i,j,k] is the coefficient in C1[i,k] = sum_j A[i,j] B[j,k].
    
    Actually, the multiplication tensor for 2x2 * 2x8 has:
    - A indices: (i, j) with i in {0,1}, j in {0,1} -> 4 entries
    - B indices: (j, k) with j in {0,1}, k in {0,...,7} -> 16 entries  
    - C indices: (k, i) with k in {0,...,7}, i in {0,1} -> 16 entries (using W[k*n+i] convention)
    
    The tensor is T[(i,j), (j,k), (k,i)] = 1 if the indices match, else 0.
    More precisely: T is a 3-tensor where T[a, b, c] = 1 if a corresponds to (i,j), b to (j,k), c to (k,i) and the indices are consistent.
    
    For Strassen to work on 2x2 * 2x8, we need to decompose this tensor.
    The 7 Strassen products give 7 rank-1 components for the 2x2 * 2x1 case.
    For 2x2 * 2x8, we have 8 different k values, so we need 7*8 = 56 rank-1 components?
    
    No, that's not right either. Let me think about this more carefully.
    
    For 2x2 * 2x1: rank 7 (Strassen).
    For 2x2 * 2x8: This is like doing 8 separate 2x2 * 2x1 multiplications in parallel,
    but they share the same A. The B matrices are different (different k values).
    
    In the tensor decomposition, each rank-1 term is a product of:
    - A linear form in A
    - A linear form in B  
    - A linear form in C
    
    For 2x2 * 2x8, a rank-1 term would be:
    (linear form in A) * (linear form in B) * (linear form in C)
    
    The Strassen decomposition of 2x2 * 2x1 gives 7 terms where each term is:
    (linear form in A) * (linear form in B[2x1]) * (linear form in C[2x1])
    
    For 2x2 * 2x8, we can use the same 7 linear forms in A, but for each k, we need
    different linear forms in B and C. So the total rank is 7 * 8 = 56?
    
    That gives total rank = 56 + 16 = 72, which is worse than schoolbook (48).
    
    I think the issue is that Strassen doesn't directly apply to rectangular matrices in this way.
    
    Let me reconsider. The known result is that 2x3 * 3x8 has rank at most 44 using
    Strassen for 2x2 subproblems. Let me look at this differently.
    
    Perhaps the decomposition is:
    - Use Strassen for the 2x2 * 2x2 part (rank 7)
    - Handle the remaining parts trivially
    
    Actually, I recall that the finding says rank 44 using Strassen for 2x2x2 subproblems.
    Let me just implement the known decomposition.
    
    The 2x3 * 3x8 multiplication:
    C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k]
    
    Split j into 0,1 and 2:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    The first two terms form a 2x2 * 2x8 multiplication.
    The last term is a 2x1 * 1x8 multiplication.
    
    For the 2x2 * 2x8 part, I'll use a decomposition that achieves rank 28 (7 per 4 k-values? No...)
    
    Actually, let me just use the trivial decomposition for now and get a valid answer,
    then optimize.
    
    Total rank with trivial: 2*3*8 = 48.
    """
    pass


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Start with schoolbook (rank 48) as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try Strassen-based decomposition for rank 44
    # Split middle dimension 3 into 2+1
    # Term 1: 2x2 * 2x8 (rows 0,1 of A, rows 0,1 of B)
    # Term 2: 2x1 * 1x8 (row 2 of A, row 2 of B)
    
    # For 2x2 * 2x8, use Strassen-like decomposition
    # The 2x2 * 2x8 tensor can be decomposed with rank 28 using
    # Strassen's 7 products applied to each pair of k values... 
    # Actually, let me use a simpler approach.
    
    # For 2x2 * 2x8:
    # We can think of this as 8 separate 2x2 * 2x1 multiplications.
    # Each 2x2 * 2x1 has rank 7 (Strassen).
    # But they share A, so we can reuse the 7 Strassen products.
    # Total rank for 2x2 * 2x8 = 7 * 8 / ... 
    
    # Let me just build it explicitly with rank 28 for the 2x2*2x8 part
    # by using 4 Strassen-like blocks of 2x2*2x2.
    
    # Split k into 4 groups of 2: k in {0,1}, {2,3}, {4,5}, {6,7}
    # For each group, use Strassen (rank 7) for 2x2 * 2x2
    # Total rank for 2x2 * 2x8 = 4 * 7 = 28
    
    # Term 2: 2x1 * 1x8 has rank 16 (trivial)
    
    # Total rank = 28 + 16 = 44
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 rows
    V = [[0] * rank_total for _ in range(m * p)]  # 24 rows
    W = [[0] * rank_total for _ in range(p * n)]  # 16 rows
    
    # Term 1: 2x2 * 2x8 using 4 Strassen blocks
    # A indices: i in {0,1}, j in {0,1} -> U[i*3+j, c] (global indexing for 2x3)
    # B indices: j in {0,1}, k in {0,...,7} -> V[j*8+k, c]
    # C indices: k in {0,...,7}, i in {0,1} -> W[k*2+i, c]
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    col = 0
    for kg in range(4):  # 4 groups of 2 k-values
        for c in range(7):  # 7 Strassen products
            # U: A[i,j] for i in {0,1}, j in {0,1}
            # Global U index: i*3 + j (since m=3, j is 0 or 1)
            for i in range(2):
                for j in range(2):
                    val = strassen_U[i * 2 + j][c]
                    if val != 0:
                        U[i * 3 + j][col] += val
            
            # V: B[j,k] for j in {0,1}, k in {2*kg, 2*kg+1}
            # Global V index: j*8 + k
            for j in range(2):
                for dk in range(2):
                    k = 2 * kg + dk
                    val = strassen_V[j * 2 + dk][c]
                    if val != 0:
                        V[j * 8 + k][col] += val
            
            # W: C[k,i] for k in {2*kg, 2*kg+1}, i in {0,1}
            # Global W index: k*2 + i
            for i in range(2):
                for dk in range(2):
                    k = 2 * kg + dk
                    val = strassen_W[dk * 2 + i][c]
                    if val != 0:
                        W[k * 2 + i][col] += val
            
            col += 1
    
    # Term 2: 2x1 * 1x8 (trivial)
    # A index: i in {0,1}, j=2 -> U[i*3+2, c]
    # B index: j=2, k in {0,...,7} -> V[2*8+k, c]
    # C index: k in {0,...,7}, i in {0,1} -> W[k*2+i, c]
    
    for i in range(2):
        for k in range(8):
            U[i * 3 + 2][col] = 1
            V[2 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Verify correctness
    # C[i,k] = sum_c U[i*3+j, c] * V[j*8+k, c] * W[k*2+i, c] summed over j
    # Actually, the decomposition is: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right. The standard decomposition is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # No, it's: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for each c,
    # but that doesn't make sense with the sums.
    
    # The correct interpretation:
    # For each term c:
    #   intermediate_A_c[i] = sum_j U[i*m+j, c] * A[i,j]  -- no, this isn't standard
    
    # Let me use the standard tensor decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 if this is a valid multiplication term, else 0.
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) where the
    # multiplication C[i,k] = sum_j A[i,j]*B[j,k] has a term.
    
    # Actually, the multiplication tensor is:
    # T[a,b,c] where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # T[a,b,c] = 1 if the indices are consistent, i.e., the j in a matches the j in b,
    # and the k in b matches the k in c, and the i in a matches the i in c.
    
    # So T[i,j, j,k, k,i] = 1, meaning for each (i,j,k), there's a 1.
    
    # The decomposition: T[i,j, j,k, k,i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let me verify with a simple test.
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0  # T[i,j,k] = 1
                    rhs = 0.0
                    for c in range(rank):
                        rhs += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(lhs - rhs) > tol:
                        return False
        # Also check that non-terms are zero (but in this case all (i,j,k) are terms)
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
