import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute decomposed tensor D[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    D = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
    
    return np.allclose(D, T, atol=1e-9, rtol=1e-9)

def _strassen_2x2x5():
    """
    Apply Strassen's 7-term algorithm to multiply a 2x2 matrix by a 2x5 matrix.
    
    This decomposes the 2x2x5 multiplication tensor (schoolbook rank 20)
    into 7 rank-1 tensors, giving rank 7.
    
    The 7 Strassen products for A(2x2) * B(2x2):
      M1 = (A11+A22)(B11+B22)
      M2 = (A11+A22)B21
      M3 = A11(B11-B21)
      M4 = A22(B22-B12)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    For 2x2 * 2x5, we apply this to each 2x2 block of B (columns 0-1, 2-3, 4).
    But since p=5 is odd, we handle it differently:
    Split B into B_left (2x2, cols 0-1) and B_right (2x3, cols 2-4).
    Apply Strassen to the 2x2 part (7 terms) and schoolbook to the 2x3 part (6 terms).
    Total: 7 + 6 = 13 terms for the 2x2x5 tensor.
    
    Actually, let me reconsider. For the GQ decomposition of 2x4x5:
    The 2x4 matrix A is split into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    The 4x5 matrix B is split into B_TL (2x5, rows 0-1) and B_BL (2x5, rows 2-3).
    
    C = A_L * B_TL + A_R * B_BL
    
    Each 2x2 * 2x5 multiplication has schoolbook rank 20.
    Applying Strassen to each: we can use 7 terms for the 2x2x2 part and handle the rest.
    
    Let me use a cleaner approach: for 2x2 * 2x5, use Strassen on a 2x2x2 sub-block
    and schoolbook for the remaining 2x2x3 part.
    
    For A(2x2) * B(2x5):
    - Strassen on cols 0-1 of B: 7 terms
    - Schoolbook on cols 2-4 of B: 2*3 = 6 terms
    Total: 13 terms
    
    For the full 2x4x5:
    - A_L * B_TL: 13 terms
    - A_R * B_BL: 13 terms
    Total: 26 terms
    
    That's better than 40 but let me think if there's a better split.
    
    Actually, for 2x2 * 2x5, we can do better by applying Strassen to the full thing
    by padding. But let's stick with 13+13=26 for now, or try a different approach.
    
    Alternative: Use the fact that 2x2x5 can be decomposed as 2x2x2 + 2x2x3.
    2x2x2 via Strassen: rank 7
    2x2x3 via schoolbook: rank 6
    Total for 2x2x5: rank 13
    
    For 2x4x5 = 2x(2+2)x5 = 2x2x5 + 2x2x5 (with different column blocks of A):
    Each 2x2x5 has rank 13, total rank 26.
    """
    n, m, p = 2, 2, 5
    # For 2x2 * 2x5:
    # Strassen on first 2 columns of B (7 terms)
    # Schoolbook on last 3 columns of B (6 terms)
    rank = 7 + 6  # = 13
    
    # U: shape (2*2, 13) = (4, 13)
    # V: shape (2*5, 13) = (10, 13)  
    # W: shape (5*2, 13) = (10, 13)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    # Strassen terms for 2x2 * 2x2 (columns 0-1 of B)
    # A = [[a11, a12], [a21, a22]]
    # B_sub = [[b11, b12], [b21, b22]] where b11=B[0,0], b12=B[0,1], b21=B[1,0], b22=B[1,1]
    
    # M1 = (A11+A22)(B11+B22)
    # C11 += M1 - M4 - M5 + M7, C12 += M5 - M7, C21 += M2 + M4, C22 += M1 - M2 + M3 + M6
    # For our purposes, we just need the tensor decomposition.
    # Each Strassen term is a rank-1 tensor in the 2x2x2 space.
    
    # Strassen rank-1 tensors for 2x2x2:
    # (U_i, V_i, W_i) for i=1..7
    # U acts on A (2x2), V acts on B (2x2), W acts on C (2x2)
    
    strassen_U = np.array([
        [1, 0, 1, 0],  # A11+A22 for M1
        [1, 0, 1, 0],  # A11+A22 for M2  
        [1, 0, 0, 0],  # A11 for M3
        [0, 0, 0, 1],  # A22 for M4
        [1, 1, 0, 0],  # A11+A12 for M5
        [0, 0, -1, 1], # A21-A11 for M6
        [0, 1, 0, -1], # A12-A22 for M7
    ], dtype=np.float64).T  # shape (4, 7)
    
    strassen_V = np.array([
        [1, 1, 1, 1],  # B11+B22 for M1
        [0, 0, 1, 1],  # B21+B22 for M2 -- wait, let me redo this
    ], dtype=np.float64)
    
    # Let me be more careful with Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B12  -- no, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11+B12)
    # M3 = (A11-A21)(B21-B22)
    # M4 = A11(B11-B21)
    # M5 = A22(B22-B12)
    # M6 = (A11+A12)B21
    # M7 = (A21+A22)B12
    
    # Let me use a well-known formulation:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a11+a22)b21
    # p3 = a11(b11-b21)
    # p4 = a22(b22-b12)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    
    # C11 = p1 - p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    strassen_U = np.array([
        [1, 1, 0, 0],   # a11+a22 for p1
        [1, 1, 0, 0],   # a11+a22 for p2
        [1, 0, 0, 0],   # a11 for p3
        [0, 0, 0, 1],   # a22 for p4
        [1, 1, 0, 0],   # a11+a12 for p5
        [0, 0, -1, 1],  # a21-a11 for p6
        [0, 1, 0, -1],  # a12-a22 for p7
    ], dtype=np.float64).T  # (4, 7)
    
    strassen_V = np.array([
        [1, 1, 1, 1],   # b11+b22 for p1
        [0, 0, 1, 1],   # b21 for p2... 
    ], dtype=np.float64)
    
    # I'm getting confused. Let me just use the direct approach.
    # For the 2x2x5 problem, let me use a simpler decomposition.
    
    # Actually, let me step back and use a cleaner GQ decomposition.
    
    # The 2x4x5 multiplication tensor can be decomposed as:
    # C = A * B where A is 2x4, B is 4x5
    # Split A = [A1 | A2] where A1, A2 are 2x2
    # Split B = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    
    # Each 2x2 * 2x5 product has schoolbook rank 2*2*5 = 20
    # We can reduce each using Strassen-like methods.
    
    # For 2x2 * 2x5:
    # Use Strassen on the 2x2x2 part (cols 0-1 of B): rank 7
    # Use schoolbook on the 2x2x3 part (cols 2-4 of B): rank 6
    # Total: rank 13
    
    # For the full problem: rank 13 + 13 = 26
    
    # Let me construct this explicitly.
    
    # First, let me build the 2x2x5 decomposition with rank 13.
    # Then combine two copies for the full 2x4x5.
    
    # 2x2 * 2x5 decomposition:
    # A = [[a00, a01], [a10, a11]] (indices 0,1,2,3 in row-major)
    # B = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14]] (indices 0-4, 5-9)
    # C = [[c00, c01, c02, c03, c04], [c10, c11, c12, c13, c14]] (indices 0-4, 5-9)
    
    # Strassen part (cols 0-1 of B):
    # p1 = (a00+a11)(b00+b10) -> contributes to c00, c01, c10, c11
    # p2 = (a00+a11)b10 -> contributes to c00, c01, c10, c11
    # p3 = a00(b00-b10) -> contributes to c00, c01, c10, c11
    # p4 = a11(b11-b01) -> contributes to c00, c01, c10, c11
    # p5 = (a00+a01)b11 -> contributes to c00, c01, c10, c11
    # p6 = (a10-a00)(b00+b01) -> contributes to c00, c01, c10, c11
    # p7 = (a01-a11)(b10+b11) -> contributes to c00, c01, c10, c11
    
    # This is getting complex. Let me just use the known result:
    # The 2x2x5 tensor has rank at most 13 (7+6).
    # The 2x4x5 tensor has rank at most 26 (13+13).
    
    # But wait - I should verify this is actually correct by construction.
    # Let me just return a valid rank-26 decomposition.
    
    # Actually, for simplicity and correctness, let me construct the rank-26
    # decomposition directly using the block structure.
    
    # For 2x2 * 2x5, the 13 terms are:
    # Terms 0-6: Strassen for 2x2x2 (B cols 0-1)
    # Terms 7-12: Schoolbook for 2x2x3 (B cols 2-4)
    
    # Strassen 2x2x2 terms (U is 4x7, V is 4x7, W is 4x7):
    # Using the formulation:
    # p1 = (a00+a11)(b00+b10): U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1]
    #   C00 += p1, C01 += 0, C10 += 0, C11 += 0 -- no, this isn't right.
    
    # I think I need to be much more careful. Let me just use the schoolbook
    # for now and return rank 40, but with the correct structure.
    
    # Actually, the inspiration says rank 36 was achieved (40->36).
    # 36 = 4*9 or 2*18 or... 
    # 36 = 2 * 18, where 18 might be the rank of a 2x2x5 subproblem?
    # Or 36 = 4 * 9 where 9 is the rank of a 2x2x2.5 subproblem?
    
    # Let me try: 2x4x5 = direct sum of four 2x1x5 sub-tensors? No, that doesn't make sense.
    
    # 36 = 4 * 9. If we split 2x4 into 4 blocks of 2x1, and each 2x1x5 has rank 9?
    # 2x1x5 schoolbook rank = 1*5 = 5. That's not 9.
    
    # 36 = 2 * 18. If we split 2x4 into two 2x2 blocks, and each 2x2x5 has rank 18?
    # 2x2x5 schoolbook rank = 20. Strassen reduces 2x2x2 to 7, so 2x2x5 could be 7+6=13.
    # 2*13 = 26, not 36.
    
    # 36 = 4 * 9. Hmm.
    # 36 = 3 * 12. 
    # 36 = 6 * 6.
    
    # Maybe the rank 36 comes from a different decomposition.
    # Let me just try to construct a rank-36 decomposition and verify it.
    
    # For now, let me return the schoolbook rank-40 solution.
    # The GQ approach should give us rank 26 or 36, but I need to get the details right.
    
    # Let me try a different approach: use the known Strassen 2x2x2 result
    # and pad to 2x2x5.
    
    # Actually, I realize I should just carefully construct the rank-26 solution.
    # Let me do it step by step.
    
    # 2x2 * 2x5 multiplication:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02, b03, b04],
    #      [b10, b11, b12, b13, b14]]
    # C = A * B = [[c00, c01, c02, c03, c04],
    #              [c10, c11, c12, c13, c14]]
    
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c02 = a00*b02 + a01*b12
    # c03 = a00*b03 + a01*b13
    # c04 = a00*b04 + a01*b14
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # c12 = a10*b02 + a11*b12
    # c13 = a10*b03 + a11*b13
    # c14 = a10*b04 + a11*b14
    
    # Schoolbook: 2*2*5 = 20 terms
    # Strassen on cols 0-1: 7 terms
    # Schoolbook on cols 2-4: 2*2*3 = 12 terms
    # Total: 7 + 12 = 19 terms? No, that's not right either.
    
    # Wait, for the Strassen part, we're computing the 2x2 product of A and B[:,0:2].
    # That gives us c00, c01, c10, c11 using 7 multiplications.
    # For the remaining cols 2-4, we use schoolbook: 2*2*3 = 12 multiplications.
    # Total: 7 + 12 = 19 terms for the 2x2x5 problem.
    
    # Then for 2x4x5: 2 * 19 = 38 terms.
    # That's better than 40 but not 36.
    
    # Hmm, let me reconsider. Maybe I can do better with the Strassen decomposition.
    
    # Actually, I think the key insight is that for the 2x2x5 problem,
    # we can use a more efficient decomposition than 7+12=19.
    
    # Let me just return the schoolbook solution for now.
    # The GQ approach is complex and I need to get it right.
    
    # For the purpose of this task, let me return a valid decomposition.
    # I'll use the schoolbook rank-40 solution.
    
    return None  # Signal to use schoolbook

def _gq_decomposition():
    """
    GQ (Gentleman-Quire) style decomposition of the 2x4x5 multiplication tensor.
    
    Split A (2x4) into A_L (2x2) and A_R (2x2).
    Split B (4x5) into B_T (2x5) and B_B (2x5).
    C = A_L * B_T + A_R * B_B
    
    For each 2x2 * 2x5 product, use a rank-18 decomposition:
    - Strassen 2x2x2 for first 2 cols of B: rank 7
    - Schoolbook 2x2x3 for last 3 cols of B: rank 12
    - Total: rank 19
    
    Wait, 7+12=19, and 2*19=38. But the target is 36.
    
    Let me try: maybe each 2x2x5 has rank 18?
    18 = 7 + 11? No, 2x2x3 schoolbook is 12.
    
    Or maybe the decomposition is different. Let me just construct
    a valid rank-38 solution and see if it verifies.
    
    Actually, let me try rank 36 by using a smarter 2x2x5 decomposition.
    36/2 = 18 per sub-problem.
    
    For 2x2x5 with rank 18:
    - 7 Strassen terms for 2x2x2
    - 11 terms for 2x2x3? That's less than schoolbook 12.
    
    This might be possible if there's a linear dependency in the 2x2x3 part.
    But 2x2x3 schoolbook rank is 12, and it's unlikely to be reducible.
    
    Let me just go with rank 38 (2*19) for now, or find the right decomposition.
    
    Actually, you know what, let me just carefully construct the rank-26 solution
    using the correct Strassen + schoolbook approach.
    """
    n, m, p = 2, 4, 5
    
    # Split into two 2x2x5 sub-problems
    # Sub-problem 1: A[:, 0:2] (2x2) * B[0:2, :] (2x5)
    # Sub-problem 2: A[:, 2:4] (2x2) * B[2:4, :] (2x5)
    
    # For each 2x2x5 sub-problem, use rank 13:
    # - 7 Strassen terms for 2x2x2 (B cols 0-1)
    # - 6 schoolbook terms for 2x2x3 (B cols 2-4)
    # Total: 13 terms per sub-problem, 26 total
    
    # Let me construct the 2x2x5 rank-13 decomposition.
    # A = [[a00, a01], [a10, a11]] (4 entries, indices 0-3)
    # B = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14]] (10 entries, indices 0-9)
    # C = [[c00, c01, c02, c03, c04], [c10, c11, c12, c13, c14]] (10 entries, indices 0-9)
    
    # Strassen for 2x2x2 (B cols 0-1):
    # p1 = (a00+a11)(b00+b10)
    # p2 = (a00+a11)(b01+b11)  
    # p3 = a00(b00-b10)
    # p4 = a11(b01-b11)
    # p5 = (a00+a01)(b10+b11)
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    
    # Hmm, I'm not sure this is the right Strassen formulation.
    # Let me use the standard one:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)(b11+b12)
    # p3 = (a11-a21)(b21-b22)
    # p4 = a11(b11-b21)
    # p5 = a22(b22-b12)
    # p6 = (a11+a12)b21
    # p7 = (a21+a22)b12
    
    # c11 = p1 - p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # In 0-indexed: a11=a00, a12=a01, a21=a10, a22=a11
    # b11=b00, b12=b01, b21=b10, b22=b11
    
    # p1 = (a00+a11)(b00+b10)
    # p2 = (a10+a11)(b00+b01)
    # p3 = (a00-a10)(b10-b11)
    # p4 = a00(b00-b10)
    # p5 = a11(b11-b01)
    # p6 = (a00+a01)b10
    # p7 = (a10+a11)b01
    
    # c00 = p1 - p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # Now, for the tensor decomposition, each p_i is a bilinear form.
    # p1 = (a00+a11)(b00+b10): U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1]
    #   But this contributes to c00, c01, c10, c11 with coefficients.
    
    # Actually, in the tensor decomposition, each rank-1 term is:
    # (U_vector, V_vector, W_vector) where the product U*V*W gives the contribution.
    
    # For p1 = (a00+a11)(b00+b10), this is a single scalar multiplication.
    # It contributes to multiple C entries:
    # c00 += p1, c01 += 0, c10 += 0, c11 += p1 (from the formula c11 = p1 - p2 + p3 + p6)
    # Wait, no. p1 contributes to c00 with coeff 1, c11 with coeff 1.
    # p4 contributes to c00 with coeff -1, c10 with coeff 1.
    # etc.
    
    # So the rank-1 tensors are:
    # Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] (contributes to c00 and c11)
    #   But W should be a vector in C-space, not just one entry.
    
    # I think I'm overcomplicating this. In the tensor decomposition:
    # T[c0,c1,c2] = sum_r U[c0,r] * V[c1,r] * W[c2,r]
    # where c0 is the A-index, c1 is the B-index, c2 is the C-index.
    
    # For the 2x2x2 sub-problem:
    # A-index: 0=a00, 1=a01, 2=a10, 3=a11
    # B-index: 0=b00, 1=b01, 2=b10, 3=b11
    # C-index: 0=c00, 1=c01, 2=c10, 3=c11
    
    # The multiplication tensor T[a,b,c] = 1 if c = a*b (in the matrix sense).
    # T[0,0,0]=1, T[0,2,0]=1, T[1,1,0]=1, T[1,3,0]=1, T[2,0,2]=1, T[2,2,2]=1, T[3,1,2]=1, T[3,3,2]=1
    
    # Schoolbook: 8 terms
    # Strassen: 7 terms
    
    # For Strassen, the 7 rank-1 tensors are:
    # (U1, V1, W1), ..., (U7, V7, W7)
    
    # This is getting too complex to derive by hand. Let me just use the schoolbook
    # and return rank 40. The GQ approach is the right idea but I need to get the
    # details exactly right.
    
    # For now, return None to signal using schoolbook.
    return None

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _strassen_2x2x2():
    """
    Return the Strassen 7-term decomposition of the 2x2x2 multiplication tensor.
    
    Returns U, V, W as numpy arrays of shape (4, 7).
    
    The 7 Strassen products:
    p1 = (a00+a11)(b00+b10)
    p2 = (a10+a11)(b00+b01)  
    p3 = (a00-a10)(b10-b11)
    p4 = a00(b00-b10)
    p5 = a11(b11-b01)
    p6 = (a00+a01)b10
    p7 = (a10+a11)b01
    
    C:
    c00 = p1 - p4 - p5 + p7
    c01 = p3 + p5
    c10 = p2 + p4
    c11 = p1 - p2 + p3 + p6
    """
    # U: shape (4, 7) -- A coefficients
    U = np.array([
        [1, 0, 1, 1, 0, 1, 0],  # a00
        [0, 0, 0, 0, 0, 1, 0],  # a01
        [0, 1, -1, 0, 0, 0, 1], # a10
        [1, 1, 0, 0, 1, 0, 1],  # a11
    ], dtype=np.float64)
    
    # V: shape (4, 7) -- B coefficients
    V = np.array([
        [1, 1, 0, 1, 0, 0, 1],  # b00
        [0, 1, 0, 0, -1, 0, 1], # b01
        [1, 0, 1, -1, 0, 1, 0], # b10
        [0, 0, -1, 0, 1, 0, 0], # b11
    ], dtype=np.float64)
    
    # W: shape (4, 7) -- C coefficients
    W = np.array([
        [1, 0, 0, -1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 1, 0, 0, 1, 0],   # c11
    ], dtype=np.float64)
    
    return U, V, W

def _strassen_2x2x5():
    """
    Return a rank-13 decomposition of the 2x2x5 multiplication tensor.
    
    Uses Strassen (7 terms) for the 2x2x2 part (B cols 0-1)
    and schoolbook (6 terms) for the 2x2x3 part (B cols 2-4).
    
    Returns U (4,13), V (10,13), W (10,13).
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Schoolbook for 2x2x3 (B cols 2-4):
    # A = [[a00, a01], [a10, a11]]
    # B_sub = [[b02, b03, b04], [b12, b13, b14]]
    # C_sub = [[c02, c03, c04], [c12, c13, c14]]
    
    # 6 schoolbook terms:
    # t1 = a00*b02, t2 = a00*b03, t3 = a00*b04
    # t4 = a01*b12, t5 = a01*b13, t6 = a01*b14
    # t7 = a10*b02, t8 = a10*b03, t9 = a10*b04
    # t10 = a11*b12, t11 = a11*b13, t12 = a11*b14
    
    # Wait, that's 12 terms, not 6. For 2x2x3, schoolbook is 2*2*3=12.
    # So total is 7+12=19, not 13.
    
    # Let me reconsider. 2x2x3 schoolbook rank is 12.
    # So 2x2x5 rank is at most 7+12=19.
    
    # For 2x4x5: 2*19 = 38.
    
    # Let me just construct the rank-38 solution.
    U = np.zeros((4, 19), dtype=np.float64)
    V = np.zeros((10, 19), dtype=np.float64)
    W = np.zeros((10, 19), dtype=np.float64)
    
    # Strassen part (cols 0-1 of B):
    U[:, 0:7] = U_s
    V[0:4, 0:7] = V_s  # b00, b01, b10, b11
    W[0:4, 0:7] = W_s  # c00, c01, c10, c11
    
    # Schoolbook part (cols 2-4 of B):
    col = 7
    for i in range(2):  # A rows
        for j in range(2):  # B rows
            for k in range(3):  # B cols 2-4
                U[i * 2 + j, col] = 1
                V[j * 5 + (2 + k), col] = 1
                W[(2 + k) * 2 + i, col] = 1
                col += 1
    
    return U, V, W

def _gq_2x4x5():
    """
    Construct the GQ decomposition of the 2x4x5 multiplication tensor.
    
    Split A (2x4) into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    Split B (4x5) into B_T (2x5, rows 0-1) and B_B (2x5, rows 2-3).
    C = A_L * B_T + A_R * B_B
    
    Each 2x2x5 sub-problem uses rank 19 (7 Strassen + 12 schoolbook).
    Total rank: 38.
    """
    U_sub, V_sub, W_sub = _strassen_2x2x5()
    
    n, m, p = 2, 4, 5
    rank = 38
    
    U = np.zeros((n * m, rank), dtype=np.float64)  # (8, 38)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (20, 38)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (10, 38)
    
    # Sub-problem 1: A_L (cols 0-1) * B_T (rows 0-1)
    # A_L indices: 0,1,2,3 (a00,a01,a10,a11)
    # B_T indices: 0-4 (b00-b04), 5-9 (b10-b14)
    # C indices: 0-4 (c00-c04), 5-9 (c10-c14)
    
    U[0:4, 0:19] = U_sub
    V[0:10, 0:19] = V_sub
    W[0:10, 0:19] = W_sub
    
    # Sub-problem 2: A_R (cols 2-3) * B_B (rows 2-3)
    # A_R indices: 4,5,6,7 (a02,a03,a12,a13)
    # B_B indices: 10-14 (b20-b24), 15-19 (b30-b34)
    # C indices: 0-4 (c00-c04), 5-9 (c10-c14)
    
    U[4:8, 19:38] = U_sub
    V[10:20, 19:38] = V_sub
    W[0:10, 19:38] = W_sub
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication.
    
    Strategy:
    1. Use GQ decomposition: split into two 2x2x5 sub-problems.
    2. Each sub-problem uses Strassen (7) + schoolbook (12) = 19 terms.
    3. Total rank: 38.
    4. Verify the solution.
    """
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    # Try GQ decomposition (rank 38)
    try:
        gq_sol = _gq_2x4x5()
        if _verify(gq_sol["U"], gq_sol["V"], gq_sol["W"]):
            return gq_sol
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 40)
    schoolbook = _schoolbook()
    if _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        return schoolbook
    
    raise ValueError("No valid decomposition found")
# EVOLVE-BLOCK-END
