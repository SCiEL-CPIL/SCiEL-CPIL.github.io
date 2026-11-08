"""Program for 4x4 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import random

def _strassen_3x3_decomposition():
    """
    Return the 27-term Strassen decomposition for 3x3 matrix multiplication.
    This is the standard 7-term Strassen algorithm applied recursively,
    but for 3x3 we use the direct 7-term Strassen on 3x3 blocks.
    
    Actually, for 3x3, the standard approach is:
    - Pad to 4x4 and use 2-level Strassen (rank 49), or
    - Use a direct 3x3 decomposition.
    
    The known optimal rank for 4x4 is 49 (2-level Strassen).
    Let me construct the 2-level Strassen decomposition for 4x4.
    
    2-level Strassen for 4x4:
    Level 1: Split 4x4 into four 2x2 blocks.
    Level 2: Each 2x2 multiplication uses 7 Strassen terms.
    
    Total: 4 (top-level Strassen terms) * 7 (bottom-level) = 28? No.
    
    Actually, 2-level Strassen:
    - Top level: 7 Strassen multiplications of 2x2 matrices
    - Each 2x2 multiplication: 7 Strassen multiplications of 1x1 (scalars)
    - Total: 7 * 7 = 49
    
    Let me construct this properly.
    
    For 4x4 matrix C = A * B:
    A = [[A11, A12], [A21, A22]] where each is 2x2
    B = [[B11, B12], [B21, B22]] where each is 2x2
    
    Strassen:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Now each Mi is a 2x2 product. For each, we use Strassen again:
    For 2x2 matrices X * Y:
    S1 = (X11 + X22)(Y11 + Y22)
    S2 = (X21 + X22)Y11
    S3 = X11(Y12 - Y22)
    S4 = X22(Y21 - Y11)
    S5 = (X11 + X12)Y22
    S6 = (X21 - X11)(Y11 + Y12)
    S7 = (X12 - X22)(Y21 + Y22)
    
    Result:
    R11 = S1 + S4 - S5 + S7
    R12 = S3 + S5
    R21 = S2 + S4
    R22 = S1 - S2 + S3 + S6
    
    So for each of the 7 top-level Mi, we have 7 scalar multiplications.
    Total: 49 scalar multiplications.
    
    I need to express this as U, V, W matrices where:
    U has shape (4*4, 49) = (16, 49)
    V has shape (4*4, 49) = (16, 49)  
    W has shape (4*4, 49) = (16, 49)
    
    The convention is:
    - U[i*m + j, c] where i is row of A, j is col of A
    - V[j*p + k, c] where j is row of B, k is col of B
    - W[k*n + i, c] where k is row of C (col of B), i is col of C (row of A)
    
    Wait, let me re-read the convention:
    "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    With n=m=p=4:
    - U[i*4+j, c]: i=row of A, j=col of A
    - V[j*4+k, c]: j=row of B, k=col of B
    - W[k*4+i, c]: k=row of C = col of B, i=col of C = row of A
    
    So C[i,k] = sum_c U[i*4+j,c] * V[j*4+k,c] * W[k*4+i,c]
    
    Wait, that's not quite right. Let me think about this more carefully.
    
    The tensor decomposition is: T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    where T_{i,j,k} = delta (i.e., the matrix multiplication tensor has T_{i,j,k} = 1 if the indices form a valid multiplication, but actually for matrix multiplication T_{i,j,k} = 1 for all i,j,k in the standard formulation).
    
    Actually, the matrix multiplication tensor M_{i,j,k} satisfies:
    (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    So the tensor is M_{i,j,k} = 1 for all i,j,k (it's the identity tensor in the sense that it picks out the product).
    
    Wait no. The tensor for matrix multiplication is defined such that:
    C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    
    The tensor T has entries T_{i,j,k} where the multiplication is:
    T(A,B)_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}
    
    So T_{i,j,k} = 1 for all i,j,k. The tensor is all ones of shape (n, m, p).
    
    A rank-r decomposition gives:
    T_{i,j,k} = sum_{c=1}^{r} U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    
    Wait, that doesn't look right either. Let me look at the seed code:
    
    U[i*m+j][column] = 1  -> U[i,j,c]
    V[j*p+k][column] = 1  -> V[j,k,c]
    W[k*n+i][column] = 1  -> W[k,i,c]
    
    And the product sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should equal 1 for all i,j,k.
    
    In the schoolbook case, for each (i,j,k), there's exactly one c where all three are 1, so the sum is 1. Correct.
    
    So I need to construct U, V, W such that for all i,j,k in {0,1,2,3}:
    sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c] = 1
    
    Now let me construct the 2-level Strassen decomposition.
    
    Let me index the 4x4 matrix as:
    A = [[A00, A01, A02, A03],
         [A10, A11, A12, A13],
         [A20, A21, A22, A23],
         [A30, A31, A32, A33]]
    
    Split into 2x2 blocks:
    A11 = [[A00, A01], [A10, A11]] (top-left)
    A12 = [[A02, A03], [A12, A13]] (top-right)
    A21 = [[A20, A21], [A30, A31]] (bottom-left)
    A22 = [[A22, A23], [A32, A33]] (bottom-right)
    
    Similarly for B.
    
    Top-level Strassen:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    Each Mi is a 2x2 matrix. For each, apply Strassen at the scalar level.
    
    For a 2x2 product X * Y where:
    X = [[x00, x01], [x10, x11]]
    Y = [[y00, y01], [y10, y11]]
    
    Strassen:
    S1 = (x00 + x11)(y00 + y11)
    S2 = (x10 + x11)y00
    S3 = x00(y01 - y11)
    S4 = x11(y10 - y00)
    S5 = (x00 + x01)y11
    S6 = (x10 - x00)(y00 + y01)
    S7 = (x01 - x11)(y10 + y11)
    
    R00 = S1 + S4 - S5 + S7
    R01 = S3 + S5
    R10 = S2 + S4
    R11 = S1 - S2 + S3 + S6
    
    Now, for each top-level Mi, I need to express it as a sum of 7 scalar products.
    Each scalar product corresponds to one column in U, V, W.
    
    Total columns: 7 * 7 = 49.
    
    Let me enumerate. For top-level term t (0-6) and bottom-level term s (0-6), column c = t*7 + s.
    
    For each (t, s), I need to determine:
    - Which entries of A (i.e., U[i*4+j]) are involved and with what coefficient
    - Which entries of B (i.e., V[j*4+k]) are involved and with what coefficient
    - Which entries of C (i.e., W[k*4+i]) receive the result and with what coefficient
    
    Let me define the block indices:
    Block (bi, bj) of A has rows 2*bi to 2*bi+1 and cols 2*bj to 2*bj+1.
    So element A_{i,j} is in block (i//2, j//2) at position (i%2, j%2).
    
    Similarly for B and C.
    
    Let me write a helper to get the coefficient of A_{i,j} in a given block combination.
    
    Actually, let me just build this programmatically.
    
    For top-level term t, the A-side is a combination of blocks:
    t=0: (A11 + A22) -> blocks (0,0) and (1,1), each with coeff +1
    t=1: (A21 + A22) -> blocks (1,0) and (1,1), each with coeff +1
    t=2: A11 -> block (0,0), coeff +1
    t=3: A22 -> block (1,1), coeff +1
    t=4: (A11 + A12) -> blocks (0,0) and (0,1), each with coeff +1
    t=5: (A21 - A11) -> block (1,0) coeff +1, block (0,0) coeff -1
    t=6: (A12 - A22) -> block (0,1) coeff +1, block (1,1) coeff -1
    
    The B-side:
    t=0: (B11 + B22) -> blocks (0,0) and (1,1), each with coeff +1
    t=1: B11 -> block (0,0), coeff +1
    t=2: (B12 - B22) -> block (0,1) coeff +1, block (1,1) coeff -1
    t=3: (B21 - B11) -> block (1,0) coeff +1, block (0,0) coeff -1
    t=4: B22 -> block (1,1), coeff +1
    t=5: (B11 + B12) -> blocks (0,0) and (0,1), each with coeff +1
    t=6: (B21 + B22) -> blocks (1,0) and (1,1), each with coeff +1
    
    For each (t, s) where s is the bottom-level Strassen term:
    The bottom-level term s operates on a 2x2 matrix product.
    
    For the A-side of term t, we have a 2x2 matrix (the block combination).
    Let's say the A-side 2x2 matrix has entries a00, a01, a10, a11 (positions within the 2x2 block structure).
    
    For bottom-level Strassen term s:
    s=0: (a00 + a11) -> positions (0,0) coeff +1, (1,1) coeff +1
    s=1: (a10 + a11) * b00 -> A side: positions (1,0) coeff +1, (1,1) coeff +1; B side: position (0,0) coeff +1
    s=2: a00 * (b01 - b11) -> A side: position (0,0) coeff +1; B side: (0,1) coeff +1, (1,1) coeff -1
    s=3: a11 * (b10 - b00) -> A side: position (1,1) coeff +1; B side: (1,0) coeff +1, (0,0) coeff -1
    s=4: (a00 + a01) * b11 -> A side: (0,0) coeff +1, (0,1) coeff +1; B side: (1,1) coeff +1
    s=5: (a10 - a00) * (b00 + b01) -> A side: (1,0) coeff +1, (0,0) coeff -1; B side: (0,0) coeff +1, (0,1) coeff +1
    s=6: (a01 - a11) * (b10 + b11) -> A side: (0,1) coeff +1, (1,1) coeff -1; B side: (1,0) coeff +1, (1,1) coeff +1
    
    Wait, I need to be more careful. The bottom-level Strassen applies to the actual 2x2 blocks.
    
    Let me redefine. For top-level term t:
    - The A-side is a 2x2 matrix formed by combining blocks. Each element of this 2x2 matrix is itself a 2x2 block.
    - Actually no. The top-level Strassen combines 2x2 blocks to form 2x2 matrices, and then multiplies them. Each element of the resulting 2x2 matrix is a scalar (since the blocks are 2x2 and we're doing standard matrix multiplication at the block level... no wait).
    
    I'm confusing myself. Let me restart the construction.
    
    4x4 matrix A:
    Row 0: A[0,0], A[0,1], A[0,2], A[0,3]
    Row 1: A[1,0], A[1,1], A[1,2], A[1,3]
    Row 2: A[2,0], A[2,1], A[2,2], A[2,3]
    Row 3: A[3,0], A[3,1], A[3,2], A[3,3]
    
    Split into 2x2 blocks:
    A_TL = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]  (top-left, block (0,0))
    A_TR = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]  (top-right, block (0,1))
    A_BL = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]  (bottom-left, block (1,0))
    A_BR = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]  (bottom-right, block (1,1))
    
    Top-level Strassen computes 7 products of 2x2 matrices:
    M1 = (A_TL + A_BR) * (B_TL + B_BR)
    M2 = (A_BL + A_BR) * B_TL
    M3 = A_TL * (B_TR - B_BR)
    M4 = A_BR * (B_BL - B_TL)
    M5 = (A_TL + A_TR) * B_BR
    M6 = (A_BL - A_TL) * (B_TL + B_TR)
    M7 = (A_TR - A_BR) * (B_BL + B_BR)
    
    Each Mi is a 2x2 matrix. To compute each Mi, we use Strassen on 2x2 matrices, giving 7 scalar products each.
    
    For example, M1 = X * Y where:
    X = A_TL + A_BR = [[A[0,0]+A[2,2], A[0,1]+A[2,3]], [A[1,0]+A[3,2], A[1,1]+A[3,3]]]
    Y = B_TL + B_BR = [[B[0,0]+B[2,2], B[0,1]+B[2,3]], [B[1,0]+B[3,2], B[1,1]+B[3,3]]]
    
    Strassen on X*Y:
    S1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
       = (A[0,0]+A[2,2]+A[1,1]+A[3,3]) * (B[0,0]+B[2,2]+B[1,1]+B[3,3])
    
    This is a product of a linear combination of A entries and a linear combination of B entries.
    In the tensor decomposition, this contributes to one column c with:
    U[i*4+j, c] = coefficient of A[i,j] in the A-side
    V[j*4+k, c] = coefficient of B[j,k] in the B-side
    W[k*4+i, c] = coefficient for C[i,k] from this term
    
    The result S1 contributes to:
    M1[0,0] += S1
    M1[1,1] += S1
    M1[0,1] -= S1 (from -S5 part... wait no)
    
    Let me redo. Strassen for 2x2:
    R[0,0] = S1 + S4 - S5 + S7
    R[0,1] = S3 + S5
    R[1,0] = S2 + S4
    R[1,1] = S1 - S2 + S3 + S6
    
    So S1 contributes +1 to R[0,0] and +1 to R[1,1].
    
    Now, M1 contributes to C as:
    C_TL[0,0] += M1[0,0]
    C_TL[0,1] += M1[0,1]
    C_TL[1,0] += M1[1,0]
    C_TL[1,1] += M1[1,1]
    C_BR[0,0] += M1[0,0]
    C_BR[0,1] += M1[0,1]
    C_BR[1,0] += M1[1,0]
    C_BR[1,1] += M1[1,1]
    
    Wait, that's not right either. Let me re-derive the top-level Strassen.
    
    C_TL = M1 + M4 - M5 + M7
    C_TR = M3 + M5
    C_BL = M2 + M4
    C_BR = M1 - M2 + M3 + M6
    
    So M1 contributes +1 to C_TL and +1 to C_BR.
    
    M1[0,0] contributes to C_TL[0,0] and C_BR[0,0], i.e., C[0,0] and C[2,2].
    M1[0,1] contributes to C_TL[0,1] and C_BR[0,1], i.e., C[0,1] and C[2,3].
    M1[1,0] contributes to C_TL[1,0] and C_BR[1,0], i.e., C[1,0] and C[3,2].
    M1[1,1] contributes to C_TL[1,1] and C_BR[1,1], i.e., C[1,1] and C[3,3].
    
    So for S1 (which contributes +1 to M1[0,0] and +1 to M1[1,1]):
    W contributions: C[0,0] += 1, C[2,2] += 1, C[1,1] += 1, C[3,3] += 1
    
    OK this is getting complex. Let me just code it up programmatically.
    
    I'll build the U, V, W matrices by iterating over all 49 terms.
    
    Let me define the structure clearly in code.
    """
    n = m = p = 4
    rank = 49
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Top-level Strassen: 7 terms
    # Each term t has:
    #   A-side: a 2x2 matrix formed from blocks of A
    #   B-side: a 2x2 matrix formed from blocks of B
    #   C-side: contribution to blocks of C
    
    # Block indices: (0,0)=TL, (0,1)=TR, (1,0)=BL, (1,1)=BR
    # Element (i,j) in 4x4 is in block (i//2, j//2) at local position (i%2, j%2)
    
    # For each top-level term, define the A-side 2x2 matrix coefficients
    # A_side[t][li][lj] = list of (block_row, block_col, local_row, local_col, coeff)
    # Actually, let me define it as: for each local position (li, lj) in the 2x2,
    # which global elements contribute and with what coefficient.
    
    # Top-level A-side definitions:
    # t=0: A_TL + A_BR
    # t=1: A_BL + A_BR
    # t=2: A_TL
    # t=3: A_BR
    # t=4: A_TL + A_TR
    # t=5: A_BL - A_TL
    # t=6: A_TR - A_BR
    
    # For each, the 2x2 matrix has entries at positions (0,0),(0,1),(1,0),(1,1)
    # Each position maps to a 2x2 block.
    
    # Let me define a11[t][pos] where pos in [(0,0),(0,1),(1,0),(1,1)]
    # and a11[t][pos] is a dict mapping (block_r, block_c, local_r, local_c) -> coeff
    
    # Actually, simpler: for top-level term t, the A-side 2x2 matrix element at
    # local position (lr, lc) is a sum of A[2*br + lr, 2*bc + lc] for various (br, bc).
    
    # Let me define:
    # a_blocks[t][lr][lc] = list of (br, bc, coeff)
    # meaning A_side[t][lr][lc] = sum over (br,bc,coeff) of A[2*br+lr, 2*bc+lc] * coeff
    
    a_blocks = [
        # t=0: A_TL + A_BR
        [[[(0,0,1.0)], [(0,0,1.0)]],   # (0,0): A_TL[0,0], (0,1): A_TL[0,1]
         [[(1,1,1.0)], [(1,1,1.0)]]],  # (1,0): A_BR[0,0], (1,1): A_BR[0,1]
        # Wait, this isn't right. Let me re-think.
    ]
    
    # I think I'm overcomplicating this. Let me use a different approach.
    # 
    # For each of the 49 terms, I'll directly compute which A[i,j], B[j,k] entries
    # are involved and with what coefficients, and which C[i,k] entries receive the result.
    #
    # Let me define the top-level Strassen more carefully.
    #
    # Top-level term t:
    #   A_t is a 2x2 matrix, B_t is a 2x2 matrix, and M_t = A_t * B_t is a 2x2 matrix.
    #   M_t contributes to C via: C += c_coeff[t] * M_t placed in specific blocks.
    #
    # Let me define:
    #   a_t[lr][lc] = linear combination of A entries
    #   b_t[lr][lc] = linear combination of B entries
    #   c_t[cr][cc] = which block of C receives M_t, with what sign
    #
    # For t=0 (M1 = (A_TL+A_BR)*(B_TL+B_BR)):
    #   a_t[0][0] = A[0,0] + A[2,2]
    #   a_t[0][1] = A[0,1] + A[2,3]
    #   a_t[1][0] = A[1,0] + A[3,2]
    #   a_t[1][1] = A[1,1] + A[3,3]
    #   b_t[0][0] = B[0,0] + B[2,2]
    #   b_t[0][1] = B[0,1] + B[2,3]
    #   b_t[1][0] = B[1,0] + B[3,2]
    #   b_t[1][1] = B[1,1] + B[3,3]
    #   c_t: M1 contributes +1 to C_TL and +1 to C_BR
    #     So C[0,0] += M1[0,0], C[0,1] += M1[0,1], C[1,0] += M1[1,0], C[1,1] += M1[1,1]
    #     C[2,2] += M1[0,0], C[2,3] += M1[0,1], C[3,2] += M1[1,0], C[3,3] += M1[1,1]
    
    # Let me code this up directly.
    
    # Define top-level terms:
    # Each term is (a_def, b_def, c_def)
    # a_def[lr][lc] = list of (i, j, coeff) for A[i,j]
    # b_def[lr][lc] = list of (j, k, coeff) for B[j,k]
    # c_def[cr][cc] = list of (i, k, coeff) for C[i,k]
    
    def make_a_t0():
        # A_TL + A_BR
        return [
            [[(0,0,1.0), (2,2,1.0)], [(0,1,1.0), (2,3,1.0)]],
            [[(1,0,1.0), (3,2,1.0)], [(1,1,1.0), (3,3,1.0)]],
        ]
    
    def make_b_t0():
        # B_TL + B_BR
        return [
            [[(0,0,1.0), (2,2,1.0)], [(0,1,1.0), (2,3,1.0)]],
            [[(1,0,1.0), (3,2,1.0)], [(1,1,1.0), (3,3,1.0)]],
        ]
    
    def make_c_t0():
        # M1 -> C_TL (+1) and C_BR (+1)
        return [
            [[(0,0,1.0), (2,2,1.0)], [(0,1,1.0), (2,3,1.0)]],
            [[(1,0,1.0), (3,2,1.0)], [(1,1,1.0), (3,3,1.0)]],
        ]
    
    def make_a_t1():
        # A_BL + A_BR
        return [
            [[(2,0,1.0), (2,2,1.0)], [(2,1,1.0), (2,3,1.0)]],
            [[(3,0,1.0), (3,2,1.0)], [(3,1,1.0), (3,3,1.0)]],
        ]
    
    def make_b_t1():
        # B_TL
        return [
            [[(0,0,1.0)], [(0,1,1.0)]],
            [[(1,0,1.0)], [(1,1,1.0)]],
        ]
    
    def make_c_t1():
        # M2 -> C_BL (+1) and C_BR (-1)
        return [
            [[(2,0,1.0)], [(2,1,1.0)]],
            [[(3,0,1.0)], [(3,1,1.0)]],
            # and C_BR gets -M2:
        ]
    
    # This is getting unwieldy. Let me just hardcode the 49 terms by computing them.
    # 
    # Actually, let me take a cleaner approach. I'll define the top-level Strassen
    # in terms of block operations, and the bottom-level Strassen in terms of scalar operations.
    # Then I'll combine them.
    
    # Top-level: 7 terms, each is a product of two 2x2 block-matrices.
    # Bottom-level: for each 2x2 product, 7 scalar products.
    
    # Let me define the top-level terms as:
    # t: (a_block_coeffs, b_block_coeffs, c_block_coeffs)
    # where *_block_coeffs is a 2x2 array of dicts mapping block_index -> coeff
    # block_index is (br, bc) for A/B, or (cr, cc) for C
    
    # A-side block coefficients for each top-level term:
    # t=0: A_TL + A_BR -> {(0,0): 1, (1,1): 1} for all positions
    # t=1: A_BL + A_BR -> {(1,0): 1, (1,1): 1}
    # t=2: A_TL -> {(0,0): 1}
    # t=3: A_BR -> {(1,1): 1}
    # t=4: A_TL + A_TR -> {(0,0): 1, (0,1): 1}
    # t=5: A_BL - A_TL -> {(1,0): 1, (0,0): -1}
    # t=6: A_TR - A_BR -> {(0,1): 1, (1,1): -1}
    
    # B-side block coefficients:
    # t=0: B_TL + B_BR -> {(0,0): 1, (1,1): 1}
    # t=1: B_TL -> {(0,0): 1}
    # t=2: B_TR - B_BR -> {(0,1): 1, (1,1): -1}
    # t=3: B_BL - B_TL -> {(1,0): 1, (0,0): -1}
    # t=4: B_BR -> {(1,1): 1}
    # t=5: B_TL + B_TR -> {(0,0): 1, (0,1): 1}
    # t=6: B_BL + B_BR -> {(1,0): 1, (1,1): 1}
    
    # C-side block coefficients (which blocks of C receive M_t, with what sign):
    # C_TL = M1 + M4 - M5 + M7
    # C_TR = M3 + M5
    # C_BL = M2 + M4
    # C_BR = M1 - M2 + M3 + M6
    
    # So:
    # t=0 (M1): C_TL +1, C_BR +1
    # t=1 (M2): C_BL +1, C_BR -1
    # t=2 (M3): C_TR +1, C_BR +1
    # t=3 (M4): C_TL +1, C_BL +1
    # t=4 (M5): C_TL -1, C_TR +1
    # t=5 (M6): C_BR +1
    # t=6 (M7): C_TL +1
    
    a_block_coeffs = [
        {(0,0): 1.0, (1,1): 1.0},  # t=0
        {(1,0): 1.0, (1,1): 1.0},  # t=1
        {(0,0): 1.0},               # t=2
        {(1,1): 1.0},               # t=3
        {(0,0): 1.0, (0,1): 1.0},  # t=4
        {(1,0): 1.0, (0,0): -1.0}, # t=5
        {(0,1): 1.0, (1,1): -1.0}, # t=6
    ]
    
    b_block_coeffs = [
        {(0,0): 1.0, (1,1): 1.0},  # t=0
        {(0,0): 1.0},               # t=1
        {(0,1): 1.0, (1,1): -1.0}, # t=2
        {(1,0): 1.0, (0,0): -1.0}, # t=3
        {(1,1): 1.0},               # t=4
        {(0,0): 1.0, (0,1): 1.0},  # t=5
        {(1,0): 1.0, (1,1): 1.0},  # t=6
    ]
    
    c_block_coeffs = [
        {(0,0): 1.0, (1,1): 1.0},  # t=0: C_TL +1, C_BR +1
        {(1,0): 1.0, (1,1): -1.0}, # t=1: C_BL +1, C_BR -1
        {(0,1): 1.0, (1,1): 1.0},  # t=2: C_TR +1, C_BR +1
        {(0,0): 1.0, (1,0): 1.0},  # t=3: C_TL +1, C_BL +1
        {(0,0): -1.0, (0,1): 1.0}, # t=4: C_TL -1, C_TR +1
        {(1,1): 1.0},               # t=5: C_BR +1
        {(0,0): 1.0},               # t=6: C_TL +1
    ]
    
    # Bottom-level Strassen: 7 terms
    # For a 2x2 product X*Y:
    # s=0: S1 = (x00+x11)(y00+y11) -> R[0,0]+=1, R[1,1]+=1
    # s=1: S2 = (x10+x11)y00 -> R[1,0]+=1
    # s=2: S3 = x00(y01-y11) -> R[0,1]+=1
    # s=3: S4 = x11(y10-y00) -> R[0,0]+=1, R[1,0]+=1
    # s=4: S5 = (x00+x01)y11 -> R[0,0]-=1, R[0,1]+=1
    # s=5: S6 = (x10-x00)(y00+y01) -> R[1,1]+=1
    # s=6: S7 = (x01-x11)(y10+y11) -> R[0,0]+=1, R[1,1]+=1
    
    # For each bottom-level term s:
    # x_coeffs[s][lr][lc] = list of (local_r, local_c, coeff) for X
    # y_coeffs[s][lr][lc] = list of (local_r, local_c, coeff) for Y
    # r_coeffs[s][lr][lc] = list of (local_r, local_c, coeff) for R
    
    x_coeffs = [
        [[(0,0,1.0)], [(1,1,1.0)]],  # s=0: x00+x11
        [[(1,0,1.0)], [(1,1,1.0)]],  # s=1: x10+x11
        [[(0,0,1.0)], []],           # s=2: x00
        [[], [(1,1,1.0)]],           # s=3: x11
        [[(0,0,1.0), (0,1,1.0)], []],# s=4: x00+x01
        [[(1,0,1.0), (0,0,-1.0)], []],# s=5: x10-x00
        [[(0,1,1.0), (1,1,-1.0)], []],# s=6: x01-x11
    ]
    
    y_coeffs = [
        [[(0,0,1.0)], [(1,1,1.0)]],  # s=0: y00+y11
        [[(0,0,1.0)], []],           # s=1: y00
        [[(0,1,1.0), (1,1,-1.0)], []],# s=2: y01-y11
        [[(1,0,1.0), (0,0,-1.0)], []],# s=3: y10-y00
        [[], [(1,1,1.0)]],           # s=4: y11
        [[(0,0,1.0), (0,1,1.0)], []],# s=5: y00+y01
        [[(1,0,1.0), (1,1,1.0)], []],# s=6: y10+y11
    ]
    
    r_coeffs = [
        [[(0,0,1.0)], [(1,1,1.0)]],  # s=0: R[0,0]+=1, R[1,1]+=1
        [[], [(1,0,1.0)]],           # s=1: R[1,0]+=1
        [[(0,1,1.0)], []],           # s=2: R[0,1]+=1
        [[(0,0,1.0)], [(1,0,1.0)]],  # s=3: R[0,0]+=1, R[1,0]+=1
        [[(0,0,-1.0), (0,1,1.0)], []],# s=4: R[0,0]-=1, R[0,1]+=1
        [[], [(1,1,1.0)]],           # s=5: R[1,1]+=1
        [[(0,0,1.0)], [(1,1,1.0)]],  # s=6: R[0,0]+=1, R[1,1]+=1
    ]
    
    # Now combine:
    # For top-level term t and bottom-level term s:
    # column c = t*7 + s
    # 
    # The A-side element at local position (lr, lc) is:
    #   sum over (br, bc, coeff_a) in a_block_coeffs[t] of A[2*br+lr, 2*bc+lc] * coeff_a
    # 
    # The B-side element at local position (lr, lc) is:
    #   sum over (br, bc, coeff_b) in b_block_coeffs[t] of B[2*br+lr, 2*bc+lc] * coeff_b
    # 
    # The result R[lr][lc] contributes to C:
    #   sum over (cr, cc, coeff_c) in c_block_coeffs[t] of C[2*cr+lr, 2*cc+lc] * coeff_c
    
    # For bottom-level term s:
    # X[lr][lc] = sum of x_coeffs[s][lr][lc]
    # Y[lr][lc] = sum of y_coeffs[s][lr][lc]
    # R[lr][lc] = sum of r_coeffs[s][lr][lc]
    
    # The scalar product for term (t,s) is:
    #   (sum of x_coeffs) * (sum of y_coeffs)
    #   = sum over all combinations of x and y coefficients
    # 
    # But in the tensor decomposition, each column c corresponds to ONE scalar product.
    # So I need to express the product as a single term in U, V, W.
    # 
    # Wait, that's the issue. The Strassen algorithm has linear combinations, so each
    # "scalar product" is actually a product of two linear combinations. In the tensor
    # decomposition, each column is a single product of one A-entry, one B-entry, and
    # one C-entry.
    # 
    # So I need to expand each Strassen term into individual products.
    # 
    # For example, S1 = (x00+x11)(y00+y11) = x00*y00 + x00*y11 + x11*y00 + x11*y11
    # This is 4 separate scalar products, not 1!
    # 
    # Oh no, that means the rank is not 49. The Strassen algorithm reduces the number of
    # multiplications, but each "multiplication" in Strassen is of linear combinations,
    # which when expanded into individual products gives more terms.
    # 
    # Wait, but the tensor rank is about the number of rank-1 tensors needed. A product
    # of two linear combinations is NOT a rank-1 tensor; it's a sum of rank-1 tensors.
    # 
    # So the 2-level Strassen does NOT give a rank-49 decomposition. It gives an algorithm
    # with 49 scalar multiplications, but the tensor rank could be higher.
    # 
    # Hmm, but the cell finding says "Using the 2-level Strassen algorithm reduces the rank
    # from 64 to 49". So maybe there's a way to make it work with rank 49?
    # 
    # Actually, I think the key insight is that in the tensor decomposition, each column c
    # corresponds to a rank-1 tensor, which is a product of three vectors. The Strassen
    # algorithm's "multiplications" are of the form (linear combo of A)(linear combo of B),
    # which is NOT a single rank-1 term.
    # 
    # So the 2-level Strassen gives an algorithm with 49 multiplications, but the tensor
    # rank is actually higher. The rank-49 decomposition would require a different approach.
    # 
    # Wait, I think I'm confusing algorithmic complexity with tensor rank. The tensor rank
    # of the 4x4 matrix multiplication tensor is known to be at most 49 (from the 2-level
    # Strassen), but this is because the Strassen algorithm can be expressed as a rank-49
    # decomposition.
    # 
    # Let me reconsider. In the Strassen algorithm, each "multiplication" is of the form:
    #   M = (sum_i a_i * A_i) * (sum_j b_j * B_j)
    # where A_i are entries of A and B_j are entries of B.
    # 
    # This can be written as:
    #   M = sum_{i,j} a_i * b_j * A_i * B_j
    # 
    # Each term a_i * b_j * A_i * B_j is a rank-1 contribution. So the total number of
    # rank-1 terms is the sum over all Strassen multiplications of (number of A terms) *
    # (number of B terms).
    # 
    # For S1 = (x00+x11)(y00+y11): 2 * 2 = 4 rank-1 terms.
    # For S2 = (x10+x11)y00: 2 * 1 = 2 rank-1 terms.
    # Etc.
    # 
    # So the total rank would be much higher than 49.
    # 
    # I think the cell finding might be referring to the number of scalar multiplications
    # in the algorithm, not the tensor rank. Or perhaps there's a clever way to pack
    # the linear combinations into the tensor decomposition.
    # 
    # Actually, wait. I think I've been misunderstanding the tensor decomposition format.
    # Let me re-read the task:
    # 
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # 
    # And the decomposition is: T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    # 
    # Each column c is a rank-1 tensor. The rank is the number of columns.
    # 
    # In the Strassen algorithm, each "multiplication" M = (sum a_i A_i)(sum b_j B_j)
    # contributes to the result as M * (some linear combo of C entries).
    # 
    # But M itself is a product of two linear combinations, which is NOT a single
    # rank-1 tensor. So to express the Strassen algorithm as a tensor decomposition,
    # I would need to expand each M into its constituent rank-1 terms.
    # 
    # Unless... the U, V, W matrices can have arbitrary real values, and the "multiplication"
    # in the Strassen sense is captured by the fact that U and V can have multiple non-zero
    # entries in the same column.
    # 
    # Wait, yes! That's the key. In the tensor decomposition, U[i,j,c] can be non-zero
    # for multiple (i,j) pairs in the same column c. Similarly for V and W. The product
    # U[i,j,c] * V[j,k,c] * W[k,i,c] for a fixed c is a rank-1 tensor, but the sum over c
    # can represent more complex structures.
    # 
    # But no, each c is still a single rank-1 tensor. The fact that U has multiple non-zero
    # entries in column c doesn't change that; it just means the rank-1 tensor has support
    # on multiple (i,j,k) indices.
    # 
    # So for S1 = (x00+x11)(y00+y11), I cannot represent this as a single column c. I
    # would need 4 columns, one for each combination.
    # 
    # Therefore, the 2-level Strassen does NOT give a rank-49 decomposition. The cell
    # finding must be referring to something else, or I'm misunderstanding.
    # 
    # Let me reconsider. Maybe the cell finding is about a different kind of decomposition
    # or a different notion of rank.
    # 
    # Actually, I just realized: maybe the "rank" in the cell finding refers to the number
    # of scalar multiplications in the algorithm, which is 49 for 2-level Strassen. And
    # the score improvement from 64 to 49 is because the score is based on the number of
    # multiplications, not the tensor rank.
    # 
    # But the task says "score is the rank", and the rank is the number of columns in U, V, W.
    # 
    # I'm confused. Let me just try to construct a valid decomposition with as few columns
    # as possible, and see what score I get.
    # 
    # For now, let me just return the schoolbook decomposition (rank 64) and see if the
    # program works. Then I can try to improve it.
    
    # Actually, let me just return the schoolbook for now and make sure the program is valid.
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
# EVOLVE-BLOCK-END
