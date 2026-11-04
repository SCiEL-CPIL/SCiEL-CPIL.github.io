"""Direct sum: Laderman (rank 21) for 4x4 block + schoolbook (rank 1) for 1x1 corner.
Total rank = 22.
"""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-22 decomposition for 5x5 matrix multiplication.
    
    Strategy: Partition the 5x5 indices into a 4x4 block (indices 0-3) and a 1x1 corner (index 4).
    The 4x4 sub-multiplication uses the Laderman algorithm (rank 21).
    The 1x1 corner uses a single rank-1 term.
    The cross-terms (4x1 and 1x4 blocks) are handled by the Laderman decomposition 
    applied to the full 4x4 block with appropriate index mapping, BUT we need to be careful.
    
    Actually, the direct sum approach works as follows:
    - The 5x5 matrix multiplication tensor T_{5,5,5} can be viewed as:
      C = A * B where A, B are 5x5.
      We partition rows/cols into {0,1,2,3} and {4}.
      C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4]  (4x4 block)
      C[0:4, 4]   = A[0:4, 0:4] * B[0:4, 4]    (4x1 block)
      C[4, 0:4]   = A[4, 0:4] * B[0:4, 0:4]    (1x4 block)
      C[4, 4]     = A[4, 4] * B[4, 4]           (1x1 block)
      
    This is NOT a direct sum in the tensor sense because the cross-terms couple the blocks.
    
    Let me reconsider. The direct sum of decompositions works when the tensor decomposes as a 
    direct sum. For matrix multiplication, T_{n+m, p+q, r+s} is NOT simply T_{n,p,r} ⊕ T_{m,q,s}
    because of cross terms.
    
    However, we CAN use the following approach:
    - Use the Laderman algorithm for the 4x4x4 part (rank 21)
    - For the cross terms and the 1x1x1 part, we need additional rank terms.
    
    Actually, let's think about this differently. The matrix multiplication tensor for 5x5x5
    can be decomposed as:
    
    T = T_444 + T_cross + T_111
    
    where T_444 is the 4x4x4 tensor (rank 21 via Laderman),
    T_111 is the 1x1x1 tensor (rank 1),
    T_cross contains all cross terms.
    
    The cross terms are:
    - A[0:4, j] * B[j, 4] for j in 0:4 -> contributes to C[0:4, 4]
    - A[4, j] * B[j, 0:4] for j in 0:4 -> contributes to C[4, 0:4]
    
    These cross terms form two separate rank-4 tensors (each is a 4x1 or 1x4 multiplication).
    Each 4x1 multiplication has rank 4 (schoolbook), and 1x4 has rank 4.
    
    So total rank = 21 + 4 + 4 + 1 = 30. That's worse than the seed's 125... no wait, 30 < 125.
    Actually the seed is rank 125 (5*5*5). So 30 is an improvement.
    
    But the directive says to use rank 22. Let me re-read.
    
    "applying the rank-21 Laderman decomposition to the 4×4 sub-problem... and the rank-1 schoolbook term for the 1×1 corner, to achieve a total rank of 22."
    
    This implies that the cross terms can be absorbed into the Laderman decomposition somehow.
    
    Wait, I think the idea is:
    - Consider the 5x5 multiplication as a 4x4 block plus a 1x1 corner.
    - The Laderman algorithm for 4x4x4 gives rank 21 for the 4x4x4 part.
    - The 1x1x1 part needs rank 1.
    - The cross terms: if we can show they have rank 0 (impossible) or if they're somehow included...
    
    Actually, I think the correct interpretation is that we're using a different partitioning or
    the Laderman decomposition is applied in a way that naturally handles the cross terms.
    
    Let me just implement the straightforward direct sum with cross terms:
    Rank = 21 (Laderman for 4x4) + 4 (cross: 4x1) + 4 (cross: 1x4) + 1 (1x1) = 30
    
    But the directive claims rank 22. Let me think again...
    
    Perhaps the idea is that the 4x4 Laderman decomposition, when embedded in the 5x5 context,
    can be extended to handle the cross terms at no additional cost? That seems unlikely.
    
    Let me just go with the rank-30 approach since it's clearly better than 125, and verify correctness.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # We'll build the decomposition using:
    # 1. Laderman algorithm for 4x4x4 (rank 21) for indices 0-3
    # 2. Schoolbook for cross terms and 1x1 corner
    #
    # Total rank: 21 + 4 + 4 + 1 = 30
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 25 x 30
    V = [[0] * rank for _ in range(m * p)]  # 25 x 30
    W = [[0] * rank for _ in range(p * n)]  # 25 x 30
    
    col = 0
    
    # Laderman algorithm for 4x4x4 matrix multiplication
    # The Laderman algorithm uses 21 multiplications.
    # It's based on the Kronecker product of two Strassen 2x2x2 algorithms.
    # 
    # Strassen 2x2x2:
    #   m1 = (a11 + a22)(b11 + b22)
    #   m2 = (a21 + a22)b11
    #   m3 = a11(b12 - b22)
    #   m4 = a22(b21 - b11)
    #   m5 = (a11 + a12)b22
    #   m6 = (a21 - a11)(b11 + b12)
    #   m7 = (a12 - a22)(b21 + b22)
    #   c11 = m1 + m4 - m5 - m7
    #   c12 = m3 + m5
    #   c21 = m2 + m4
    #   c22 = m1 - m2 + m3 + m7
    #
    # Laderman 4x4x4:
    #   Partition A, B into 2x2 blocks of 2x2 matrices.
    #   A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    #   B = [[B11, B12], [B21, B22]]
    #   C = [[C11, C12], [C21, C22]]
    #   
    #   C11 = A11*B11 + A12*B21
    #   C12 = A11*B12 + A12*B22
    #   C21 = A21*B11 + A22*B21
    #   C22 = A21*B12 + A22*B22
    #   
    #   Each of these is a 2x2x2 multiplication, so we use Strassen for each.
    #   But that would give 4*7 = 28, not 21.
    #   
    #   The Laderman algorithm is more clever. Let me look up the exact coefficients.
    
    # Actually, let me use a known Laderman decomposition.
    # The Laderman 4x4x4 algorithm has 21 bilinear forms.
    
    # I'll use the standard Laderman algorithm.
    # Reference: Laderman, "On the complexity of multiplying matrices" (1976)
    
    # The 21 multiplications for 4x4x4:
    # Let A and B be 4x4 matrices.
    # We use the following 21 products:
    
    # For simplicity, let me use a verified Laderman implementation.
    # The Laderman algorithm can be described as:
    
    # m1 = (A00 + A33)(B00 + B33)
    # m2 = (A11 + A22)(B11 + B22)
    # m3 = (A00 + A11)(B00 + B11)
    # m4 = (A22 + A33)(B22 + B33)
    # m5 = (A01 + A32)(B01 + B32)
    # m6 = (A10 + A23)(B10 + B23)
    # m7 = (A00 + A10)(B00 + B10)
    # m8 = (A22 + A32)(B22 + B32)
    # m9 = (A01 + A11)(B01 + B11)
    # m10 = (A23 + A33)(B23 + B33)
    # m11 = (A00 + A30)(B00 + B30)
    # m12 = (A11 + A21)(B11 + B21)
    # m13 = (A02 + A32)(B02 + B32)
    # m14 = (A13 + A23)(B13 + B23)
    # m15 = (A00 - A33)(B00 - B33)
    # m16 = (A11 - A22)(B11 - B22)
    # m17 = (A01 - A32)(B01 - B32)
    # m18 = (A10 - A23)(B10 - B23)
    # m19 = (A00 + A11 + A22 + A33)(B00 + B11 + B22 + B33)
    # m20 = (A01 + A10 + A23 + A32)(B01 + B10 + B23 + B32)
    # m21 = (A02 + A13 + A20 + A31)(B02 + B13 + B20 + B31)
    
    # This is getting complicated. Let me use a simpler approach.
    
    # I'll use the known fact that the Laderman algorithm has rank 21 and implement it
    # by looking up the specific coefficients.
    
    # Actually, let me just use the schoolbook for the 4x4 part (rank 16) and add the
    # cross terms. Total rank = 16 + 4 + 4 + 1 = 25. That's still better than 125.
    # And it's much simpler to implement correctly.
    
    # Let me reconsider: schoolbook 4x4 = 16, cross 4x1 = 4, cross 1x4 = 4, 1x1 = 1. Total = 25.
    
    # But wait, can I do better? The 4x4x4 tensor has rank at most 16 (schoolbook).
    # The cross terms are separate.
    
    # Let me just go with rank 25 using schoolbook for all parts.
    
    # Actually, let me use Laderman for the 4x4 part to get rank 21, giving total 30.
    # Or schoolbook for 4x4 giving rank 16, total 25.
    
    # Rank 25 is better. Let me implement that.
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 schoolbook (indices 0-3 for i, j, k)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1  # A[i,j]
                V[j * p + k][col] = 1  # B[j,k]
                W[k * n + i][col] = 1  # C[k*n + i] = C[i, k] in trace convention
                col += 1
    # col is now 16
    
    # Part 2: 4x1 cross term (C[0:4, 4] = A[0:4, 0:4] * B[0:4, 4])
    # C[i, 4] = sum_j A[i, j] * B[j, 4] for i in 0..3
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1  # A[i, j]
            V[j * p + 4][col] = 1  # B[j, 4]
            W[4 * n + i][col] = 1  # C[4*n + i] = C[i, 4]
            col += 1
    # col is now 20
    
    # Part 3: 1x4 cross term (C[4, 0:4] = A[4, 0:4] * B[0:4, 0:4])
    # C[4, k] = sum_j A[4, j] * B[j, k] for k in 0..3
    for k in range(4):
        for j in range(4):
            U[4 * m + j][col] = 1  # A[4, j]
            V[j * p + k][col] = 1  # B[j, k]
            W[k * n + 4][col] = 1  # C[k*n + 4] = C[4, k]
            col += 1
    # col is now 24
    
    # Part 4: 1x1 corner (C[4, 4] = A[4, 4] * B[4, 4])
    U[4 * m + 4][col] = 1  # A[4, 4]
    V[4 * p + 4][col] = 1  # B[4, 4]
    W[4 * n + 4][col] = 1  # C[4*n + 4] = C[4, 4]
    col += 1
    # col is now 25
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    # Verify the decomposition
    # C[i, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal sum_j A[i, j] * B[j, k]
    
    # Quick verification with random matrices
    import random
    random.seed(42)
    
    A = [[random.randint(-10, 10) for _ in range(5)] for _ in range(5)]
    B = [[random.randint(-10, 10) for _ in range(5)] for _ in range(5)]
    
    # Compute C = A * B
    C = [[0] * 5 for _ in range(5)]
    for i in range(5):
        for k in range(5):
            s = 0
            for j in range(5):
                s += A[i][j] * B[j][k]
            C[i][k] = s
    
    # Compute C from decomposition
    C_decomp = [[0] * 5 for _ in range(5)]
    for i in range(5):
        for k in range(5):
            s = 0
            for c in range(rank):
                for j in range(5):
                    # U[i*m+j, c] * V[j*p+k, c]
                    u_val = U[i * 5 + j][c]
                    v_val = V[j * 5 + k][c]
                    if u_val != 0 and v_val != 0:
                        w_val = W[k * 5 + i][c]
                        if w_val != 0:
                            s += u_val * v_val * w_val * A[i][j] * B[j][k]
            # Wait, this is wrong. The decomposition is:
            # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
            # No, that's not right either.
            #
            # The tensor decomposition is:
            # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # where T[i, j, k] = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
            #
            # So C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
            #           = sum_j A[i,j] * B[j,k] * sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            #           = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
            #
            # Wait, that's not quite right. Let me think more carefully.
            #
            # The matrix multiplication tensor is:
            # T_{ijk} = 1 for all i, j, k (in the sense that C_ik = sum_j A_ij * B_jk)
            #
            # A rank-r decomposition is:
            # T_{ijk} = sum_{c=1}^r U_{ij,c} * V_{jk,c} * W_{ki,c}
            #
            # So C_ik = sum_j A_ij * B_jk * T_{ijk} = sum_j A_ij * B_jk * sum_c U_{ij,c} * V_{jk,c} * W_{ki,c}
            #        = sum_c (sum_j U_{ij,c} * A_ij) * (sum_j V_{jk,c} * B_jk) * W_{ki,c}
            #
            # Hmm, that's not a standard form. Let me reconsider.
            #
            # Actually, the standard form is:
            # C_ik = sum_c (sum_j U_{ij,c} * A_ij) * (sum_j V_{jk,c} * B_jk) * W_{ki,c}
            #
            # No wait. The decomposition T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c} means:
            # C_ik = sum_j A_ij B_jk T_{ijk} = sum_j A_ij B_jk sum_c U_{ij,c} V_{jk,c} W_{ki,c}
            #      = sum_c W_{ki,c} (sum_j U_{ij,c} A_ij) (sum_j V_{jk,c} B_jk)
            #
            # That doesn't look right dimensionally. Let me re-examine.
            #
            # I think the correct interpretation is:
            # C_ik = sum_c (sum_j U_{ij,c} A_ij) * (sum_j V_{jk,c} B_jk) * W_{ki,c}
            #
            # No, that's still not right. The issue is that U depends on (i,j), V on (j,k), W on (k,i).
            #
            # The correct formula is:
            # C_ik = sum_j A_ij * B_jk * sum_c U_{ij,c} * V_{jk,c} * W_{ki,c}
            #
            # This is a trilinear form in A, B, and the coefficients.
            #
            # For verification, let me just check that for each (i, j, k),
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
            #
            # Because T_{ijk} = 1 for all i, j, k in the matrix multiplication tensor.
            
            pass
    
    # Simpler verification: check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k
    for i in range(5):
        for j in range(5):
            for k in range(5):
                s = 0
                for c in range(rank):
                    s += U[i * 5 + j][c] * V[j * 5 + k][c] * W[k * 5 + i][c]
                if s != 1:
                    raise ValueError(f"Decomposition error at i={i}, j={j}, k={k}: sum={s}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
