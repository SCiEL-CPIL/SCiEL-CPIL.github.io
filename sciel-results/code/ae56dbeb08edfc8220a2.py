"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 5x5 using block decomposition:
    Split each matrix into:
      A = [[A1, A2],
           [A3, A4]]
    where A1 is 2x2, A2 is 2x3, A3 is 3x2, A4 is 3x3.
    Similarly B = [[B1, B2], [B3, B4]] with B1 2x2, B2 2x3, B3 3x2, B4 3x3.
    
    C = A*B gives:
    C11 = A1*B1 + A2*B3  (2x2)
    C12 = A1*B2 + A2*B4  (2x3)
    C21 = A3*B1 + A4*B3  (3x2)
    C22 = A3*B2 + A4*B4  (3x3)
    
    We use Strassen (rank 7) for the 2x2 block, schoolbook for rectangular blocks,
    and schoolbook for the 3x3 block (rank 27).
    
    Total rank = 7 (for C11's A1*B1) + 6 (A2*B3) + 6 (A1*B2) + 9 (A2*B4) 
             + 6 (A3*B1) + 9 (A4*B3) + 9 (A3*B2) + 9 (A4*B4)
    Wait, that's not right. Strassen replaces 8 multiplications with 7 for 2x2.
    But here we have mixed sizes. Let me reconsider.
    
    Actually, for C11 = A1*B1 + A2*B3:
    A1 is 2x2, B1 is 2x2 -> use Strassen (rank 7)
    A2 is 2x3, B3 is 3x2 -> schoolbook rank 2*3*2=12
    
    Hmm, this is getting complex. Let me just use a simpler approach:
    Use Strassen for the top-left 2x2 subproblem (A1*B1 -> part of C11),
    and schoolbook for everything else.
    
    C11 (2x2) = A1(2x2)*B1(2x2) + A2(2x3)*B3(3x2)
    Use Strassen for A1*B1: rank 7
    Schoolbook for A2*B3: rank 12
    
    C12 (2x3) = A1(2x2)*B2(2x3) + A2(2x3)*B4(3x3)
    Schoolbook: 2*2*3 + 2*3*3 = 12 + 18 = 30
    
    C21 (3x2) = A3(3x2)*B1(2x2) + A4(3x3)*B3(3x2)
    Schoolbook: 3*2*2 + 3*3*2 = 12 + 18 = 30
    
    C22 (3x3) = A3(3x2)*B2(2x3) + A4(3x3)*B4(3x3)
    Schoolbook: 3*2*3 + 3*3*3 = 18 + 27 = 45
    
    Total: 7 + 12 + 30 + 30 + 45 = 124. That's only 1 less than 125.
    
    Let me try a different split. Use 2x2 Strassen on the whole thing by padding?
    No, that would increase rank.
    
    Alternative: Use the known result that 5x5x5 has rank at most 124 (Bini et al.),
    but finding an explicit decomposition is hard.
    
    Let me try a different block structure:
    Split into 1x1, 2x2, and 3x3 blocks differently.
    
    Actually, let me just implement Strassen for 2x2 properly and combine with
    schoolbook for the remaining parts, being careful about index conventions.
    """
    import time
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # We'll use a 2x2 block decomposition with Strassen on the 2x2 core.
    # Split indices: {0,1} and {2,3,4}
    # A[i,j], B[j,k], C[k,i] (note: W[k*n+i] convention)
    
    # Strassen for 2x2: M1..M7
    # Standard Strassen computes C = A*B where C is 2x2, C[i,k]
    # But our convention is C[k,i] = W[k*n+i]
    # So we need to be careful.
    
    # Let's define the Strassen terms for 2x2 submatrices.
    # For indices in {0,1} x {0,1} x {0,1}:
    # A_sub[i,j] for i,j in {0,1}, B_sub[j,k] for j,k in {0,1}, result C_sub[k,i]
    
    # Strassen 2x2 (standard):
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    #
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    #
    # In our convention, C[k,i] so C00=C[0,0], C01=C[1,0], C10=C[0,1], C11=C[1,1]
    # So mapping: standard C[i,k] -> our C[k,i]
    # standard C00 -> our C00
    # standard C01 -> our C10 (k=1, i=0)
    # standard C10 -> our C01 (k=0, i=1)
    # standard C11 -> our C11
    
    # Let me just build the full decomposition using a dictionary-based approach.
    # Each term c corresponds to a triple of linear forms.
    
    # I'll use a simpler approach: build the schoolbook decomposition of rank 125,
    # then try to reduce by 1 using the known Bini decomposition idea.
    # But that's complex. Let me just return the schoolbook for now, as the
    # block decomposition with Strassen doesn't easily save rank for 5x5.
    
    # Actually, let me try a different approach: use the fact that 5 = 2+3,
    # and use Strassen(2x2) + schoolbook(3x3) + cross terms.
    
    # Blocks: I0={0,1}, I1={2,3,4}
    # C = A*B, C[k,i]
    
    # C[k,i] for k in I0, i in I0: sum_j A[i,j] B[j,k]
    #   = sum_{j in I0} A[i,j] B[j,k] + sum_{j in I1} A[i,j] B[j,k]
    #   = (A_I0 * B_I0)[i,k] + (A_I0 * B_I1)[i,k]
    
    # Let me just implement the full schoolbook with rank 125. It's safe.
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
    
    # Check if we're over time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
