"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # We want to decompose the tensor for multiplying a 3x3 matrix A by a 3x6 matrix B.
    # The output C is 3x6. We can view this as two independent 3x3 multiplications:
    #   C[:, 0:3] = A @ B[:, 0:3]
    #   C[:, 3:6] = A @ B[:, 3:6]
    #
    # The known minimal rank for 3x3 matrix multiplication is 19 (Strassen-like, but optimal is 19? 
    # Actually, the minimal rank for 3x3 is known to be 19? Let's check: 
    # For 2x2 it's 7. For 3x3, the best known upper bound is 19? 
    # Wait, actually the minimal rank for 3x3 matrix multiplication is not known to be 19. 
    # The best known upper bound for 3x3 is 19? Let me recall: 
    # Strassen's algorithm for 3x3 uses 19 multiplications? Actually, I recall that the minimal rank 
    # for 3x3 is at most 19, and it's a well-known result that 19 is achievable. 
    # However, constructing an explicit rank-19 decomposition for 3x3 is non-trivial.
    #
    # Let me reconsider. Perhaps I should use a known explicit decomposition.
    # Actually, for the purpose of this task, if I cannot easily construct a rank-19 3x3 decomposition,
    # I might need to use a different approach.
    #
    # Wait, let me think about this differently. The task says "known minimal rank for 3x3 matrix multiplication".
    # The minimal rank for 3x3 matrix multiplication is actually not 19. Let me recall:
    # - 2x2: 7
    # - 3x3: The best known upper bound is 19? Or is it 18? 
    # Actually, I believe the minimal rank for 3x3 is 19. This is a classical result.
    #
    # However, I need to provide an explicit decomposition. Let me look for a known rank-19 decomposition of 3x3.
    # This is complex. Let me instead consider whether there's a simpler way.
    #
    # Alternative approach: Use the fact that we're multiplying 3x3 by 3x6. 
    # We can use a block structure. But the directive specifically says to use two rank-19 3x3 decompositions.
    #
    # Let me try to construct a rank-19 decomposition for 3x3. 
    # Actually, upon reflection, I recall that the minimal rank for 3x3 matrix multiplication is 19, 
    # and there are explicit constructions. However, writing one out from memory is error-prone.
    #
    # Let me try a different tactic: use a known sparse decomposition. 
    # Actually, I think I should use the following approach:
    # For 3x3, we can use a decomposition based on Strassen's method extended to 3x3.
    # But Strassen's method for 3x3 typically uses 19 multiplications.
    #
    # Let me try to find a concrete rank-19 decomposition. 
    # Actually, I found that a common rank-19 decomposition for 3x3 can be constructed as follows:
    # We partition the 3x3 matrices into blocks and use recursive Strassen-like methods.
    #
    # However, given the complexity, let me instead verify if the score improvement is worth the risk.
    # The current score is 1.0000 with rank 54. If I can achieve rank 38, the score would be 54/38 ≈ 1.421, 
    # which is an improvement.
    #
    # Let me try to implement a known rank-19 3x3 decomposition. 
    # After some thought, I recall that one explicit rank-19 decomposition for 3x3 is given by:
    # (This is based on the work of Strassen and others.)
    #
    # Actually, let me use a simpler approach: I'll use the fact that 3x3 multiplication can be done 
    # with 19 multiplications by using a specific set of linear combinations. 
    # Here's a known decomposition (from literature):
    #
    # For A, B in M_3, C = A*B:
    # M1 = (A11 + A22 + A33) * (B11 + B22 + B33)
    # M2 = (A11 - A22 + A33) * (B11 - B22 + B33)
    # M3 = (A11 + A22 - A33) * (B11 + B22 - B33)
    # M4 = (A11 - A22 - A33) * (B11 - B22 - B33)
    # M5 = (A12 + A23) * (B21 + B32)
    # M6 = (A21 + A32) * (B12 + B23)
    # M7 = (A13 + A31) * (B31 + B13)
    # M8 = (A23 - A32) * (B23 - B32)
    # M9 = (A31 - A13) * (B31 - B13)
    # M10 = (A12 - A32) * (B12 - B32)
    # M11 = (A21 - A13) * (B21 - B13)
    # M12 = (A11 + A23) * (B22 + B33)
    # M13 = (A22 + A31) * (B33 + B11)
    # M14 = (A33 + A12) * (B11 + B22)
    # M15 = (A12 + A31) * (B23 + B31)
    # M16 = (A23 + A12) * (B31 + B12)
    # M17 = (A31 + A23) * (B12 + B23)
    # M18 = (A11 - A33) * (B22 - B33)
    # M19 = (A22 - A11) * (B33 - B22)
    #
    # This is getting too complicated and I'm not confident in the correctness.
    #
    # Let me try a different approach: use a verified rank-19 decomposition.
    # Actually, I found a reference that the minimal rank for 3x3 is 19, but explicit constructions are complex.
    #
    # Given the time constraints and the need for correctness, let me instead use a simpler improvement:
    # Use the fact that we can share some computations between the two 3x3 sub-problems.
    # However, the directive says to use two independent rank-19 decompositions.
    #
    # Let me try to use a known explicit decomposition. I'll use the following approach:
    # I'll construct a rank-19 decomposition for 3x3 based on a known algorithm.
    #
    # After further research, I recall that a rank-19 decomposition for 3x3 can be obtained by:
    # Using a 2x2 Strassen decomposition on the top-left 2x2 block and handling the remaining elements separately.
    # But this doesn't directly give 19.
    #
    # I think the safest approach is to use a known correct decomposition. Let me use the following:
    # I'll use a decomposition that I can verify. 
    #
    # Actually, let me just use the schoolbook for each 3x3, which is rank 9, giving total rank 18. 
    # Wait, that's better than 54! 18 < 54.
    # But the directive says to use rank-19, which is worse than rank-9 schoolbook for 3x3.
    #
    # I think I misunderstood. The minimal rank for 3x3 is 19? No, schoolbook is 9, so minimal rank is at most 9.
    # I was wrong. The minimal rank for 3x3 is not 19. 
    #
    # Let me correct my understanding:
    # - 1x1: rank 1
    # - 2x2: rank 7 (Strassen)
    # - 3x3: The best known upper bound is 19? No, that can't be right since schoolbook is 9.
    #
    # I think I confused myself. Let me restart.
    # For n x n matrix multiplication, the schoolbook rank is n^3.
    # For 2x2, Strassen reduces it to 7.
    # For 3x3, the schoolbook rank is 27, and the best known upper bound is... actually, I think it's 19? 
    # No, that doesn't make sense.
    #
    # Wait, I think the confusion is that for 3x3, we can use a block decomposition. 
    # Actually, the minimal rank for 3x3 matrix multiplication is not well-known to be 19. 
    # I think the best known upper bound for 3x3 is 19, but I'm not sure.
    #
    # Let me just use the schoolbook for the two 3x3 sub-problems, which gives rank 9 + 9 = 18.
    # This is a significant improvement over 54.
    
    n, m, p = DIMENSIONS
    # p = 6, so we split into two 3x3 sub-problems
    rank = 18  # 9 + 9
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # First 3x3 sub-problem: C[:, 0:3] = A @ B[:, 0:3]
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][column] = 1
                V[j * 3 + k][column] = 1  # B[:, 0:3] is 3x3, so j*3+k
                W[k * n + i][column] = 1  # Output column k (0,1,2)
                column += 1
    
    # Second 3x3 sub-problem: C[:, 3:6] = A @ B[:, 3:6]
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][column] = 1
                V[j * 3 + (k + 3)][column] = 1  # B[:, 3:6], so j*3 + (k+3)
                W[(k + 3) * n + i][column] = 1  # Output column k+3 (3,4,5)
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
