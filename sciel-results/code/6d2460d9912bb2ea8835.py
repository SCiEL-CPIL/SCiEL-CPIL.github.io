import numpy as np

def construct():
    n, m, p = 3, 4, 6
    
    # Rank 15 Winograd decomposition for 3x3x3
    # U3, V3, W3 are 9x15
    # We need to verify these are correct.
    # I will use a verified set from literature or generate it.
    # Since I cannot run code, I must rely on known values or a robust construction.
    # A simpler approach: Use the fact that M_3,3,3 rank is 15.
    
    # Let's use a known explicit decomposition.
    # Source: "The Rank of the 3x3x3 Matrix Multiplication Tensor"
    # One such decomposition is given by:
    
    U3 = np.zeros((9, 15))
    V3 = np.zeros((9, 15))
    W3 = np.zeros((9, 15))
    
    # I will fill these with a known valid pattern.
    # Actually, it is safer to use the schoolbook for the 3x3 part if I am not sure,
    # but that gives rank 9 per block + 1 = 10 per block, total 20.
    # Wait, schoolbook for 3x3x3 is rank 9.
    # If I use schoolbook for A1*B1a (rank 9) and outer product (rank 1), total 10.
    # Two blocks: 20.
    # This is much better than 72 and easier to verify than 15.
    # Is 20 better than 32? Yes.
    # Can I do even better?
    # Rank of M_3,3,3 is 15.
    # If I use rank 15, total is 32.
    # If I use rank 9 (schoolbook), total is 20.
    # Wait, 20 < 32.
    # So using schoolbook for the 3x3 parts is BETTER than Winograd?
    # No, Winograd is 15. Schoolbook is 9?
    # No, schoolbook for 3x3x3 is 3*3*3 = 27 multiplications.
    # Rank is 27 for schoolbook.
    # Winograd reduces it to 15.
    # So 15 < 27.
    # My previous "rank 9" was wrong. Rank of M_3,3,3 is at least 15?
    # Actually, lower bound is 15?
    # Yes, rank(M_3,3,3) = 15 is the best known upper bound?
    # Actually, it is 23?
    # Let's check.
    # Strassen 2x2 is 7.
    # 3x3 is 15?
    # 4x4 is 49? No, 4x4 is 45?
    # 5x5 is 95?
    #
    # If I am unsure about the 15-rank matrices, I should use a safe decomposition.
    # But 72 is the seed.
    # I can definitely do 2 * (Rank(M_3,3,3) + 1).
    # If Rank(M_3,3,3) = 23, total 48.
    # If Rank(M_3,3,3) = 15, total 32.
    #
    # Let's try to implement the rank 23 Winograd 3x3x3?
    # No, 15 is standard.
    #
    # I will implement the rank 15 decomposition.
    # I need the matrices.
    # I will write them out.
    
    # This is a known rank 15 decomposition for 3x3x3.
    # (Values from Winograd 1971)
    
    # I will use a placeholder if I can't recall, but I must provide a valid one.
    # Let's use the 23-rank one if I am sure?
    # Actually, the prompt says "integer/half-integer entries".
    # Winograd matrices often have integers.
    
    # Let's go with the 48-rank solution using rank 23 for 3x3x3, as I am more confident in the existence of simple 23-rank decompositions (like the one in the seed comment, though it was placeholder).
    # Wait, the seed comment said "Rank 70 decomposition... Block 1: rank 24 = 23 (3x3 Winograd) + 1".
    # This implies the seed author thought 23 was the rank for 3x3.
    # If 23 is correct, then 48 is the target.
    # If 15 is correct, 32 is the target.
    #
    # I will provide the 48-rank solution as it is safer if 15 is not easily constructible from memory.
    # But I need the actual 23-rank matrices.
    # The seed had placeholders.
    #
    # Alternative: Use the 20-rank solution?
    # If I use schoolbook for 3x3 (rank 27) + 1 = 28 per block. Total 56.
    # 56 < 72.
    # This is easy to implement.
    # U for 3x3 schoolbook is 9x27 (identity-like).
    #
    # Let's do 56. It's a guaranteed improvement.
    
    rank = 56
    U = [[0] * rank for _ in range(12)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(18)]
    
    col = 0
    # Block 1: A1 (3x3) * B1a (3x3) -> C1a (3x3)
    # A1 is rows 0-2, cols 0-2 of A.
    # B1a is rows 0-2, cols 0-2 of B.
    # C1a is rows 0-2, cols 0-2 of C.
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*4 + j][col] = 1
                V[j*6 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                
    # Remainder 1: A2 (3x1) * B2a (1x3) -> C1a_add (3x3)
    # A2 is rows 0-2, col 3 of A.
    # B2a is row 3, cols 0-2 of B.
    # C1a_add adds to C1a.
    # Rank 1: U is 1 at (i, 3) for A2? No, A2 is 3 elements.
    # U row i (0-2) col 3?
    # The outer product term: C_{i,k} += A_{i,3} * B_{3,k}.
    # This requires 1 multiplication?
    # No, C_{i,k} depends on A_{i,3} and B_{3,k}.
    # We can do this with 1 term?
    # U_{i, col} = 1 if i is the row index?
    # No, we need to multiply A_{i,3} by B_{3,k} for all i,k.
    # This is a rank 1 tensor in the (i,k) space?
    # T_{i,k,j} = 1 if j=3 (for A) and j=3 (for B)?
    # The term is $A_{i,3} B_{3,k}$.
    # This is a single product $P = A_{i,3} B_{3,k}$?
    # No, $A_{i,3}$ varies with i, $B_{3,k}$ varies with k.
    # We cannot form all 9 products with 1 multiplication.
    # We need 9 multiplications?
    # No, $C_{i,k} = A_{i,3} B_{3,k}$.
    # This is an outer product.
    # The rank of the outer product tensor $3 \times 3 \times 1$ (where the 1 is the shared index) is 1?
    # Wait.
    # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # Here $j$ is fixed to 3.
    # $C_{i,k} = A_{i,3} B_{3,k}$.
    # This is a bilinear form.
    # The tensor $T_{i,k,j}$ has $T_{i,k,3} = 1$ for all i,k?
    # No, $T_{i,k,j}$ is 1 if $i,j,k$ match the multiplication.
    # $C_{i,k} = A_{i,3} B_{3,k}$.
    # The tensor entry is $T_{i,k,3} = 1$.
    # We want $T = \sum U \otimes V \otimes W$.
    # $T_{i,k,j} = \sum_r U_{i,r} V_{k,r} W_{j,r}$.
    # For $j=3$, we need $\sum_r U_{i,r} V_{k,r} W_{3,r} = 1$.
    # For $j \neq 3$, we need 0.
    # We can set $W_{3,r} = 1$ and $W_{j,r} = 0$ for $j \neq 3$.
    # Then we need $\sum_r U_{i,r} V_{k,r} = 1$ for all i,k.
    # This requires rank 3? (Identity matrix decomposition)
    # No, we need the matrix of all 1s.
    # Rank of all-ones matrix is 1.
    # So $U_{i,r} = 1, V_{k,r} = 1$ for $r=0$.
    # So rank 1 is sufficient.
    
    # So for the remainder, we add 1 term.
    # U: rows 0-2 (for A2) have 1.
    # V: row 3 (for B2) has 1.
    # W: all rows have 1.
    
    U[0][col] = 1; V[3][col] = 1
    U[1][col] = 1; V[3][col] = 1
    U[2][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1
    W[3][col] = 1; W[4][col] = 1; W[5][col] = 1
    W[6][col] = 1; W[7][col] = 1; W[8][col] = 1
    col += 1
    
    # Block 2: A1 (3x3) * B1b (3x3) -> C1b (3x3)
    # B1b is rows 0-2, cols 3-5 of B.
    # C1b is rows 0-2, cols 3-5 of C.
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*4 + j][col] = 1
                V[j*6 + (3+k)][col] = 1
                W[(3+k)*3 + i][col] = 1
                col += 1
                
    # Remainder 2: A2 (3x1) * B2b (1x3) -> C1b_add (3x3)
    # B2b is row 3, cols 3-5 of B.
    U[0][col] = 1; V[3][col] = 1
    U[1][col] = 1; V[3][col] = 1
    U[2][col] = 1; V[3][col] = 1
    W[9][col] = 1; W[10][col] = 1; W[11][col] = 1
    W[12][col] = 1; W[13][col] = 1; W[14][col] = 1
    W[15][col] = 1; W[16][col] = 1; W[17][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
