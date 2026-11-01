import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x5) into B1 (2x5) and B2 (1x5)
    # C = A1 B1 + A2 B2
    
    # Part 1: A1 (2x2) * B1 (2x5)
    # Use Strassen for first 2 cols of B1, Schoolbook for last 3 cols
    # Strassen 2x2x2: 7 mults
    # Schoolbook 2x2x3: 12 mults
    # Total for Part 1: 19
    
    # Part 2: A2 (2x1) * B2 (1x5)
    # Schoolbook: 2 * 1 * 5 = 10 mults
    # Total for Part 2: 10
    
    # Total Rank: 29
    rank = 29
    U = [[0] * rank for _ in range(n * m)] # 6 x 29
    V = [[0] * rank for _ in range(m * p)] # 15 x 29
    W = [[0] * rank for _ in range(p * n)] # 10 x 29
    
    col = 0
    
    # --- Part 1: A1 (rows 0,1 of A; cols 0,1 of A) * B1 (rows 0,1 of B) ---
    # A1 indices in U: i*m + j for i in [0,1], j in [0,1]
    # B1 indices in V: j*p + k for j in [0,1], k in [0,4]
    # W indices: k*n + i for k in [0,4], i in [0,1]
    
    # Strassen for B1[:, 0:2] (cols 0,1 of B)
    # A1 is 2x2. B1_sub is 2x2.
    # A1: [[a11, a12], [a21, a22]]
    # B1_sub: [[b11, b12], [b21, b22]]
    
    # m1 = (a11+a22)(b11+b22)
    U[0][col] = 1; U[3][col] = 1 # a11, a22 (indices: 0*3+0=0, 1*3+1=4? No. i=0,j=0 -> 0. i=1,j=1 -> 4)
    # Wait, U index is i*m+j.
    # i=0, j=0 -> 0. i=0, j=1 -> 1.
    # i=1, j=0 -> 3. i=1, j=1 -> 4.
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[7][col] = 1 # b11 (0*5+0), b22 (1*5+1=6? No. j=1, k=1 -> 1*5+1=6)
    # B1_sub indices: j in [0,1], k in [0,1].
    # b11: j=0, k=0 -> 0.
    # b12: j=0, k=1 -> 1.
    # b21: j=1, k=0 -> 5.
    # b22: j=1, k=1 -> 6.
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[2][col] = 1; W[5][col] = 1; W[7][col] = 1 # k=0,i=0->0; k=0,i=1->2; k=1,i=0->5; k=1,i=1->7
    # C11 += m1, C12 += m1, C21 += m1, C22 += m1
    col += 1
    
    # m2 = (a21+a22)b11
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[2][col] = 1; W[7][col] = 1 # C21, C22
    col += 1
    
    # m3 = a11(b22-b11)
    U[0][col] = 1
    V[6][col] = 1; V[0][col] = -1
    W[5][col] = 1; W[7][col] = 1 # C12, C22?
    # C12 += m3, C22 += m3?
    # C12 = a11 b12 + a12 b22.
    # C22 = a21 b12 + a22 b22.
    # Strassen:
    # C11 = m1 - m3 - m4 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m5 - m6 + m7
    # My m3: a11(b22-b11).
    # C12 uses m3? C12 = a11 b12 + a12 b22.
    # m5 = (a11+a12)b21.
    # m3 is not directly in C12 standard Strassen?
    # Standard:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b22-b11)
    # p4 = a22(b11-b21)
    # p5 = (a11+a12)b21
    # p6 = (a21-a11)(b12+b11)
    # p7 = (a12-a22)(b21+b22)
    # C11 = p1 - p3 - p4 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p5 - p6 + p7
    
    # So m3 (p3) goes to C12 and C22?
    # C12 = p3 + p5.
    # C22 = p1 - p5 - p6 + p7.
    # p3 is in C12. Is it in C22? No.
    # Wait, C12 = a11 b12 + a12 b22.
    # p3 = a11(b22-b11).
    # p5 = (a11+a12)b21.
    # p3+p5 = a11 b22 - a11 b11 + a11 b21 + a12 b21.
    # This does not look like C12.
    # Ah, the indices for Strassen on 2x2x2 are local.
    # Here B1 is 2x5. We are applying Strassen to the first 2 columns.
    # So B1_sub is 2x2.
    # The result is 2x2.
    # C11 (of the 2x2 block) = a11 b11 + a12 b21.
    # C12 (of the 2x2 block) = a11 b12 + a12 b22.
    # C21 (of the 2x2 block) = a21 b11 + a22 b21.
    # C22 (of the 2x2 block) = a21 b12 + a22 b22.
    
    # Let's just use Schoolbook for 2x2x2 to be safe? Rank 4.
    # 4 + 12 + 10 = 26?
    # 2x2x2 Schoolbook is 4.
    # 2x2x3 Schoolbook is 12.
    # Total Part 1: 16.
    # Part 2: 10.
    # Total: 26.
    
    # Can we do 2x2x5 in 16?
    # Yes, schoolbook is 2*2*5 = 20.
    # Strassen saves 1 mult for 2x2x2 (7 vs 4? No, 7 vs 4 is worse).
    # Strassen is 7, Schoolbook is 4.
    # So Schoolbook is better for 2x2x2!
    # Why is Strassen used? Because it composes.
    # But here we are just doing one block.
    # So use Schoolbook for 2x2x5.
    # Rank 20.
    # Total 20 + 10 = 30.
    # This is the original rank.
    
    # So we MUST use Strassen to get below 30?
    # Strassen 2x2x2 is 7. Schoolbook is 4.
    # 7 > 4.
    # So Strassen is worse for a single 2x2x2 block.
    
    # However, for 2x2x5:
    # Split B into 2x2 and 2x3.
    # 2x2x2: Strassen (7) vs Schoolbook (4).
    # 2x2x3: Schoolbook (12).
    # Total Strassen: 7 + 12 = 19.
    # Total Schoolbook: 4 + 12 = 16.
    # So Schoolbook is better (16 < 19).
    
    # So how to get 28?
    # Maybe 2x3x5 has a specific structure.
    # Let's stick to Rank 29?
    # If Schoolbook is 16 for 2x2x5, and 10 for 2x1x5, total is 26?
    # Wait.
    # A1 is 2x2. B1 is 2x5.
    # Schoolbook: sum_{j=1}^2 A1[:,j] B1[j,:].
    # j=1: 2x1 * 1x5 = 10 mults.
    # j=2: 2x1 * 1x5 = 10 mults.
    # Total 20.
    # My previous "16" was wrong.
    # 2x2x2 Schoolbook is 4.
    # 2x2x3 Schoolbook is 12.
    # 4 + 12 = 16?
    # 2x2x2: 4 mults.
    # 2x2x3: 12 mults.
    # Total 16.
    # Why did I say 20?
    # 2x2x5: 2*2*5 = 20.
    # 4 + 12 = 16.
    # So Schoolbook for 2x2x5 is 16.
    # Part 2: 10.
    # Total 26.
    
    # Is 26 possible?
    # Lower bound for 2x3x5 is 28?
    # If lower bound is 28, then 26 is impossible.
    # Where is the error?
    # 2x2x2 Schoolbook is 4.
    # 2x2x3 Schoolbook is 12.
    # Total 16.
    # But 2x2x5 is not 2x2x2 + 2x2x3 in terms of rank additivity?
    # Yes it is. The terms are disjoint.
    # So rank(2x2x5) <= 16.
    # But mu(2,2,5) is known to be 19?
    # No, mu(2,2,5) <= 16.
    # Actually, mu(2,2,5) = 19 is a bound?
    # Let's check.
    # If mu(2,2,5) = 16, then mu(2,3,5) <= 16 + 10 = 26.
    # But the task says "Find a rank-minimal...".
    # And the seed is 30.
    # If 26 is possible, the answer is 26.
    # But I suspect my rank calculation for 2x2x5 is wrong.
    # 2x2x2: 4 mults.
    # 2x2x3: 12 mults.
    # Total 16.
    # Is it that simple?
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # C13 = A11 B13 + A12 B23
    # C14 = A11 B14 + A12 B24
    # C15 = A11 B15 + A12 B25
    # C23 = A21 B13 + A22 B23
    # C24 = A21 B14 + A22 B24
    # C25 = A21 B15 + A22 B25
    
    # Mults:
    # A11 B11, A12 B21 (2)
    # A11 B12, A12 B22 (2)
    # A21 B11, A22 B21 (2)
    # A21 B12, A22 B22 (2)
    # A11 B13, A12 B23 (2)
    # A11 B14, A12 B24 (2)
    # A11 B15, A12 B25 (2)
    # A21 B13, A22 B23 (2)
    # A21 B14, A22 B24 (2)
    # A21 B15, A22 B25 (2)
    # Total 20.
    # Why did I think 16?
    # 2x2x2 is 4.
    # 2x2x3 is 12.
    # 4+12=16.
    # But 2x2x5 has 10 entries.
    # 2x2x2 has 4 entries.
    # 2x2x3 has 6 entries.
    # 4+6=10.
    # So 16 mults for 10 entries?
    # Yes.
    # So mu(2,2,5) <= 16.
    # Then mu(2,3,5) <= 16 + 10 = 26.
    
    # I will implement Rank 26.
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x5)
    # Schoolbook
    for i in range(2):
        for j in range(2):
            for k in range(5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # 20 terms
    
    # Part 2: A2 (2x1) * B2 (1x5)
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1 # A[:,2] (index 2)
            V[2 * p + k][col] = 1 # B[2,:] (index 2)
            W[k * n + i][col] = 1
            col += 1
    # 10 terms
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
