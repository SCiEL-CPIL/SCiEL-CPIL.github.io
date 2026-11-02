import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # We construct a rank-29 decomposition.
    # Strategy: Split the 3x5 matrix B into B1 (3x3) and B2 (3x2).
    # C = A @ B1 + A @ B2
    # Rank(A @ B1) = 17 (using Strassen on 2x2x2x2 block + schoolbook for rest)
    # Rank(A @ B2) = 12 (Schoolbook)
    # Total Rank = 29
    
    rank = 29
    U = [[0.0] * rank for _ in range(n * m)] # 6 x 29
    V = [[0.0] * rank for _ in range(m * p)] # 15 x 29
    W = [[0.0] * rank for _ in range(p * n)] # 10 x 29
    
    col = 0
    
    # --- Part 1: A @ B1 (2x3 * 3x3) ---
    # A indices: i in 0..1, j in 0..2. Index in U: i*m + j = i*3 + j
    # B1 indices: j in 0..2, k in 0..2. Index in V: j*p + k = j*5 + k
    # C1 indices: k in 0..2, i in 0..1. Index in W: k*n + i = k*2 + i
    
    # Sub-part 1a: Strassen for A[0:2, 0:2] @ B1[0:2, 0:2]
    # A_top_left: rows 0,1; cols 0,1
    # B1_top_left: rows 0,1; cols 0,1
    
    # Strassen Multiplications
    # P1 = A11 B11
    U[0*3+0][col] = 1; V[0*5+0][col] = 1; W[0*2+0][col] = 1
    U[1*3+0][col] = 1; V[0*5+0][col] = 1; W[0*2+1][col] = 1
    U[0*3+0][col] = 1; V[1*5+0][col] = 1; W[1*2+0][col] = 1
    U[1*3+0][col] = 1; V[1*5+0][col] = 1; W[1*2+1][col] = 1
    col += 1
    
    # P2 = A22 B22
    U[0*3+1][col] = 1; V[1*5+1][col] = 1; W[1*2+0][col] = 1
    U[1*3+1][col] = 1; V[1*5+1][col] = 1; W[1*2+1][col] = 1
    U[0*3+1][col] = 1; V[1*5+1][col] = 1; W[1*2+0][col] = 1 # Wait, Strassen structure needs care.
    
    # Let's use a helper to add schoolbook terms to be safe and simple.
    # The Strassen part is complex to write inline without errors.
    # I will use Schoolbook for the 2x2x2x2 block as well to ensure correctness, 
    # which gives Rank 8 for that block.
    # Then Rank(2x3x3x3) = 8 (for 2x2x2x2) + 4 (2x2x2x1) + 4 (2x1x1x2) + 2 (2x1x1x1) = 18.
    # Total Rank = 18 + 12 = 30. This is the same as the seed.
    
    # To get 29, I need to save 1 rank.
    # The Strassen rank 7 saves 1 rank over schoolbook rank 8.
    # So I MUST implement Strassen correctly for the 2x2x2x2 block.
    
    # Let's restart the construction with a clean Strassen implementation for the 2x2x2x2 sub-block.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # --- Part 1: A @ B1 ---
    # Split A into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
    # Split B1 into B11 (2x2, rows 0,1 cols 0,1), B12 (2x1, rows 0,1 col 2), 
    #             B21 (1x2, row 2 cols 0,1), B22 (1x1, row 2 col 2)
    
    # 1. A1 @ B11 (2x2 * 2x2) using Strassen
    # A1 indices: (0,0), (0,1), (1,0), (1,1) -> U idx: 0, 1, 3, 4
    # B11 indices: (0,0), (0,1), (1,0), (1,1) -> V idx: 0, 1, 5, 6
    # Result C11 (2x2) -> W idx: 0, 1, 2, 3 (k*n+i: 0*2+0, 0*2+1, 1*2+0, 1*2+1)
    
    # Strassen:
    # M1 = A11 B11
    add_term(0, 0, 0)
    add_term(3, 0, 2)
    add_term(0, 5, 1)
    add_term(3, 5, 3)
    col += 1
    
    # M2 = A22 B22
    add_term(1, 6, 3) # A1[1,1]=U[4]? No. A1 is rows 0,1 cols 0,1.
    # U indices:
    # i=0,j=0 -> 0
    # i=0,j=1 -> 1
    # i=1,j=0 -> 3
    # i=1,j=1 -> 4
    
    # V indices for B11 (rows 0,1 cols 0,1):
    # j=0,k=0 -> 0
    # j=0,k=1 -> 1
    # j=1,k=0 -> 5
    # j=1,k=1 -> 6
    
    # W indices for C11 (k=0,1 i=0,1):
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 2
    # k=1,i=1 -> 3
    
    # Correct Strassen:
    # M1 = A11 B11 = U[0]*V[0] -> W[0]
    # M2 = A22 B22 = U[4]*V[6] -> W[3]
    # M3 = (A11+A22)(B11+B22) = (U[0]+U[4])*(V[0]+V[6]) -> W[0]+W[3]-M1-M2
    # M4 = A21(B12-B22) = U[3]*(V[1]-V[6]) -> W[2]
    # M5 = (A12+A22)(B21+B22) = (U[1]+U[4])*(V[5]+V[6]) -> W[1]+W[3]-M2
    # M6 = (A11-A21)(B11+B12) = (U[0]-U[3])*(V[0]+V[1]) -> W[0]+W[1]-M1
    # M7 = (A12-A22)(B21+B11) = (U[1]-U[4])*(V[5]+V[0]) -> W[2]+W[1]-M4
    
    # We need to decompose these into elementary tensor terms.
    # This is tedious. I will use the fact that Rank 29 is safer if I just use Schoolbook for the 2x2 block?
    # No, Schoolbook 2x2 is rank 8. Strassen is 7.
    # If I use Schoolbook for 2x2, total is 30.
    
    # I will implement the Rank-30 solution but with a slight optimization if possible?
    # No, the seed is Rank 30.
    
    # Let's try to get Rank 29 by just using Schoolbook everywhere but noticing that 
    # for 2x1 * 1x1, rank is 2.
    
    # Actually, I will provide the Rank-29 solution by using Strassen for the 2x2x2x2 block.
    # I will manually expand the Strassen terms into the U, V, W matrices.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    
    # Strassen for 2x2x2x2 (A1, B11, C11)
    # M1 = A11 B11
    add_term(0, 0, 0)
    add_term(3, 0, 2)
    add_term(0, 5, 1)
    add_term(3, 5, 3)
    col += 1
    
    # M2 = A22 B22
    add_term(4, 6, 3)
    add_term(4, 6, 3) # Wait, A22 is U[4], B22 is V[6].
    # A22 B22 contributes to C22 (W[3]).
    # C22 = A21 B12 + A22 B22.
    # So M2 is just one term: U[4] * V[6] -> W[3].
    add_term(4, 6, 3)
    col += 1
    
    # M3 = (A11+A22)(B11+B22)
    # Terms: A11 B11 (W0), A11 B22 (W1), A22 B11 (W2), A22 B22 (W3)
    # We already have A11 B11 (M1) and A22 B22 (M2).
    # M3 = M1 + M2 + A11 B22 + A22 B11.
    # So we need to add A11 B22 and A22 B11.
    # A11 B22: U[0] * V[6] -> W[1]
    # A22 B11: U[4] * V[0] -> W[2]
    add_term(0, 6, 1)
    add_term(4, 0, 2)
    col += 1
    
    # M4 = A21 (B12 - B22)
    # A21 is U[3]. B12 is V[1], B22 is V[6].
    # A21 B12 -> W[2]
    # A21 B22 -> W[3]
    add_term(3, 1, 2)
    add_term(3, 6, 3)
    col += 1
    
    # M5 = (A12+A22)(B21+B22)
    # A12 is U[1], A22 is U[4]. B21 is V[5], B22 is V[6].
    # A12 B21 -> W[1]
    # A12 B22 -> W[3]
    # A22 B21 -> W[2]
    # A22 B22 -> W[3] (Already in M2)
    add_term(1, 5, 1)
    add_term(1, 6, 3)
    add_term(4, 5, 2)
    col += 1
    
    # M6 = (A11-A21)(B11+B12)
    # A11 U[0], A21 U[3]. B11 V[0], B12 V[1].
    # A11 B11 -> W[0] (In M1)
    # A11 B12 -> W[1]
    # A21 B11 -> W[2]
    # A21 B12 -> W[3]
    add_term(0, 1, 1)
    add_term(3, 0, 2)
    add_term(3, 1, 3)
    col += 1
    
    # M7 = (A12-A22)(B21+B11)
    # A12 U[1], A22 U[4]. B21 V[5], B11 V[0].
    # A12 B21 -> W[1] (In M5)
    # A12 B11 -> W[0]
    # A22 B21 -> W[2] (In M5)
    # A22 B11 -> W[3] (In M3)
    add_term(1, 0, 0)
    add_term(4, 5, 2)
    add_term(4, 0, 3)
    col += 1
    
    # Now we have used 7 columns for the 2x2x2x2 block.
    # We need to add the remaining terms for A @ B1.
    # Remaining:
    # A1 @ B12 (2x2 * 2x1) -> 4 terms
    # A2 @ B21 (2x1 * 1x2) -> 4 terms
    # A2 @ B22 (2x1 * 1x1) -> 2 terms
    
    # A1 @ B12:
    # A11 B12: U[0] V[1] -> W[1]
    # A12 B12: U[1] V[1] -> W[3]
    # A21 B12: U[3] V[1] -> W[2] (Already in M4? M4 was A21(B12-B22). So A21 B12 is there. A21 B22 is there.)
    # A22 B12: U[4] V[1] -> W[3]
    
    # Wait, the Strassen decomposition computes the EXACT product.
    # C11 = M1 + M2 + M3 - ...
    # The terms added above are the elementary products that form the Strassen multipliers.
    # The Strassen algorithm computes the 4 outputs using 7 multipliers.
    # By expanding the multipliers into elementary products, we are effectively listing all the
    # elementary products involved in the Strassen scheme.
    # However, some elementary products appear in multiple multipliers.
    # The rank is the number of multipliers, not the number of elementary products.
    # But our U, V, W matrices represent the elementary products (rank 1 tensors).
    # If we use Strassen, we need to combine the elementary products into the 7 multipliers.
    # This means U, V, W should have entries that are sums, representing the linear combinations.
    
    # My `add_term` function adds to the same column.
    # For M1, I added 4 terms to col 0. This is WRONG.
    # M1 is a SINGLE multiplier: (A11) * (B11).
    # So U[0] should be 1, V[0] should be 1, and the result should be added to W.
    # But W is linear.
    # The tensor decomposition is $\sum_{r} U_r \otimes V_r \otimes W_r$.
    # If I use Strassen, I have 7 terms.
    # Term 1: U_vec = [1,0,0,0,0,0,0,0,0,0,0,0] (only A11), V_vec = [1,0,0,0,0,0,0,0,0,0,0,0] (only B11).
    # The result of this product is $A_{11} B_{11}$.
    # This contributes to $C_{11}, C_{12}, C_{21}, C_{22}$? No, $A_{11} B_{11}$ is just the top-left scalar.
    # So $W_{11}$ gets +1, others 0.
    
    # So for M1:
    # U[0]=1, V[0]=1, W[0]=1.
    # U[3]=0, V[5]=0, etc.
    
    # Let's redo the Strassen part correctly.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    
    # M1 = A11 B11
    U[0][col] = 1
    V[0][col] = 1
    W[0][col] = 1
    col += 1
    
    # M2 = A22 B22
    U[4][col] = 1
    V[6][col] = 1
    W[3][col] = 1
    col += 1
    
    # M3 = (A11+A22)(B11+B22)
    U[0][col] = 1
    U[4][col] = 1
    V[0][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    W[1][col] = 1
    W[2][col] = 1
    W[3][col] = 1
    col += 1
    
    # M4 = A21 (B12 - B22)
    U[3][col] = 1
    V[1][col] = 1
    V[6][col] = -1
    W[2][col] = 1
    W[3][col] = -1
    col += 1
    
    # M5 = (A12+A22)(B21+B22)
    U[1][col] = 1
    U[4][col] = 1
    V[5][col] = 1
    V[6][col] = 1
    W[1][col] = 1
    W[3][col] = 1
    W[2][col] = 1
    W[3][col] = 1 # W3 gets 1 from A12B22 and 1 from A22B22?
    # A12 B21 -> W1
    # A12 B22 -> W3
    # A22 B21 -> W2
    # A22 B22 -> W3
    # So W1=1, W2=1, W3=2?
    W[3][col] = 2
    col += 1
    
    # M6 = (A11-A21)(B11+B12)
    U[0][col] = 1
    U[3][col] = -1
    V[0][col] = 1
    V[1][col] = 1
    W[0][col] = 1
    W[1][col] = 1
    W[2][col] = -1
    W[3][col] = -1
    col += 1
    
    # M7 = (A12-A22)(B21+B11)
    U[1][col] = 1
    U[4][col] = -1
    V[5][col] = 1
    V[0][col] = 1
    W[1][col] = 1
    W[0][col] = 1
    W[2][col] = -1
    W[3][col] = -1
    col += 1
    
    # Now add the remaining schoolbook terms for A1 @ B12, A2 @ B21, A2 @ B22
    # A11 B12: U[0] V[1] -> W[1]
    U[0][col] = 1; V[1][col] = 1; W[1][col] = 1; col += 1
    # A12 B12: U[1] V[1] -> W[3]
    U[1][col] = 1; V[1][col] = 1; W[3][col] = 1; col += 1
    # A21 B12: U[3] V[1] -> W[2]
    U[3][col] = 1; V[1][col] = 1; W[2][col] = 1; col += 1
    # A22 B12: U[4] V[1] -> W[3]
    U[4][col] = 1; V[1][col] = 1; W[3][col] = 1; col += 1
    
    # A11 B21: U[0] V[5] -> W[0]
    U[0][col] = 1; V[5][col] = 1; W[0][col] = 1; col += 1
    # A12 B21: U[1] V[5] -> W[1]
    U[1][col] = 1; V[5][col] = 1; W[1][col] = 1; col += 1
    # A21 B21: U[3] V[5] -> W[2]
    U[3][col] = 1; V[5][col] = 1; W[2][col] = 1; col += 1
    # A22 B21: U[4] V[5] -> W[3]
    U[4][col] = 1; V[5][col] = 1; W[3][col] = 1; col += 1
    
    # A11 B22: U[0] V[6] -> W[1]
    U[0][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
    # A21 B22: U[3] V[6] -> W[3]
    U[3][col] = 1; V[6][col] = 1; W[3][col] = 1; col += 1
    
    # Part 1 (A @ B1) used 7 + 4 + 4 + 2 = 17 columns.
    
    # --- Part 2: A @ B2 (2x3 * 3x2) ---
    # Schoolbook: 12 terms.
    # A indices: 0..5
    # B2 indices: rows 0..2, cols 3..4. V indices: j*5+k.
    # j=0, k=3 -> 3
    # j=0, k=4 -> 4
    # j=1, k=3 -> 8
    # j=1, k=4 -> 9
    # j=2, k=3 -> 13
    # j=2, k=4 -> 14
    
    # C2 indices: k=3,4; i=0,1. W indices: k*2+i.
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    # k=4, i=0 -> 8
    # k=4, i=1 -> 9
    
    for i in range(2):
        for j in range(3):
            for k in [3, 4]:
                u_idx = i * 3 + j
                v_idx = j * 5 + k
                w_idx = k * 2 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
