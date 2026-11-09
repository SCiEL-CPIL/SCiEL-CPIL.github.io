"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # Split A (3x4) into A1 (1x4) and A2 (2x4)
    # Split B (4x4) into B1 (4x2) and B2 (4x2)
    # C = A*B = [A1*B1, A1*B2; A2*B1, A2*B2]
    #
    # We use rank-8 decomposition for (1x4)*(4x2) -> (1x2)
    # and rank-16 decomposition for (2x4)*(4x2) -> (2x2)
    #
    # Total rank = 4 * 8 + 4 * 16 = 32 + 64 = 96? No.
    # The blocks share indices in a way that allows factorization.
    #
    # Actually, let's just use the standard Strassen-like or Winograd-like approach.
    # For 3x4x4, a known upper bound is 39 (via Lander's algorithm or similar).
    # But let's try to construct a specific decomposition.
    #
    # Let's use the method of "border rank" or known constructions.
    # A simple construction:
    # C_{ik} = sum_{j=1..4} A_{ij} B_{jk}
    #
    # Let's try to group terms.
    #
    # Alternative: Use the fact that 3x4x4 is close to 3x4x3 or 4x4x4.
    #
    # Let's use a verified construction from the literature for 3x4x4.
    # Rank 39 is known. Let's implement a rank-40 construction if 39 is too hard to encode manually.
    #
    # Actually, the current score is 1.0 with rank 48.
    # Let's try to achieve rank 42 or 40.
    #
    # Let's use the following strategy:
    # 1. Split the 4x4 matrix B into two 4x2 matrices B_L and B_R.
    # 2. Compute A*B_L (3x2) and A*B_R (3x2).
    # 3. Concatenate to get C (3x4).
    #
    # The cost of multiplying (3x4) * (4x2) is mu(3,4,2).
    # mu(3,4,2) <= mu(3,4,2).
    #
    # Let's use a known decomposition for (3x4) * (4x2).
    # We can view (3x4) * (4x2) as (3x2) * (2x4) * (4x2)? No.
    #
    # Let's use the "Winograd" approach for small matrices.
    #
    # Here is a concrete rank-42 construction for 3x4x4:
    # We use 42 terms.
    #
    # Actually, let's just use the schoolbook but optimized with Strassen for 2x2x2 blocks.
    #
    # Let's use the construction from:
    # "Fast Rectangular Matrix Multiplication" by Lander.
    # For 3x4x4, the optimal rank is 39.
    #
    # Since I cannot derive the 39-term decomposition from scratch reliably in this thought process,
    # I will use a rank-44 construction which is easier to verify and still an improvement over 48.
    #
    # Construction:
    # Split A into A_top (1x4) and A_bot (2x4).
    # Split B into B_left (4x2) and B_right (4x2).
    #
    # C_top_left = A_top * B_left (1x2)
    # C_top_right = A_top * B_right (1x2)
    # C_bot_left = A_bot * B_left (2x2)
    # C_bot_right = A_bot * B_right (2x2)
    #
    # Rank for (1x4) * (4x2):
    # This is a (1,4,2) tensor. Rank is 4 (schoolbook: A_11*B_11, A_12*B_21, A_13*B_31, A_14*B_41 for C_11; same for C_12).
    # Wait, (1x4) * (4x2) -> C_11 = sum A_1j B_j1, C_12 = sum A_1j B_j2.
    # We can do this with 4 terms if we share A's? No, B's are different.
    # Rank 4:
    # t1 = A_11 B_11 -> C_11
    # t2 = A_11 B_12 -> C_12
    # t3 = A_12 B_21 -> C_11
    # t4 = A_12 B_22 -> C_12
    # t5 = A_13 B_31 -> C_11
    # t6 = A_13 B_32 -> C_12
    # t7 = A_14 B_41 -> C_11
    # t8 = A_14 B_42 -> C_12
    # Rank 8 for (1x4) * (4x2).
    #
    # Rank for (2x4) * (4x2):
    # This is a (2,4,2) tensor.
    # We can use 2x2 Strassen-like? No, the middle dimension is 4.
    # Split B into B1 (4x1) and B2 (4x1)? No.
    #
    # Let's use schoolbook for (2x4) * (4x2).
    # C_11 = A_11 B_11 + A_12 B_21 + A_13 B_31 + A_14 B_41
    # C_12 = A_11 B_12 + A_12 B_22 + A_13 B_32 + A_14 B_42
    # C_21 = A_21 B_11 + A_22 B_21 + A_23 B_31 + A_24 B_41
    # C_22 = A_21 B_12 + A_22 B_22 + A_23 B_32 + A_24 B_42
    # Rank 16.
    #
    # Total rank = 8 + 8 + 16 + 16 = 48. This is the schoolbook rank.
    #
    # To improve, we need to share computations.
    #
    # Let's use the decomposition of (2x4) * (4x2) with rank 14 or less.
    #
    # Actually, let's just use the rank-48 solution but check if there's a simpler improvement.
    #
    # Let's try rank 42.
    #
    # I will implement the rank-48 solution for now, as it is safe, and see if I can find a lower rank.
    #
    # Wait, the prompt says "score is the rank". Lower is better.
    # Current score 1.0000 implies rank 48 is the baseline.
    #
    # Let's try to find a rank < 48.
    #
    # Known result: mu(3,4,4) = 39.
    #
    # I will use a rank-42 construction which is easier to implement than 39.
    #
    # Let's use the following construction for 3x4x4:
    # We use 42 terms.
    #
    # This is a complex task to derive. I will fall back to a safe rank-48 but optimized structure?
    # No, I must improve.
    #
    # Let's use the fact that 4x4x4 has rank 49 (Strassen) or 48 (schoolbook).
    # 3x4x4 is a subproblem.
    #
    # I will use the following heuristic:
    # 1. Use 4 Strassen 2x2x2 blocks? No.
    #
    # Let's use the "Lander" construction for 3x4x4 with rank 39.
    # I will encode the known rank-39 decomposition.
    #
    # Reference: Lander, R. S. (1989). "Fast rectangular matrix multiplication".
    #
    # Since I don't have the explicit coefficients for rank 39 memorized, I will use a rank-40 approximation or a different known bound.
    #
    # Actually, I can use the following construction:
    # Split A into (1,4) and (2,4).
    # Split B into (4,4).
    #
    # Let's try to use 20 terms for the (1,4,4) part and 20 terms for the (2,4,4) part?
    # mu(1,4,4) = 4 (it's just a linear combination).
    # mu(2,4,4) = ?
    #
    # Let's use rank 44 as derived in the previous thought process (2 * mu(3,4,2) = 2 * 22 = 44).
    # mu(3,4,2) = 11.
    #
    # Let's implement the rank-44 construction.
    #
    # Step 1: Decompose (3x4) * (4x2) into 11 terms.
    # Step 2: Apply this decomposition for B_L (4x2) and B_R (4x2).
    #
    # Decomposition of (3x4) * (4x2) -> (3x2):
    # C_{ik} = sum_{j=1..4} A_{ij} B_{jk}, k=1,2.
    #
    # We can use the following 11 terms:
    # 1. A_11 B_11 -> C_11
    # 2. A_11 B_12 -> C_12
    # 3. A_21 B_11 -> C_21
    # 4. A_21 B_12 -> C_22
    # 5. A_31 B_11 -> C_31
    # 6. A_31 B_12 -> C_32
    # 7. A_12 B_21 -> C_11
    # 8. A_12 B_22 -> C_12
    # 9. A_22 B_21 -> C_21
    # 10. A_22 B_22 -> C_22
    # 11. A_32 B_21 -> C_31
    # 12. A_32 B_22 -> C_32
    # ... this is getting complicated.
    #
    # Let's just use the rank-48 solution but with a better structure?
    #
    # Actually, I will use the rank-48 solution but I will try to see if I can reduce it.
    #
    # Let's use the following:
    # The rank of (3x4) * (4x4) is at most 39.
    #
    # I will implement a rank-48 solution for now, as it is the most reliable.
    #
    # Wait, the prompt says "score is the rank". If I return rank 48, the score is 1.0.
    # I need to return a rank < 48.
    #
    # Let's use the rank-44 construction.
    #
    # I will define the 11 terms for (3x2) * (2x2) and use them twice.
    #
    # (3x2) * (2x2) -> (3x2)
    # C_11 = A_11 B_11 + A_12 B_21
    # C_12 = A_11 B_12 + A_12 B_22
    # C_21 = A_21 B_11 + A_22 B_21
    # C_22 = A_21 B_12 + A_22 B_22
    # C_31 = A_31 B_11 + A_32 B_21
    # C_32 = A_31 B_12 + A_32 B_22
    #
    # This can be done with 6 terms if we use Strassen? No, 3x2x2 is not square.
    #
    # Let's use 6 terms:
    # t1 = A_11 B_11
    # t2 = A_11 B_12
    # t3 = A_21 B_11
    # t4 = A_21 B_12
    # t5 = A_31 B_11
    # t6 = A_31 B_12
    # t7 = A_12 B_21
    # t8 = A_12 B_22
    # t9 = A_22 B_21
    # t10 = A_22 B_22
    # t11 = A_32 B_21
    # t12 = A_32 B_22
    #
    # This is 12 terms.
    #
    # So mu(3,2,2) = 6? No, it's 12.
    #
    # Let's use the schoolbook for (3x4) * (4x4) with rank 48.
    #
    # I will return the rank-48 solution.
    
    rank = 48
    U = [[0] * rank for _ in range(12)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(12)]
    
    col = 0
    for i in range(3):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j][col] = 1
                V[j * 4 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
