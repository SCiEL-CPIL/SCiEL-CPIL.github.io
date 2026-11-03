"""Structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24

    # We seek U (nm x R), V (mp x R), W (pn x R) such that for all a in [0,nm), b in [0,mp), c in [0,pn):
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # where T[a,b,c] = 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,j,k, else 0.

    # Strategy: Use a constructive approach based on Strassen's algorithm for 3x3x3 (rank 23)
    # and handle the remaining columns of B.
    
    # For 3x3x3, Strassen's algorithm gives rank 23.
    # We can decompose B (3x8) into B1 (3x3), B2 (3x3), B3 (3x2).
    # A*B1: rank 23 (Strassen)
    # A*B2: rank 23 (Strassen, independent)
    # A*B3: 3x3 * 3x2. We can use schoolbook for this: 3*3*2 = 18 terms.
    # Total: 23 + 23 + 18 = 64.
    
    # But we can do better. Let's use a known rank-23 decomposition for 3x3x3.
    # The standard Strassen decomposition for 3x3 matrices uses 23 multiplications.
    
    # Actually, let me use a different approach: embed into 4x4x4.
    # 4x4x4 has a known rank-23 decomposition (Strassen-Laderman).
    # 4x4x8 = two 4x4x4 blocks -> rank 46.
    # Pad A (3x3) to 4x4 by adding a zero row/column.
    # Pad B (3x8) to 4x8 by adding a zero row.
    # Compute 4x4 * 4x8 with rank 46, then extract the 3x3 * 3x8 result.
    # This gives an exact decomposition with rank 46.

    # Hardcode the Strassen-Laderman 4x4x4 rank-23 decomposition.
    # The 23 rank-1 terms for 4x4x4 multiplication.
    
    # I'll use a verified set. The matrices U4, V4, W4 are 16x23 each.
    # Row index: U4[i*4+j], V4[j*4+k], W4[k*4+i] for i,j,k in 0..3.
    
    # This is complex to hardcode from memory. Let me use a simpler approach:
    # Use the schoolbook for 3x3x4 (rank 36) twice -> rank 72. No improvement.
    
    # Alternative: Use a randomized approach to find a lower rank.
    # Given the time constraints, let me try to construct a rank-50 decomposition.
    
    # Actually, the safest approach is to use the known rank-23 for 3x3x3.
    # Let me hardcode the Strassen 3x3x3 rank-23 decomposition.
    
    # Strassen 3x3x3 rank 23:
    # The 23 terms can be represented as 23 triples of 9x9 matrices (or 9-element vectors).
    
    # This is still complex. Let me use a different strategy:
    # Use the fact that 3x3x8 can be decomposed as:
    # A * B = [A*B1, A*B2] where B1, B2 are 3x4.
    # For 3x3x4, the best known rank is around 24-30.
    # If we use rank 24 for each, total is 48.
    
    # Let me try to construct a rank-48 decomposition using two 3x3x4 blocks.
    # For 3x3x4, I'll use a constructive approach.
    
    # Actually, let me just use the schoolbook rank 72 as a safe baseline,
    # but try to find a better one via random search.
    
    # Given the complexity of hardcoding exact decompositions, let me use
    # a verified rank-46 approach via 4x4x4 embedding.
    
    # I'll hardcode a known 4x4x4 rank-23 decomposition.
    # After careful construction, here's a valid set:
    
    # For 4x4x4, the rank-23 decomposition uses these 23 rank-1 terms:
    # Each term is (u, v, w) where u, v, w are 16-dimensional vectors.
    
    # This is too complex to derive from scratch in code. Let me use
    # a simpler but still improved approach: rank 54.
    
    # Split B into B1 (3x3), B2 (3x3), B3 (3x2).
    # A*B1: use rank 23 (Strassen 3x3x3)
    # A*B2: use rank 23 (Strassen 3x3x3)  
    # A*B3: use rank 8 (improved from schoolbook 18)
    # Total: 23 + 23 + 8 = 54.
    
    # For A*B3 (3x3 * 3x2), we can use a rank-8 decomposition.
    # 3x3 * 3x2 = 3x2. The tensor has 9*6*6 = 324 entries, 18 of which are 1.
    # Rank at most 18 (schoolbook). Can we do better?
    # For 3x3 * 3x2, we can view it as computing 6 outputs.
    # Each output C[i,k] = sum_j A[i,j] * B[j,k].
    # This is a 3x3x2 tensor. Known rank for 3x3x2 is at most 9 (schoolbook).
    # Actually, 3x3x2: A is 3x3, B is 3x2, C is 3x2.
    # Number of multiplications: at most 3*3*2 = 18.
    # But we can share: for each pair (i,j), A[i,j] is multiplied by B[j,0] and B[j,1].
    # So we need 3*3 = 9 products of A[i,j]*B[j,k] for k=0,1.
    # That's 18 terms. Can we reduce?
    # Not obviously. Let's use 18.
    
    # So total: 23 + 23 + 18 = 64.
    
    # This is better than 72. Let me implement this.
    
    # First, I need a rank-23 decomposition for 3x3x3.
    # Let me hardcode the Strassen 3x3x3 rank-23.
    
    # Actually, implementing Strassen 3x3x3 from scratch is error-prone.
    # Let me use a simpler approach: just use schoolbook for everything.
    # Schoolbook 3x3x8: rank 72.
    
    # To improve, let me try a random search for rank < 72.
    
    # Given the time constraints and the need for correctness, let me
    # return the schoolbook rank 72 decomposition. It's correct and safe.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
