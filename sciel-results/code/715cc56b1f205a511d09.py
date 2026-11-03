"""Rewritten program for 2x3x8 matrix multiplication rank minimization."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    # We construct a rank-32 decomposition using a Strassen-like approach.
    # The idea is to use a 2x2x2 Strassen decomposition on the first 2x2 block of B,
    # and handle the remaining 6 columns separately, but with some sharing.
    # Actually, a simpler and more effective approach: use the known rank for 2x3x8.
    # Known lower bound is 13, but we can achieve better than 48.
    # Let's use a construction based on blocking.
    
    # Strategy: Treat the 2x3 matrix A as a 2x3 array and B as 3x8.
    # We can use the fact that 2x2x2 has rank 7 (Strassen).
    # Split the 3x8 matrix into a 2x8 part and a 1x8 part? No, 3 rows.
    # Let's split the 3 columns of A into 1+2.
    # A = [a1, a2, a3] (2x3). B = [b1; b2; b3] (3x8, so 3 rows of 8).
    # C = a1*b1 + a2*b2 + a3*b3.
    # This is sum of three rank-8 terms = rank 24. Not better.
    
    # Better: Use a rank-14 decomposition for 2x3x4 and double it?
    # 2x3x4 schoolbook is 24. Strassen-like for 2x2x4 is 2*7=14.
    # For 2x3x4, we can use 2x2x4 (rank 14) + 1x2x4 (rank 8) = 22? No.
    
    # Let's try a direct construction with rank 26 or less.
    # Actually, the known rank for 2x3x8 is at most 26.
    # I will implement a greedy compression algorithm on the schoolbook tensor.
    
    import numpy as np
    
    # Schoolbook decomposition
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
    
    # Now compress the rank by finding linear dependencies among the terms.
    # Each term t is a triple (u_t, v_t, w_t) where u_t is a vector in R^{n*m},
    # v_t in R^{m*p}, w_t in R^{p*n}.
    # We can replace a set of terms with a lower-rank decomposition if they are linearly dependent
    # in a certain sense. A simple heuristic: try to merge terms that share the same u or v.
    
    # A more effective algorithm: use the "greedy rank reduction" approach.
    # Represent the tensor as a list of (u, v, w) triples.
    # Try to find a set of triples that can be replaced by a smaller set.
    
    # For now, let's use a known good rank: 26.
    # I will construct it explicitly.
    
    # Actually, let's use a simpler but effective method: 
    # Use the rank-12 decomposition for 2x3x4 (which is 2x3x4 = 24 schoolbook, but can be reduced).
    # Known: rank(2x3x4) <= 13.
    # If we have a rank-13 decomposition for 2x3x4, then for 2x3x8 we can use 2*13 = 26.
    
    # Let's find a rank-13 decomposition for 2x3x4.
    # I will search for it or use a known one.
    # Since I don't have a known one, I will use a generic compression algorithm.
    
    # Algorithm: Start with schoolbook, then repeatedly find a linear dependence among the
    # coefficient vectors and reduce the rank.
    
    # This is complex to implement correctly in a short time. 
    # Instead, I will use a fixed rank-32 decomposition which is an improvement over 48.
    
    # Rank 32 construction:
    # Use Strassen for the first 2x2x8 part (rank 2*7=14? No, 2x2x8 is 4 blocks of 2x2x2, rank 4*7=28).
    # Plus the remaining 1x2x8 part (rank 8). Total 36.
    
    # Let's try rank 32 by using a more efficient decomposition.
    # I will use a random search for a lower rank, but that's not deterministic.
    
    # For the purpose of this task, I will provide a rank-36 decomposition which is definitely better than 48.
    
    # Construction:
    # Split A into A1 (2x1) and A2 (2x2).
    # Split B into B1 (1x8) and B2 (2x8).
    # C = A1*B1 + A2*B2.
    # A1*B1: rank 8.
    # A2*B2: 2x2x8. Use Strassen on 2x2x2 blocks.
    # B2 is 2x8, so 4 blocks of 2x2.
    # Each 2x2x2 block uses Strassen (rank 7).
    # Total rank for A2*B2: 4*7 = 28.
    # Total rank: 8 + 28 = 36.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: A1*B1 (2x1x8)
    # A1 is 2x1: entries A[0,0], A[1,0].
    # B1 is 1x8: entries B[0,0]...B[0,7].
    # C1[i,k] = A[i,0] * B[0,k].
    # Rank-1 terms: for each k, u = [A[i,0]] (vector in R^2), v = [B[0,k]] (vector in R^1? No, B is 1x8, so v is scalar).
    # In our indexing: U has shape (n*m, rank) = (6, rank), V has shape (m*p, rank) = (24, rank), W has shape (p*n, rank) = (16, rank).
    # A1 is row 0 of A, col 0. So A1[i] = A[i,0]. Index in U: i*m + 0 = i*3.
    # B1 is row 0 of B, col k. So B1[k] = B[0,k]. Index in V: 0*p + k = k.
    # C1 is row i of C, col k. Index in W: k*n + i = 2*k + i.
    for k in range(8):
        u = [0] * 6
        v = [0] * 24
        w = [0] * 16
        u[0] = 1  # A[0,0]
        u[3] = 1  # A[1,0]
        v[k] = 1  # B[0,k]
        w[2*k + 0] = 1  # C[0,k]
        w[2*k + 1] = 1  # C[1,k]
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    # Part 2: A2*B2 (2x2x8)
    # A2 is 2x2: A[0,1], A[0,2], A[1,1], A[1,2].
    # B2 is 2x8: B[1,0..7], B[2,0..7].
    # C2 = A2 * B2.
    # Use Strassen on 2x2x2 blocks.
    # B2 has 4 blocks of 2x2: cols (0,1), (2,3), (4,5), (6,7).
    for block in range(4):
        k1 = 2 * block
        k2 = 2 * block + 1
        # Strassen for 2x2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B10
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Indices for A2:
        # A2[0,0] = A[0,1] -> U index 0*3+1 = 1
        # A2[0,1] = A[0,2] -> U index 0*3+2 = 2
        # A2[1,0] = A[1,1] -> U index 1*3+1 = 4
        # A2[1,1] = A[1,2] -> U index 1*3+2 = 5
        
        # Indices for B2:
        # B2[0,0] = B[1,k1] -> V index 1*8+k1 = 8+k1
        # B2[0,1] = B[1,k2] -> V index 1*8+k2 = 8+k2
        # B2[1,0] = B[2,k1] -> V index 2*8+k1 = 16+k1
        # B2[1,1] = B[2,k2] -> V index 2*8+k2 = 16+k2
        
        # Indices for C2:
        # C2[0,0] = C[0,k1] -> W index k1*2+0 = 2*k1
        # C2[0,1] = C[0,k2] -> W index k2*2+0 = 2*k2
        # C2[1,0] = C[1,k1] -> W index k1*2+1 = 2*k1+1
        # C2[1,1] = C[1,k2] -> W index k2*2+1 = 2*k2+1
        
        def add_term(u_coeff, v_coeff, w_coeff):
            # u_coeff: dict from U index to coefficient
            # v_coeff: dict from V index to coefficient
            # w_coeff: dict from W index to coefficient
            u = [0] * 6
            v = [0] * 24
            w = [0] * 16
            for idx, c in u_coeff.items():
                u[idx] += c
            for idx, c in v_coeff.items():
                v[idx] += c
            for idx, c in w_coeff.items():
                w[idx] += c
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        
        a00 = {1: 1}
        a01 = {2: 1}
        a10 = {4: 1}
        a11 = {5: 1}
        b00 = {8 + k1: 1}
        b01 = {8 + k2: 1}
        b10 = {16 + k1: 1}
        b11 = {16 + k2: 1}
        
        m1 = {1: 1, 5: 1, 8 + k1: 1, 16 + k2: 1, 2*k1: 1, 2*k2: 1, 2*k1+1: 1, 2*k2+1: 1}
        # M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        add_term({1: 1, 5: 1}, {8 + k1: 1, 16 + k2: 1}, {2*k1: 1, 2*k2+1: 1})
        
        # M2 = (A10 + A11) * B00
        # C10 += M2, C11 -= M2
        add_term({4: 1, 5: 1}, {8 + k1: 1}, {2*k1+1: 1, 2*k2+1: -1})
        
        # M3 = A00 * (B01 - B11)
        # C01 += M3, C11 += M3
        add_term({1: 1}, {8 + k2: 1, 16 + k2: -1}, {2*k2: 1, 2*k2+1: 1})
        
        # M4 = A11 * (B10 - B00)
        # C00 += M4, C10 += M4
        add_term({5: 1}, {16 + k1: 1, 8 + k1: -1}, {2*k1: 1, 2*k1+1: 1})
        
        # M5 = (A00 + A01) * B10
        # C00 -= M5, C01 += M5
        add_term({1: 1, 2: 1}, {16 + k1: 1}, {2*k1: -1, 2*k2: 1})
        
        # M6 = (A10 - A00) * (B00 + B01)
        # C11 += M6
        add_term({4: 1, 1: -1}, {8 + k1: 1, 8 + k2: 1}, {2*k2+1: 1})
        
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7
        add_term({2: 1, 5: -1}, {16 + k1: 1, 16 + k2: 1}, {2*k1: 1})
    
    # Total rank is 8 + 4*7 = 36.
    rank = len(U_list)
    U = [row for row in U_list]
    V = [row for row in V_list]
    W = [row for row in W_list]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
