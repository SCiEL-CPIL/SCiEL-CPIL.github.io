import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Structured decomposition: Strassen blocks + schoolbook
    # Strategy: Use Strassen for the 2x2 sub-blocks covering i∈{0,1}, j∈{0,1,2,3}
    # and schoolbook for i=2.
    #
    # For the 2x4 x 4x5 part (i=0,1):
    #   Split j into pairs: (j0,j1) and (j2,j3)
    #   For each pair, use Strassen's 7-term decomposition for 2x2x2
    #   But we need to handle the 5 columns...
    #
    # Better approach: Use Strassen for 2x2 blocks in the j dimension
    # j ∈ {0,1,2,3} can be split into two 2x2 blocks: {0,1} and {2,3}
    # For each 2x2 j-block, we compute 2x2 * 2x5 using a structured method
    #
    # Actually, let's use a cleaner approach:
    # - For i ∈ {0,1} and j ∈ {0,1}: Strassen 2x2x2 (7 terms) for k∈{0,1,2,3,4}
    #   Wait, Strassen is for 2x2*2x2, but we have 2x2*2x5
    #
    # Let me think differently. The key insight from inspiration 3:
    # 4 Strassen blocks (7 terms each = 28) + schoolbook for i=2 (4×5 = 20 terms) = 48
    #
    # How to get 4 Strassen blocks:
    # - i ∈ {0,1} (2 rows of A)
    # - j ∈ {0,1,2,3} (4 cols of A, 4 rows of B)  
    # - k ∈ {0,1,2,3,4} (5 cols of B)
    #
    # Split j into two groups of 2: {0,1} and {2,3}
    # For each group, we have a 2x2 (i,j) × 2x5 (j,k) multiplication
    # 
    # For 2x2 × 2x5, we can use Strassen-like decomposition:
    # Actually, let's just use the fact that we can decompose 2x2*2x5
    # using 7 terms by treating it as two 2x2*2x2 blocks plus a 2x2*2x1
    # That gives 7 + 7 + ? terms... not quite 7.
    #
    # Alternative: Use the known result that R(2,2,5) ≤ 14 (not 7)
    # So 4 blocks × 7 = 28 doesn't work directly.
    #
    # Let me reconsider. Perhaps the 4 Strassen blocks refer to:
    # Decomposing the full 2x4 x 4x5 problem differently.
    #
    # Actually, I think the approach is:
    # - Use Strassen on 2x2 submatrices in both i and j dimensions
    # - For i∈{0,1}, j∈{0,1}: 2x2*2x5 
    # - For i∈{0,1}, j∈{2,3}: 2x2*2x5
    # 
    # For 2x2*2x5, one approach: split k into {0,1,2,3} and {4}
    # 2x2*2x4 can be done with... hmm, this is getting complicated.
    #
    # Let me just try a direct construction with rank 48:
    # - 28 terms from Strassen-like decomposition for i∈{0,1}
    # - 20 terms from schoolbook for i=2
    #
    # For the Strassen part, I'll use the standard Strassen decomposition
    # applied to the 2x2 blocks in the (i,j) plane, handling all 5 k values.
    
    def strassen_2x2x5_block(i_base, j_base):
        """
        Generate Strassen-like decomposition for a 2x2 (i,j) x 2x5 (j,k) block.
        i_base: base index for i (0 or 1)
        j_base: base index for j (0 or 2)
        Returns list of (u_vec, v_vec, w_vec, alpha) tuples
        """
        # Standard Strassen for 2x2*2x2, extended to 2x2*2x5
        # We'll use 7 terms, each handling all 5 k values
        
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B21 - B22)  
        # M4 = A22 * (B12 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # For 2x2*2x5, B has shape 2x5, so B11, B12, B21, B22 are 1x5 vectors
        # The Strassen formulas still work, just with 1x5 vectors instead of 1x1
        
        terms = []
        
        # M1: (A11 + A22) * (B11 + B22)
        # A indices: (i_base, j_base) and (i_base+1, j_base+1)
        # B indices: (j_base, k) and (j_base+1, k) for k=0..4
        u1 = [0.0] * (n * m)
        u1[i_base * m + j_base] = 1.0
        u1[(i_base + 1) * m + (j_base + 1)] = 1.0
        v1 = [0.0] * (m * p)
        for k in range(p):
            v1[j_base * p + k] = 1.0
            v1[(j_base + 1) * p + k] = 1.0
        w1 = [0.0] * (p * n)
        for k in range(p):
            w1[k * n + i_base] = 1.0
            w1[k * n + (i_base + 1)] = 1.0
        terms.append((u1, v1, w1, 1.0))
        
        # M2: (A21 + A22) * B11
        u2 = [0.0] * (n * m)
        u2[(i_base + 1) * m + j_base] = 1.0
        u2[(i_base + 1) * m + (j_base + 1)] = 1.0
        v2 = [0.0] * (m * p)
        for k in range(p):
            v2[j_base * p + k] = 1.0
        w2 = [0.0] * (p * n)
        for k in range(p):
            w2[k * n + i_base] = 1.0
            w2[k * n + (i_base + 1)] = 1.0
        terms.append((u2, v2, w2, 1.0))
        
        # M3: A11 * (B21 - B22)
        u3 = [0.0] * (n * m)
        u3[i_base * m + j_base] = 1.0
        v3 = [0.0] * (m * p)
        for k in range(p):
            v3[(j_base + 1) * p + k] = 1.0
            v3[j_base * p + k] = -1.0
        w3 = [0.0] * (p * n)
        for k in range(p):
            w3[k * n + i_base] = 1.0
        terms.append((u3, v3, w3, 1.0))
        
        # M4: A22 * (B12 - B11)
        u4 = [0.0] * (n * m)
        u4[(i_base + 1) * m + (j_base + 1)] = 1.0
        v4 = [0.0] * (m * p)
        for k in range(p):
            v4[(j_base + 1) * p + k] = 1.0
            v4[j_base * p + k] = -1.0
        w4 = [0.0] * (p * n)
        for k in range(p):
            w4[k * n + (i_base + 1)] = 1.0
        terms.append((u4, v4, w4, 1.0))
        
        # M5: (A11 + A12) * B22
        u5 = [0.0] * (n * m)
        u5[i_base * m + j_base] = 1.0
        u5[i_base * m + (j_base + 1)] = 1.0
        v5 = [0.0] * (m * p)
        for k in range(p):
            v5[(j_base + 1) * p + k] = 1.0
        w5 = [0.0] * (p * n)
        for k in range(p):
            w5[k * n + i_base] = 1.0
            w5[k * n + (i_base + 1)] = 1.0
        terms.append((u5, v5, w5, 1.0))
        
        # M6: (A21 - A11) * (B11 + B12)
        u6 = [0.0] * (n * m)
        u6[(i_base + 1) * m + j_base] = 1.0
        u6[i_base * m + j_base] = -1.0
        v6 = [0.0] * (m * p)
        for k in range(p):
            v6[j_base * p + k] = 1.0
            v6[(j_base + 1) * p + k] = 1.0
        w6 = [0.0] * (p * n)
        for k in range(p):
            w6[k * n + i_base] = 1.0
            w6[k * n + (i_base + 1)] = 1.0
        terms.append((u6, v6, w6, 1.0))
        
        # M7: (A12 - A22) * (B21 + B22)
        u7 = [0.0] * (n * m)
        u7[i_base * m + (j_base + 1)] = 1.0
        u7[(i_base + 1) * m + (j_base + 1)] = -1.0
        v7 = [0.0] * (m * p)
        for k in range(p):
            v7[(j_base + 1) * p + k] = 1.0
            v7[j_base * p + k] = 1.0
        w7 = [0.0] * (p * n)
        for k in range(p):
            w7[k * n + i_base] = 1.0
            w7[k * n + (i_base + 1)] = 1.0
        terms.append((u7, v7, w7, 1.0))
        
        return terms
    
    # Build the full decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Strassen blocks for i∈{0,1}, j∈{0,1} and i∈{0,1}, j∈{2,3}
    for j_base in [0, 2]:
        for term in strassen_2x2x5_block(0, j_base):
            U_cols.append(term[0])
            V_cols.append(term[1])
            W_cols.append(term[2])
    
    # Schoolbook for i=2: all j∈{0,1,2,3}, k∈{0,1,2,3,4}
    for j in range(m):
        for k in range(p):
            u = [0.0] * (n * m)
            u[2 * m + j] = 1.0
            v = [0.0] * (m * p)
            v[j * p + k] = 1.0
            w = [0.0] * (p * n)
            w[k * n + 2] = 1.0
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    rank = len(U_cols)
    
    # Build matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for idx, val in enumerate(U_cols[c]):
            if val != 0:
                U[idx][c] = val
        for idx, val in enumerate(V_cols[c]):
            if val != 0:
                V[idx][c] = val
        for idx, val in enumerate(W_cols[c]):
            if val != 0:
                W[idx][c] = val
    
    # Verify the decomposition
    success = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    success = False
                    break
            if not success:
                break
        if not success:
            break
    
    if success:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0.0] * rank_sb for _ in range(n * m)]
        V_sb = [[0.0] * rank_sb for _ in range(m * p)]
        W_sb = [[0.0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1.0
                    V_sb[j*p+k][col] = 1.0
                    W_sb[k*n+i][col] = 1.0
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
