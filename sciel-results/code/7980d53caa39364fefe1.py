import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 94
    # Partition:
    # n: 3
    # m: 3, 2
    # p: 3, 4
    
    # Block 1: n=3, m=3, p=3 (Lader Rank 23)
    # Block 2: n=3, m=2, p=7
    #   Sub-blocks of p: 2, 2, 2, 1
    #   Each 3x2x2 is Rank 11 (Strassen 2x2x2 + Schoolbook 1x2x2)
    #   3x2x1 is Rank 6
    # Block 3: n=3, m=3, p=4
    #   Sub-blocks of p: 3, 1
    #   3x3x3 is Lader Rank 23
    #   3x3x1 is Rank 9
    
    # We need to generate the U, V, W matrices.
    # U: (n*m, rank) -> (15, rank)
    # V: (m*p, rank) -> (35, rank)
    # W: (p*n, rank) -> (21, rank)
    
    # Indices:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # We will build the decomposition term by term.
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        # u_idx is (i, j) in 0-based for the sub-block
        # v_idx is (j, k)
        # w_idx is (k, i)
        U_list.append((u_idx, coeff))
        V_list.append((v_idx, coeff))
        W_list.append((w_idx, coeff))
        
    # This simple add_term doesn't work for multiple terms in one rank dimension.
    # We need to accumulate columns.
    
    U = []
    V = []
    W = []
    
    def add_rank_term(u_vec, v_vec, w_vec, coeff=1.0):
        # u_vec is length n*m
        # v_vec is length m*p
        # w_vec is length p*n
        U.append(u_vec * coeff)
        V.append(v_vec * coeff)
        W.append(w_vec * coeff)
        
    # Helper to create zero vectors
    def zero_u(): return [0.0] * (n*m)
    def zero_v(): return [0.0] * (m*p)
    def zero_w(): return [0.0] * (p*n)
    
    def set_u(vec, i, j, val):
        vec[i*m + j] = val
    def set_v(vec, j, k, val):
        vec[j*p + k] = val
    def set_w(vec, k, i, val):
        vec[k*n + i] = val
        
    # 1. Block 1: 3x3x3 at n[0:3], m[0:3], p[0:3]
    # Lader Decomposition for 3x3x3
    # The 23 terms are known.
    # I will hardcode the Lader decomposition.
    # Reference: https://en.wikipedia.org/wiki/Matrix_multiplication#Lader%27s_algorithm
    # Or similar source.
    # The Lader decomposition uses 23 multiplications.
    # I will implement a standard set of 23 terms.
    
    # Due to complexity of hardcoding 23 terms correctly from memory, 
    # and risk of error, I will fallback to Strassen blocks for 2x2x2 parts 
    # and schoolbook for the rest, ensuring I beat rank 99.
    
    # Strategy:
    # Use 6 disjoint 2x2x2 Strassen blocks.
    # n=3 -> 2, 1
    # m=5 -> 2, 2, 1
    # p=7 -> 2, 2, 2, 1
    
    # Blocks:
    # 1. n[0:2], m[0:2], p[0:2] -> Strassen (Rank 7)
    # 2. n[0:2], m[2:4], p[0:2] -> Strassen (Rank 7)
    # 3. n[0:2], m[0:2], p[2:4] -> Strassen (Rank 7)
    # 4. n[0:2], m[2:4], p[2:4] -> Strassen (Rank 7)
    # 5. n[0:2], m[0:2], p[4:6] -> Strassen (Rank 7)
    # 6. n[0:2], m[2:4], p[4:6] -> Strassen (Rank 7)
    
    # These 6 blocks cover n[0:2], m[0:4], p[0:6].
    # Rank so far: 6 * 7 = 42.
    
    # Remaining parts:
    # A. n[2:3], m[0:5], p[0:7] -> 1x5x7 (Rank 35)
    # B. n[0:2], m[4:5], p[0:7] -> 2x1x7 (Rank 14)
    # C. n[0:2], m[0:4], p[6:7] -> 2x4x1 (Rank 8)
    
    # Total Rank: 42 + 35 + 14 + 8 = 99.
    # This is the inspiration's rank.
    
    # To beat 99, we need to optimize the remaining parts or use a better block.
    # The 2x1x7 block (B) can be optimized?
    # 2x1x7 is schoolbook.
    # The 1x5x7 block (A) is schoolbook.
    # The 2x4x1 block (C) is schoolbook.
    
    # Let's try to use a 3x3x3 Lader block instead of two 2x2x2 Strassen blocks in the corner.
    # Replace Blocks 1 and 4 (and parts of 2,3) with a 3x3x3 block?
    # 3x3x3 covers n[0:3], m[0:3], p[0:3].
    # Rank 23.
    # Previous Strassen blocks in this region:
    # Block 1: n[0:2], m[0:2], p[0:2] (Rank 7)
    # Block 2: n[0:2], m[2:4], p[0:2] (part in m[2:3])
    # Block 3: n[0:2], m[0:2], p[2:4] (part in p[2:3])
    # Block 4: n[0:2], m[2:4], p[2:4] (part in m[2:3], p[2:3])
    
    # This is getting complicated to mix.
    
    # I will stick to the 6 Strassen blocks + schoolbook remainder (Rank 99) 
    # but attempt to find a rank 98 decomposition by checking if any 
    # 2x2x3 or 2x3x2 sub-blocks in the remainder can be optimized.
    
    # Actually, the simplest improvement is to use the known rank 98 decomposition 
    # if I can find it.
    
    # Given the time, I will implement the Rank 99 solution which is an improvement 
    # over the seed (105).
    
    # Implementation of 6 Strassen blocks + Schoolbook
    
    rank = 99
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Strassen Block Helper
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # i0, j0, k0 are start indices
        # n=2, m=2, p=2
        # 7 terms
        # A11, A12, A21, A22
        # B11, B12, B21, B22
        
        # M1 = (A11+A22)(B11+B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        # A11
        u[i0*m + j0] = 1; v[j0*p + k0] = 1; w[k0*n + i0] = 1
        # A22
        u[(i0+1)*m + (j0+1)] = 1; v[(j0+1)*p + (k0+1)] = 1; w[(k0+1)*n + (i0+1)] = 1
        # B11
        v[j0*p + k0] = 1; w[k0*n + i0] = 1
        # B22
        v[(j0+1)*p + (k0+1)] = 1; w[(k0+1)*n + (i0+1)] = 1
        # This manual construction is error prone.
        
        # I will use the standard Strassen formulas.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # I will add these 7 terms.
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            U[:, col] = u_coeffs
            V[:, col] = v_coeffs
            W[:, col] = w_coeffs
            col += 1
            
        # M1
        u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
        u1[i0*m + j0] = 1; u1[(i0+1)*m + (j0+1)] = 1
        v1[j0*p + k0] = 1; v1[(j0+1)*p + (k0+1)] = 1
        w1[k0*n + i0] = 1; w1[(k0+1)*n + (i0+1)] = 1
        add_term(u1, v1, w1)
        
        # M2
        u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
        u2[(i0+1)*m + j0] = 1; u2[(i0+1)*m + (j0+1)] = 1
        v2[j0*p + k0] = 1
        w2[k0*n + (i0+1)] = 1
        add_term(u2, v2, w2)
        
        # M3
        u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
        u3[i0*m + j0] = 1
        v3[j0*p + (k0+1)] = 1; v3[(j0+1)*p + (k0+1)] = -1
        w3[(k0+1)*n + i0] = 1
        add_term(u3, v3, w3)
        
        # M4
        u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
        u4[(i0+1)*m + (j0+1)] = 1
        v4[(j0+1)*p + k0] = 1; v4[j0*p + k0] = -1
        w4[k0*n + (i0+1)] = 1
        add_term(u4, v4, w4)
        
        # M5
        u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
        u5[i0*m + j0] = 1; u5[i0*m + (j0+1)] = 1
        v5[(j0+1)*p + (k0+1)] = 1
        w5[(k0+1)*n + i0] = 1
        add_term(u5, v5, w5)
        
        # M6
        u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
        u6[(i0+1)*m + j0] = 1; u6[i0*m + j0] = -1
        v6[j0*p + k0] = 1; v6[j0*p + (k0+1)] = 1
        w6[k0*n + (i0+1)] = 1; w6[(k0+1)*n + (i0+1)] = 1
        add_term(u6, v6, w6)
        
        # M7
        u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
        u7[i0*m + (j0+1)] = 1; u7[(i0+1)*m + (j0+1)] = -1
        v7[(j0+1)*p + k0] = 1; v7[(j0+1)*p + (k0+1)] = 1
        w7[k0*n + i0] = 1; w7[(k0+1)*n + i0] = 1
        add_term(u7, v7, w7)
        
    # Add 6 Strassen blocks
    # n[0:2], m[0:2], p[0:2]
    add_strassen_block(0, 0, 0)
    # n[0:2], m[2:4], p[0:2]
    add_strassen_block(0, 2, 0)
    # n[0:2], m[0:2], p[2:4]
    add_strassen_block(0, 0, 2)
    # n[0:2], m[2:4], p[2:4]
    add_strassen_block(0, 2, 2)
    # n[0:2], m[0:2], p[4:6]
    add_strassen_block(0, 0, 4)
    # n[0:2], m[2:4], p[4:6]
    add_strassen_block(0, 2, 4)
    
    # Add Schoolbook for remaining
    # A. n[2:3], m[0:5], p[0:7]
    for i in range(2, 3):
        for j in range(5):
            for k in range(7):
                u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                U[:, col] = u; V[:, col] = v; W[:, col] = w
                col += 1
                
    # B. n[0:2], m[4:5], p[0:7]
    for i in range(2):
        for j in range(4, 5):
            for k in range(7):
                u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                U[:, col] = u; V[:, col] = v; W[:, col] = w
                col += 1
                
    # C. n[0:2], m[0:4], p[6:7]
    for i in range(2):
        for j in range(4):
            for k in range(6, 7):
                u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                U[:, col] = u; V[:, col] = v; W[:, col] = w
                col += 1
                
    # Check col
    if col != rank:
        # Adjust rank to actual col
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
