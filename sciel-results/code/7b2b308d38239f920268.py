"""
Strategy: Structural Block Decomposition using Strassen for 2x2x2 blocks.
Decomposes 3x4x8 into:
1. 2x4x8 block (using two 2x2x8 Strassen-based sub-blocks)
2. 1x4x8 block (using schoolbook)
Total rank: 28 + 28 + 32 = 88.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Total rank
    rank = 88
    
    # Initialize U, V, W
    # U: (n*m, rank) = (12, 88)
    # V: (m*p, rank) = (32, 88)
    # W: (p*n, rank) = (24, 88)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a local 2x2x2 Strassen block
    # i_base, j_base, k_base are the starting indices for the local 2x2x2 block
    # s_offset is the starting column index in the rank dimension
    def add_strassen_block(i_base, j_base, k_base, s_offset):
        # Local indices i, j, k in {0, 1}
        # Global indices:
        # i_g = i_base + i
        # j_g = j_base + j
        # k_g = k_base + k
        
        # U index: i_g * m + j_g
        # V index: j_g * p + k_g
        # W index: k_g * n + i_g
        
        # Strassen factors (7 of them)
        # P1 = (A00 + A11)(B00 + B11)
        # P2 = (A10 + A11)B00
        # P3 = A00(B01 - B11)
        # P4 = A11(B10 - B00)
        # P5 = (A00 + A01)B11
        # P6 = (A10 - A00)(B00 + B01)
        # P7 = (A01 - A11)(B10 + B11)
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Define u, v, w for each s in 0..6
        # u_s is vector of length 4 (for A00, A01, A10, A11)
        # v_s is vector of length 4 (for B00, B01, B10, B11)
        # w_s is vector of length 4 (for C00, C01, C10, C11) -> indices 0,1,2,3 correspond to (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
        # Wait, W index is k*n + i.
        # For local block, n=2.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # P1
        u1 = np.array([1, 0, 0, 1]) # A00, A11
        v1 = np.array([1, 0, 0, 1]) # B00, B11
        w1 = np.array([1, 0, 0, 1]) # C00, C11
        
        # P2
        u2 = np.array([0, 0, 1, 1]) # A10, A11
        v2 = np.array([1, 0, 0, 0]) # B00
        w2 = np.array([0, 0, 1, -1]) # C10, -C11
        
        # P3
        u3 = np.array([1, 0, 0, 0]) # A00
        v3 = np.array([0, 1, 0, -1]) # B01, -B11
        w3 = np.array([0, 1, 0, 1]) # C01, C11
        
        # P4
        u4 = np.array([0, 0, 0, 1]) # A11
        v4 = np.array([-1, 0, 1, 0]) # -B00, B10
        w4 = np.array([1, 0, 1, 0]) # C00, C10
        
        # P5
        u5 = np.array([1, 1, 0, 0]) # A00, A01
        v5 = np.array([0, 0, 0, 1]) # B11
        w5 = np.array([-1, 1, 0, 0]) # -C00, C01
        
        # P6
        u6 = np.array([-1, 0, 1, 0]) # -A00, A10
        v6 = np.array([1, 1, 0, 0]) # B00, B01
        w6 = np.array([0, 0, 0, 1]) # C11
        
        # P7
        u7 = np.array([0, 1, 0, -1]) # A01, -A11
        v7 = np.array([0, 0, 1, 1]) # B10, B11
        w7 = np.array([1, 0, 0, 0]) # C00
        
        factors = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for s_idx, (u, v, w) in enumerate(factors):
            s = s_offset + s_idx
            for i in range(2):
                for j in range(2):
                    u_idx = (i_base + i) * m + (j_base + j)
                    U[u_idx, s] += u[i * 2 + j]
                    
            for j in range(2):
                for k in range(2):
                    v_idx = (j_base + j) * p + (k_base + k)
                    V[v_idx, s] += v[j * 2 + k]
                    
            for k in range(2):
                for i in range(2):
                    w_idx = (k_base + k) * n + (i_base + i)
                    W[w_idx, s] += w[k * 2 + i]
                    
    # Part 1: 2x4x8 block (i in 0..1)
    # Split j into 0..1 and 2..3
    # Split k into 0..1, 2..3, 4..5, 6..7
    
    # Block 1.1: i in 0..1, j in 0..1, k in 0..1
    add_strassen_block(0, 0, 0, 0)
    
    # Block 1.2: i in 0..1, j in 0..1, k in 2..3
    add_strassen_block(0, 0, 2, 7)
    
    # Block 1.3: i in 0..1, j in 0..1, k in 4..5
    add_strassen_block(0, 0, 4, 14)
    
    # Block 1.4: i in 0..1, j in 0..1, k in 6..7
    add_strassen_block(0, 0, 6, 21)
    
    # Block 1.5: i in 0..1, j in 2..3, k in 0..1
    add_strassen_block(0, 2, 0, 28)
    
    # Block 1.6: i in 0..1, j in 2..3, k in 2..3
    add_strassen_block(0, 2, 2, 35)
    
    # Block 1.7: i in 0..1, j in 2..3, k in 4..5
    add_strassen_block(0, 2, 4, 42)
    
    # Block 1.8: i in 0..1, j in 2..3, k in 6..7
    add_strassen_block(0, 2, 6, 49)
    
    # Part 2: 1x4x8 block (i in 2)
    # Schoolbook for i=2
    # C[2, k] = sum_j A[2, j] B[j, k]
    # For each j in 0..3, k in 0..7, we have one factor.
    # Total 4 * 8 = 32 factors.
    s_offset = 56
    s_idx = 0
    for j in range(4):
        for k in range(8):
            s = s_offset + s_idx
            s_idx += 1
            
            # U[2 * 4 + j, s] = 1
            U[2 * m + j, s] = 1.0
            
            # V[j * 8 + k, s] = 1
            V[j * p + k, s] = 1.0
            
            # W[k * 3 + 2, s] = 1
            W[k * n + 2, s] = 1.0
            
    # Verify solution
    def check_solution(U, V, W, rank):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not check_solution(U, V, W, rank):
        # Fallback to schoolbook if structural construction fails (should not happen)
        rank_school = n * m * p
        U_base = np.zeros((n * m, rank_school), dtype=float)
        V_base = np.zeros((m * p, rank_school), dtype=float)
        W_base = np.zeros((p * n, rank_school), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_base[i * m + j, col] = 1
                    V_base[j * p + k, col] = 1
                    W_base[k * n + i, col] = 1
                    col += 1
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
