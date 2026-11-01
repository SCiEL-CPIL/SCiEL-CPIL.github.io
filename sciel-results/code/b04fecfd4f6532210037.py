import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=200, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            # Factor A: sum_j A[i,j] * U[i*m+j, c]
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            
            # Factor B: sum_j V[j*p+k, c] * B[j,k]
            factor_B = np.zeros(p)
            for k in range(p):
                factor_B[k] = sum(V[j * p + k][c] * B[j, k] for j in range(m))
            
            # Multiply and scale by W
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # A indices order: a, b, c, d
    # B indices order: e, f, g, h
    # C indices order: c00, c01, c10, c11
    
    # U coefficients (7 factors)
    # P1: a+d, P2: c+d, P3: a, P4: d, P5: a+b, P6: c-d, P7: b+d
    U_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [0, 0, 1, -1],# P6
        [0, 1, 0, 1]  # P7
    ]
    
    # V coefficients (7 factors)
    # P1: e+h, P2: h, P3: e-g, P4: h-f, P5: g, P6: e+f, P7: g+h
    V_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 0, 1], # P2
        [1, 0, -1, 0],# P3
        [0, -1, 0, 1],# P4
        [0, 0, 1, 0], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    # W coefficients (4 entries, 7 factors)
    # C00: P1 + P4 - P5 + P7
    # C01: P3 + P5
    # C10: P2 + P6
    # C11: P1 - P2 + P3 + P6
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 0, 0, 1, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Define blocks
    # Each block: (a_indices, b_indices, c_indices)
    # a_indices: [a, b, c, d] in U
    # b_indices: [e, f, g, h] in V
    # c_indices: [c00, c01, c10, c11] in W
    
    blocks = []
    
    # A1 = A[0:2, 0:2] -> U indices: 0, 1, 4, 5
    # A2 = A[0:2, 2:4] -> U indices: 2, 3, 6, 7
    
    A1_idx = [0, 1, 4, 5]
    A2_idx = [2, 3, 6, 7]
    
    # B_top = B[0:2, :]
    # B_bot = B[2:4, :]
    
    # B blocks are 2x2
    # B[j, k] index in V is j*p + k = j*8 + k
    
    # Top part (j=0,1)
    B_top_0 = [0*8+0, 0*8+1, 1*8+0, 1*8+1] # e,f,g,h for cols 0,1
    B_top_1 = [0*8+2, 0*8+3, 1*8+2, 1*8+3] # cols 2,3
    B_top_2 = [0*8+4, 0*8+5, 1*8+4, 1*8+5] # cols 4,5
    B_top_3 = [0*8+6, 0*8+7, 1*8+6, 1*8+7] # cols 6,7
    
    # Bottom part (j=2,3)
    B_bot_0 = [2*8+0, 2*8+1, 3*8+0, 3*8+1] # cols 0,1
    B_bot_1 = [2*8+2, 2*8+3, 3*8+2, 3*8+3] # cols 2,3
    B_bot_2 = [2*8+4, 2*8+5, 3*8+4, 3*8+5] # cols 4,5
    B_bot_3 = [2*8+6, 2*8+7, 3*8+6, 3*8+7] # cols 6,7
    
    # C indices: k*n + i = k*2 + i
    # C cols 0,1: k=0,1
    C_01 = [0*2+0, 0*2+1, 1*2+0, 1*2+1] # c00, c01, c10, c11
    C_23 = [2*2+0, 2*2+1, 3*2+0, 3*2+1]
    C_45 = [4*2+0, 4*2+1, 5*2+0, 5*2+1]
    C_67 = [6*2+0, 6*2+1, 7*2+0, 7*2+1]
    
    # Combine blocks
    # 4 blocks for A1, 4 blocks for A2
    for i in range(4):
        blocks.append((A1_idx, [B_top_0, B_top_1, B_top_2, B_top_3][i], [C_01, C_23, C_45, C_67][i]))
        blocks.append((A2_idx, [B_bot_0, B_bot_1, B_bot_2, B_bot_3][i], [C_01, C_23, C_45, C_67][i]))
        
    # Wait, the order of blocks doesn't matter for correctness, but let's interleave or just append.
    # Let's just append A1 blocks then A2 blocks.
    # Actually, the loop above appends A1 block i, then A2 block i.
    # This is fine.
    
    col_offset = 0
    for a_idx, b_idx, c_idx in blocks:
        for f in range(7):
            col = col_offset + f
            # Fill U
            for r in range(4):
                U[a_idx[r]][col] = U_strassen[f][r]
            # Fill V
            for r in range(4):
                V[b_idx[r]][col] = V_strassen[f][r]
            # Fill W
            for r in range(4):
                W[c_idx[r]][col] = W_strassen[r][f]
        col_offset += 7
        
    # Verify
    if _verify(U, V, W, n, m, p, trials=50, tol=1e-9):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
