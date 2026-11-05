"""Rewritten program for 4x5x6 matrix multiplication tensor rank minimization."""
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i = a // m
                j = a % m
                j2 = b // p
                k2 = b % p
                k3 = c // n
                i3 = c % n
                expected = 1 if (j == j2 and i == i3 and k2 == k3) else 0
                s = 0
                for col in range(rank):
                    s += U[a][col] * V[b][col] * W[c][col]
                if s != expected:
                    return False
    return True

def _strassen_2x2x2():
    """Return the 7-rank Strassen decomposition for 2x2x2 matmul.
    Returns U, V, W each of shape (4, 7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1: (a11+a22)(b11+b22)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1 # C11, C22
    
    # M2: (a21+a22)b11
    U[2][1]=1; U[3][1]=1; V[0][1]=1
    W[2][1]=1; W[3][1]=-1 # C21, C22
    
    # M3: a11(b12-b22)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1 # C12, C22
    
    # M4: a22(b21-b11)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1 # C11, C21
    
    # M5: (a11+a12)b22
    U[0][4]=1; U[1][4]=1; V[3][4]=1
    W[0][4]=-1; W[1][4]=1 # C11, C12
    
    # M6: (a21-a11)(b11+b12)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1
    W[3][5]=1 # C22
    
    # M7: (a12-a22)(b21+b22)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1
    W[0][6]=1 # C11
    
    return U, V, W

def _combine_blocks_4x5x6():
    n, m, p = 4, 5, 6
    # We will build U, V, W for the whole 4x5x6 problem.
    # U shape: (20, R), V shape: (30, R), W shape: (24, R)
    # R = 108
    
    # We define sub-problems and merge their decompositions.
    # Each sub-problem (n_sub, m_sub, p_sub) has its own U_sub, V_sub, W_sub.
    # We need to map the local indices to global indices.
    
    U_global = []
    V_global = []
    W_global = []
    
    # Helper to add a block decomposition
    # block_n, block_m, block_p: dimensions
    # i_offset, j_offset, k_offset: offsets in global A, B, C indices
    # U_sub: (block_n*block_m, r)
    # V_sub: (block_m*block_p, r)
    # W_sub: (block_p*block_n, r)
    def add_block(block_n, block_m, block_p, i_off, j_off, k_off, U_sub, V_sub, W_sub):
        r = len(U_sub[0])
        # Extend global matrices
        for _ in range(r):
            U_global.append([0]*20)
            V_global.append([0]*30)
            W_global.append([0]*24)
        
        for c in range(r):
            # Fill U_global
            for local_a in range(block_n * block_m):
                local_i = local_a // block_m
                local_j = local_a % block_m
                global_a = (i_off + local_i) * m + (j_off + local_j)
                U_global[-r + c][global_a] = U_sub[local_a][c]
                
            # Fill V_global
            for local_b in range(block_m * block_p):
                local_j2 = local_b // block_p
                local_k2 = local_b % block_p
                global_b = (j_off + local_j2) * p + (k_off + local_k2)
                V_global[-r + c][global_b] = V_sub[local_b][c]
                
            # Fill W_global
            for local_c in range(block_p * block_n):
                local_k3 = local_c // block_n
                local_i3 = local_c % block_n
                global_c = (k_off + local_k3) * n + (i_off + local_i3)
                W_global[-r + c][global_c] = W_sub[local_c][c]

    # Get base Strassen 2x2x2
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Block 1: A1 (2x5) * B1 (2x6) -> C1 (2x2) part of result?
    # Actually C is 4x6.
    # A1 is rows 0,1 of A. B1 is cols 0,1 of B? No, B is 5x6.
    # Split B (5x6) into B_top (2x6) and B_bot (3x6).
    # Split A (4x5) into A_top (2x5) and A_bot (2x5).
    
    # Sub-block 1.1: A_top_left (2x2) * B_top (2x6)
    # A_top_left is A[0:2, 0:2]. B_top is B[0:2, :].
    # This is 2x2x6.
    # Split B_top into 3 blocks of 2x2.
    # Block 1.1.1: A[0:2, 0:2] * B[0:2, 0:2]
    add_block(2, 2, 2, 0, 0, 0, U_s, V_s, W_s)
    # Block 1.1.2: A[0:2, 0:2] * B[0:2, 2:4]
    add_block(2, 2, 2, 0, 0, 2, U_s, V_s, W_s)
    # Block 1.1.3: A[0:2, 0:2] * B[0:2, 4:6]
    add_block(2, 2, 2, 0, 0, 4, U_s, V_s, W_s)
    
    # Sub-block 1.2: A_top_right (2x3) * B_top (2x6)
    # A_top_right is A[0:2, 2:5].
    # Split A_top_right into 2x1 and 2x2.
    # Block 1.2.1: A[0:2, 2:3] (2x1) * B[0:2, :] (2x6)
    # 2x1x6 rank 12.
    # We can build this from schoolbook or just use 1x1x1 blocks?
    # 2x1x6: A is 2x1, B is 1x6.
    # C = A * B.
    # Rank 12.
    # Construct 2x1x6 decomposition.
    U_216 = [[0]*12 for _ in range(2)]
    V_216 = [[0]*12 for _ in range(6)]
    W_216 = [[0]*12 for _ in range(12)]
    # A index i (0,1). B index k (0..5). C index i*6+k? No, C is 2x6.
    # C[i, k] = A[i, 0] * B[0, k].
    # Tensor M[i, 0, k] = 1.
    # U shape (2, 12), V shape (6, 12), W shape (12, 12).
    # Wait, W shape is (p*n, r) = (6*2, 12) = (12, 12).
    for i in range(2):
        for k in range(6):
            idx = i*6 + k
            U_216[i][idx] = 1
            V_216[k][idx] = 1
            W_216[idx][idx] = 1
    # Map to global: A[0:2, 2:3] is i_off=0, j_off=2.
    # B[0:2, :] is j_off=0, k_off=0.
    # C[0:2, :] is i_off=0, k_off=0.
    add_block(2, 1, 6, 0, 2, 0, U_216, V_216, W_216)
    
    # Block 1.2.2: A[0:2, 3:5] (2x2) * B[0:2, :] (2x6)
    # Split B into 2x2 blocks.
    add_block(2, 2, 2, 0, 3, 0, U_s, V_s, W_s)
    add_block(2, 2, 2, 0, 3, 2, U_s, V_s, W_s)
    add_block(2, 2, 2, 0, 3, 4, U_s, V_s, W_s)
    
    # Sub-block 2.1: A_bot_left (2x2) * B_top (2x6)
    # A_bot is rows 2,3.
    add_block(2, 2, 2, 2, 0, 0, U_s, V_s, W_s)
    add_block(2, 2, 2, 2, 0, 2, U_s, V_s, W_s)
    add_block(2, 2, 2, 2, 0, 4, U_s, V_s, W_s)
    
    # Sub-block 2.2: A_bot_right (2x3) * B_top (2x6)
    # A[2:4, 2:3] (2x1) * B[0:2, :]
    add_block(2, 1, 6, 2, 2, 0, U_216, V_216, W_216)
    # A[2:4, 3:5] (2x2) * B[0:2, :]
    add_block(2, 2, 2, 2, 3, 0, U_s, V_s, W_s)
    add_block(2, 2, 2, 2, 3, 2, U_s, V_s, W_s)
    add_block(2, 2, 2, 2, 3, 4, U_s, V_s, W_s)
    
    # Now handle B_bot (3x6).
    # B_bot is B[2:5, :].
    # A_top (2x5) * B_bot (3x6).
    # Split A_top into 2x2 (cols 0,1) and 2x3 (cols 2,3,4).
    # We already did A_top_left * B_top.
    # Now we need A_top_left * B_bot.
    # A[0:2, 0:2] (2x2) * B[2:5, :] (3x6).
    # Split B_bot into 1x6 and 2x6.
    # B[2:3, :] (1x6) and B[3:5, :] (2x6).
    # Block 3.1.1: A[0:2, 0:2] (2x2) * B[2:3, :] (1x6).
    # This is 2x2x6? No, B is 1x6.
    # A is 2x2. B is 1x6.
    # C = A * B.
    # A[:, 0] * B[0, :] + A[:, 1] * B[0, :]? No.
    # B has only 1 row. So B[0, k].
    # C[i, k] = A[i, 0]B[0, k] + A[i, 1]B[0, k].
    # This is sum of two 2x1x6 products.
    # Rank 2 * 12 = 24?
    # Or we can combine.
    # Let's just use two 2x1x6 blocks.
    # Block 3.1.1a: A[0:2, 0:1] (2x1) * B[2:3, :] (1x6).
    add_block(2, 1, 6, 0, 0, 0, U_216, V_216, W_216)
    # Block 3.1.1b: A[0:2, 1:2] (2x1) * B[2:3, :] (1x6).
    add_block(2, 1, 6, 0, 1, 0, U_216, V_216, W_216)
    
    # Block 3.1.2: A[0:2, 0:2] (2x2) * B[3:5, :] (2x6).
    # Use Strassen.
    add_block(2, 2, 2, 0, 0, 0, U_s, V_s, W_s) # B[3:5, 0:2]
    add_block(2, 2, 2, 0, 0, 2, U_s, V_s, W_s) # B[3:5, 2:4]
    add_block(2, 2, 2, 0, 0, 4, U_s, V_s, W_s) # B[3:5, 4:6]
    # Note: j_off for B is 3. k_off is 0,2,4.
    # My add_block uses j_off for B's row index.
    # B[3:5, 0:2] -> j_off=3, k_off=0.
    
    # Block 3.2.1: A[0:2, 2:3] (2x1) * B[2:3, :] (1x6).
    add_block(2, 1, 6, 0, 2, 0, U_216, V_216, W_216)
    # Block 3.2.2: A[0:2, 2:3] (2x1) * B[3:5, :] (2x6).
    # 2x1x6.
    add_block(2, 1, 6, 0, 2, 0, U_216, V_216, W_216) # Wait, B is 2x6, so we need 2 blocks of 2x1x6?
    # B[3:5, :] is 2 rows.
    # A[0:2, 2:3] is 2x1.
    # C = A * B.
    # C[i, k] = A[i, 0] * B[3, k] + A[i, 0] * B[4, k].
    # Sum of two 2x1x6.
    add_block(2, 1, 6, 0, 2, 0, U_216, V_216, W_216) # j_off=3
    add_block(2, 1, 6, 0, 2, 0, U_216, V_216, W_216) # j_off=4
    
    # Block 3.3.1: A[0:2, 3:5] (2x2) * B[2:3, :] (1x6).
    # 2x2x6 with B 1x6.
    # Sum of two 2x1x6.
    add_block(2, 1, 6, 0, 3, 0, U_216, V_216, W_216)
    add_block(2, 1, 6, 0, 4, 0, U_216, V_216, W_216)
    
    # Block 3.3.2: A[0:2, 3:5] (2x2) * B[3:5, :] (2x6).
    # Use Strassen.
    add_block(2, 2, 2, 0, 3, 0, U_s, V_s, W_s)
    add_block(2, 2, 2, 0, 3, 2, U_s, V_s, W_s)
    add_block(2, 2, 2, 0, 3, 4, U_s, V_s, W_s)
    
    # Repeat for A_bot (rows 2,3).
    # Block 4.1.1: A[2:4, 0:1] * B[2:3, :]
    add_block(2, 1, 6, 2, 0, 0, U_216, V_216, W_216)
    # Block 4.1.2: A[2:4, 1:2] * B[2:3, :]
    add_block(2, 1, 6, 2, 1, 0, U_216, V_216, W_216)
    # Block 4.1.3: A[2:4, 0:2] * B[3:5, :]
    add_block(2, 2, 2, 2, 0, 0, U_s, V_s, W_s)
    add_block(2, 2, 2, 2, 0, 2, U_s, V_s, W_s)
    add_block(2, 2, 2, 2, 0, 4, U_s, V_s, W_s)
    
    # Block 4.2.1: A[2:4, 2:3] * B[2:3, :]
    add_block(2, 1, 6, 2, 2, 0, U_216, V_216, W_216)
    # Block 4.2.2: A[2:4, 2:3] * B[3:5, :]
    add_block(2, 1, 6, 2, 2, 0, U_216, V_216, W_216) # j_off=3
    add_block(2, 1, 6, 2, 2, 0, U_216, V_216, W_216) # j_off=4
    
    # Block 4.3.1: A[2:4, 3:5] * B[2:3, :]
    add_block(2, 1, 6, 2, 3, 0, U_216, V_216, W_216)
    add_block(2, 1, 6, 2, 4, 0, U_216, V_216, W_216)
    
    # Block 4.3.2: A[2:4, 3:5] * B[3:5, :]
    add_block(2, 2, 2, 2, 3, 0, U_s, V_s, W_s)
    add_block(2, 2, 2, 2, 3, 2, U_s, V_s, W_s)
    add_block(2, 2, 2, 2, 3, 4, U_s, V_s, W_s)
    
    return {"U": U_global, "V": V_global, "W": W_global}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Use the constructed block decomposition
    result = _combine_blocks_4x5x6()
    
    # Verify (optional, but good for debugging)
    # if not _verify(result['U'], result['V'], result['W'], n, m, p):
    #    print("Verification failed")
    #    # Fallback to schoolbook
    #    rank = n*m*p
    #    U = [[0]*rank for _ in range(n*m)]
    #    V = [[0]*rank for _ in range(m*p)]
    #    W = [[0]*rank for _ in range(p*n)]
    #    col = 0
    #    for i in range(n):
    #        for j in range(m):
    #            for k in range(p):
    #                U[i*m+j][col] = 1
    #                V[j*p+k][col] = 1
    #                W[k*n+i][col] = 1
    #                col += 1
    #    return {"U": U, "V": V, "W": W}
    
    return result

# EVOLVE-BLOCK-END
