import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Trivial solution
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
                
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = rank_trivial

    # Block decomposition strategy
    # Partition i into {0,1} and {2}
    # Partition j into {0,1}, {2,3}, {4}
    # Partition k into {0,1}, {2,3}, {4,5}
    
    # Blocks:
    # 1. i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 (Strassen, rank 7)
    # 2. i in {0,1}, j in {2,3}, k in {0,1} -> 2x2x2 (Strassen, rank 7)
    # 3. i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2 (Strassen, rank 7)
    # 4. i in {0,1}, j in {2,3}, k in {2,3} -> 2x2x2 (Strassen, rank 7)
    # 5. i in {0,1}, j in {0,1}, k in {4,5} -> 2x2x2 (Strassen, rank 7)
    # 6. i in {0,1}, j in {2,3}, k in {4,5} -> 2x2x2 (Strassen, rank 7)
    # 7. i in {0,1}, j in {4}, k in {0..5} -> 2x1x6 (Rank 1)
    # 8. i in {2}, j in {0..4}, k in {0..5} -> 1x5x6 (Rank 1)
    
    # Total Rank = 6*7 + 1 + 1 = 44
    
    rank_block = 44
    U_block = [[0] * rank_block for _ in range(n * m)]
    V_block = [[0] * rank_block for _ in range(m * p)]
    W_block = [[0] * rank_block for _ in range(p * n)]
    
    current_rank = 0
    
    # Strassen factors for 2x2x2
    # U_local (4x7), V_local (4x7), W_local (4x7)
    # Indices:
    # U: i in {0,1}, j in {0,1} -> idx = i*2 + j
    # V: j in {0,1}, k in {0,1} -> idx = j*2 + k
    # W: k in {0,1}, i in {0,1} -> idx = k*2 + i
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [0, 1, 0, 1, 0, 1, -1], # i=1, j=0
        [0, 0, -1, 1, 0, 0, 1]  # i=1, j=1
    ]
    # Wait, let's re-verify U_strassen from my derivation.
    # r=1: U[0,0]=1, U[1,1]=1. (idx 0, 3)
    # r=2: U[1,0]=1, U[1,1]=1. (idx 2, 3)
    # r=3: U[0,0]=1. (idx 0)
    # r=4: U[1,1]=1. (idx 3)
    # r=5: U[0,0]=1, U[0,1]=1. (idx 0, 1)
    # r=6: U[1,0]=1, U[0,0]=-1. (idx 2, 0)
    # r=7: U[0,1]=1, U[1,1]=-1. (idx 1, 3)
    
    # My list above:
    # idx 0 (0,0): r1=1, r3=1, r5=1, r6=-1. -> [1, 0, 1, 0, 1, -1, 0]. Correct.
    # idx 1 (0,1): r5=1, r7=1. -> [0, 0, 0, 0, 1, 0, 1]. Correct.
    # idx 2 (1,0): r2=1, r6=1. -> [0, 1, 0, 0, 0, 1, 0]. 
    #    My list: [0, 1, 0, 1, 0, 1, -1]. Mismatch at r4?
    #    r4: U[1,1]=1. So idx 2 should be 0 for r4.
    #    My list has 1 at r4 for idx 2. Error.
    #    Let's rebuild U_strassen carefully.
    
    U_s = [[0]*7 for _ in range(4)]
    # r=0 (m1)
    U_s[0][0] = 1; U_s[3][0] = 1
    # r=1 (m2)
    U_s[2][1] = 1; U_s[3][1] = 1
    # r=2 (m3)
    U_s[0][2] = 1
    # r=3 (m4)
    U_s[3][3] = 1
    # r=4 (m5)
    U_s[0][4] = 1; U_s[1][4] = 1
    # r=5 (m6)
    U_s[2][5] = 1; U_s[0][5] = -1
    # r=6 (m7)
    U_s[1][6] = 1; U_s[3][6] = -1
    
    V_s = [[0]*7 for _ in range(4)]
    # r=0 (m1)
    V_s[0][0] = 1; V_s[3][0] = 1
    # r=1 (m2)
    V_s[0][1] = 1
    # r=2 (m3)
    V_s[1][2] = 1; V_s[3][2] = -1
    # r=3 (m4)
    V_s[2][3] = 1; V_s[0][3] = -1
    # r=4 (m5)
    V_s[3][4] = 1
    # r=5 (m6)
    V_s[0][5] = 1; V_s[1][5] = 1
    # r=6 (m7)
    V_s[2][6] = 1; V_s[3][6] = 1
    
    W_s = [[0]*7 for _ in range(4)]
    # r=0 (m1)
    W_s[0][0] = 1; W_s[3][0] = 1
    # r=1 (m2)
    W_s[1][1] = 1
    # r=2 (m3)
    W_s[2][2] = 1
    # r=3 (m4)
    W_s[1][3] = 1
    # r=4 (m5)
    W_s[0][4] = -1
    # r=5 (m6)
    W_s[3][5] = 1
    # r=6 (m7)
    W_s[0][6] = 1
    
    def add_block_2x2x2(i_offset, j_offset, k_offset, rank_start):
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_val = U_s[i*2+j][r]
                    if u_val != 0:
                        U_block[(i_offset+i)*m + (j_offset+j)][rank_start+r] = u_val
            for j in range(2):
                for k in range(2):
                    v_val = V_s[j*2+k][r]
                    if v_val != 0:
                        V_block[(j_offset+j)*p + (k_offset+k)][rank_start+r] = v_val
            for k in range(2):
                for i in range(2):
                    w_val = W_s[k*2+i][r]
                    if w_val != 0:
                        W_block[(k_offset+k)*n + (i_offset+i)][rank_start+r] = w_val

    # Block 1
    add_block_2x2x2(0, 0, 0, 0)
    # Block 2
    add_block_2x2x2(0, 2, 0, 7)
    # Block 3
    add_block_2x2x2(0, 0, 2, 14)
    # Block 4
    add_block_2x2x2(0, 2, 2, 21)
    # Block 5
    add_block_2x2x2(0, 0, 4, 28)
    # Block 6
    add_block_2x2x2(0, 2, 4, 35)
    
    # Block 7: i in {0,1}, j=4, k in {0..5}. Rank 1.
    # T[i,4,k] = 1.
    # U[i,4,36] = 1, V[4,k,36] = 1, W[k,i,36] = 1?
    # Sum: U[i,4] V[4,k] W[k,i].
    # If U=1, V=1, W=1, then 1*1*1 = 1.
    for i in range(2):
        U_block[i*m + 4][36] = 1
    for k in range(6):
        V_block[4*p + k][36] = 1
    for k in range(6):
        for i in range(2):
            W_block[k*n + i][36] = 1

    # Block 8: i=2, j in {0..4}, k in {0..5}. Rank 1.
    # T[2,j,k] = 1.
    U_block[2*m + 0][37] = 1 # Any j? No, U index is i*m+j.
    # We need sum_j U[2,j] V[j,k] W[k,2] = 1.
    # If we set U[2,0]=1, V[0,k]=1, W[k,2]=1?
    # Then sum_j: only j=0 contributes. 1*1*1 = 1.
    # But we need it for all j?
    # The tensor T[2,j,k] is 1 for all j,k.
    # The decomposition must satisfy T[2,j,k] = sum_r U[2,j,r] V[j,k,r] W[k,2,r].
    # If we use rank 1: U[2,j,37]=1, V[j,k,37]=1, W[k,2,37]=1.
    # Then sum_r (only r=37) = 1*1*1 = 1.
    # This works for all j,k.
    for j in range(5):
        U_block[2*m + j][37] = 1
    for j in range(5):
        for k in range(6):
            V_block[j*p + k][37] = 1
    for k in range(6):
        W_block[k*n + 2][37] = 1

    # Verify
    Ur = np.array(U_block).reshape(n, m, -1)
    Vr = np.array(V_block).reshape(m, p, -1)
    Wr = np.array(W_block).reshape(p, n, -1)
    T_check = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    
    if np.allclose(T_check, np.ones((n, m, p)), atol=1e-9):
        return {"U": U_block, "V": V_block, "W": W_block}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
