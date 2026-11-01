import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    Uses a block-Strassen approach:
    - Treat the multiplication as [2x2 | 2x1] * [2x10 | 1x10].
    - Use rank 7 Strassen blocks for the 2x2 * 2x10 part (5 blocks).
    - Use rank 1 standard terms for the 2x1 * 1x10 part (20 terms).
    - Total rank: 5*7 + 20 = 55.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Dimensions check
    # n=2, m=3, p=10
    # A is 2x3, B is 3x10, C is 2x10
    
    # We split the middle dimension m=3 into m1=2 and m2=1.
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B = [B1; B2] where B1 is 2x10, B2 is 1x10
    # C = A1*B1 + A2*B2
    
    # Part 1: A1 (2x2) * B1 (2x10)
    # We use 5 Strassen blocks, each handling 2 columns of B1.
    # B1 has 10 columns, so 5 blocks of width 2.
    
    # Part 2: A2 (2x1) * B2 (1x10)
    # This is just 2 * 10 = 20 standard multiplications.
    # A2[0,0] * B2[0,k] -> C[0,k]
    # A2[1,0] * B2[0,k] -> C[1,k]
    
    rank = 5 * 7 + 20
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for 2x2 * 2x10 ---
    
    # Indices for A1 (2x2) in U (size n*m = 2*3 = 6)
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[2]
    # A[1,1] -> U[3]
    
    # Indices for B1 (2x10) in V (size m*p = 3*10 = 30)
    # B[0,0..9] -> V[0..9]
    # B[1,0..9] -> V[10..19]
    
    # Strassen Linear Forms for A (inputs are a, b, c, d)
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # m1: a+d
    # m2: c+d
    # m3: a
    # m4: d
    # m5: a+b
    # m6: c-a
    # m7: b-d
    
    LA = [
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 0, 0, 0], # a
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [0, 0, 1, -1],# c-a
        [0, 1, 0, -1] # b-d
    ]
    
    # Strassen Linear Forms for B (inputs are e, f, g, h for a 2x2 block)
    # For block t (0..4), the inputs are:
    # e = B[0, 2t]
    # f = B[0, 2t+1]
    # g = B[1, 2t]
    # h = B[1, 2t+1]
    
    LB = [
        [1, 0, 0, 1], # e+h
        [1, 0, 0, 0], # e
        [0, 1, 0, -1],# f-h
        [0, -1, 1, 0],# g-e
        [0, 0, 0, 1], # h
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1]  # g+h
    ]
    
    # W matrices for Strassen (2x2 each)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    W_strassen = [
        [[1, 0], [0, 1]], # m1
        [[0, 0], [1, -1]],# m2
        [[0, 1], [0, 1]], # m3
        [[1, 0], [1, 0]], # m4
        [[-1, 1], [0, 0]],# m5
        [[0, 0], [0, 1]], # m6
        [[1, 0], [0, 0]]  # m7
    ]
    
    for block in range(5):
        k_start = block * 2
        k_end = k_start + 2
        
        # V indices for this block
        # e = B[0, k_start] -> V[0*10 + k_start]
        # f = B[0, k_start+1] -> V[0*10 + k_start + 1]
        # g = B[1, k_start] -> V[1*10 + k_start]
        # h = B[1, k_start+1] -> V[1*10 + k_start + 1]
        
        for r in range(7):
            # U contributions
            for ai in range(4):
                if LA[r][ai] != 0:
                    U[ai][col] += LA[r][ai]
            
            # V contributions
            # Map local indices 0,1,2,3 to global V indices
            v_indices = [
                0 * 10 + k_start,
                0 * 10 + k_start + 1,
                1 * 10 + k_start,
                1 * 10 + k_start + 1
            ]
            for bi in range(4):
                if LB[r][bi] != 0:
                    V[v_indices[bi]][col] += LB[r][bi]
            
            # W contributions
            # Output C is 2x2 block at rows 0,1 and cols k_start, k_start+1
            # W index: k*n + i
            # C[0, k_start] -> W[k_start*n + 0]
            # C[0, k_start+1] -> W[(k_start+1)*n + 0]
            # C[1, k_start] -> W[k_start*n + 1]
            # C[1, k_start+1] -> W[(k_start+1)*n + 1]
            
            w_local_indices = [
                k_start * n + 0,
                (k_start + 1) * n + 0,
                k_start * n + 1,
                (k_start + 1) * n + 1
            ]
            
            # W_strassen[r] is a 2x2 matrix [[w00, w01], [w10, w11]]
            # w00 contributes to C[0, k_start]
            # w01 contributes to C[0, k_start+1]
            # w10 contributes to C[1, k_start]
            # w11 contributes to C[1, k_start+1]
            
            if W_strassen[r][0][0] != 0:
                W[w_local_indices[0]][col] += W_strassen[r][0][0]
            if W_strassen[r][0][1] != 0:
                W[w_local_indices[1]][col] += W_strassen[r][0][1]
            if W_strassen[r][1][0] != 0:
                W[w_local_indices[2]][col] += W_strassen[r][1][0]
            if W_strassen[r][1][1] != 0:
                W[w_local_indices[3]][col] += W_strassen[r][1][1]
            
            col += 1
    
    # --- Part 2: Standard for 2x1 * 1x10 ---
    # A2 is A[0,2] and A[1,2] -> U[2] and U[3]? No.
    # A is 2x3.
    # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2]
    # A[1,0]=U[3], A[1,1]=U[4], A[1,2]=U[5]
    # Wait, my previous mapping for A1 was:
    # a=A[0,0]->U[0], b=A[0,1]->U[1], c=A[1,0]->U[2], d=A[1,1]->U[3]
    # So A2 is A[0,2]->U[2]? No, U index is i*m + j.
    # U[0] = A[0,0]
    # U[1] = A[0,1]
    # U[2] = A[0,2]
    # U[3] = A[1,0]
    # U[4] = A[1,1]
    # U[5] = A[1,2]
    
    # Let's re-verify the U indices used in Part 1.
    # In Part 1, I used:
    # LA[r][0] -> U[0] (A[0,0])
    # LA[r][1] -> U[1] (A[0,1])
    # LA[r][2] -> U[2] (A[1,0]) -> ERROR. A[1,0] is U[3].
    # LA[r][3] -> U[3] (A[1,1]) -> ERROR. A[1,1] is U[4].
    
    # Correction needed in Part 1 logic or index mapping.
    # Let's fix the U index mapping in Part 1.
    
    # Reset U, V, W and col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Correct U indices for A1 (2x2):
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[3]
    # A[1,1] -> U[4]
    
    for block in range(5):
        k_start = block * 2
        k_end = k_start + 2
        
        v_indices = [
            0 * 10 + k_start,
            0 * 10 + k_start + 1,
            1 * 10 + k_start,
            1 * 10 + k_start + 1
        ]
        
        u_indices = [
            0, # A[0,0]
            1, # A[0,1]
            3, # A[1,0]
            4  # A[1,1]
        ]
        
        for r in range(7):
            # U contributions
            for ai in range(4):
                if LA[r][ai] != 0:
                    U[u_indices[ai]][col] += LA[r][ai]
            
            # V contributions
            for bi in range(4):
                if LB[r][bi] != 0:
                    V[v_indices[bi]][col] += LB[r][bi]
            
            # W contributions
            w_local_indices = [
                k_start * n + 0,
                (k_start + 1) * n + 0,
                k_start * n + 1,
                (k_start + 1) * n + 1
            ]
            
            if W_strassen[r][0][0] != 0:
                W[w_local_indices[0]][col] += W_strassen[r][0][0]
            if W_strassen[r][0][1] != 0:
                W[w_local_indices[1]][col] += W_strassen[r][0][1]
            if W_strassen[r][1][0] != 0:
                W[w_local_indices[2]][col] += W_strassen[r][1][0]
            if W_strassen[r][1][1] != 0:
                W[w_local_indices[3]][col] += W_strassen[r][1][1]
            
            col += 1
    
    # Part 2: A2 (2x1) * B2 (1x10)
    # A2[0] = A[0,2] -> U[2]
    # A2[1] = A[1,2] -> U[5]
    # B2[0,k] = B[2,k] -> V[2*10 + k] = V[20+k]
    
    for k in range(10):
        # Term 1: A[0,2] * B[2,k] -> C[0,k]
        U[2][col] = 1
        V[20 + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: A[1,2] * B[2,k] -> C[1,k]
        U[5][col] = 1
        V[20 + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    if U[u_idx][c] != 0 and V[v_idx][c] != 0 and W[w_idx][c] != 0:
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][c] = 1
                    V_fb[j * p + k][c] = 1
                    W_fb[k * n + i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
