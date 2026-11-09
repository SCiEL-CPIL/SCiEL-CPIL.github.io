import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Rank for 4x4x4 Strassen is 49
    # Rank for 4x4x2 Schoolbook is 32
    # Total Rank = 81
    max_rank = 100
    
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Define the 7 outer Strassen steps for 4x4x4
    # Each step is (A_blocks, B_blocks, C_blocks)
    # A_blocks: {(br, bc): coeff}
    # B_blocks: {(br, bc): coeff}
    # C_blocks: {(br, bc): coeff}
    
    outer_steps = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(0,1): 1, (1,1): 1}, {(0,0): 1}),
        # M5
        ({(0,0): 1, (1,0): 1}, {(1,1): 1, (1,0): -1}, {(0,0): -1, (1,0): 1}),
        # M6
        ({(1,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # M7
        ({(0,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): -1, (0,1): 1})
    ]
    
    # Define the 7 inner Strassen steps for 2x2x2
    # Each step is (X_coeffs, Y_coeffs, Z_coeffs)
    # X_coeffs: {(r, c): coeff}
    # Y_coeffs: {(r, c): coeff}
    # Z_coeffs: {(r, c): coeff}
    
    inner_steps = [
        # P1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # P3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # P4
        ({(1,1): 1}, {(0,1): 1, (1,1): 1}, {(0,0): 1}),
        # P5
        ({(0,0): 1, (1,0): 1}, {(1,1): 1, (1,0): -1}, {(0,0): -1, (1,0): 1}),
        # P6
        ({(1,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # P7
        ({(0,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): -1, (0,1): 1})
    ]
    
    # Generate the 49 terms for 4x4x4
    for a_blocks, b_blocks, c_blocks in outer_steps:
        for x_coeff, y_coeff, z_coeff in inner_steps:
            # Add one rank-1 term
            # U (A)
            # X[r,c] = sum_{(br,bc)} a_blocks[br,bc] * A[br,bc][r,c]
            # Term uses X[r,c] with coeff x_coeff[r,c]
            # So A[br,bc][r,c] gets coeff a_blocks[br,bc] * x_coeff[r,c]
            for (br, bc), val_a in a_blocks.items():
                for (r, c), val_x in x_coeff.items():
                    global_i = br * 2 + r
                    global_j = bc * 2 + c
                    U[global_i * m + global_j][current_rank] += val_a * val_x
            
            # V (B)
            # Y[r,c] = sum_{(br,bc)} b_blocks[br,bc] * B[br,bc][r,c]
            # Term uses Y[r,c] with coeff y_coeff[r,c]
            # So B[br,bc][r,c] gets coeff b_blocks[br,bc] * y_coeff[r,c]
            for (br, bc), val_b in b_blocks.items():
                for (r, c), val_y in y_coeff.items():
                    global_j = br * 2 + r
                    global_k = bc * 2 + c
                    V[global_j * p + global_k][current_rank] += val_b * val_y
            
            # W (C)
            # Z[r,c] gets coeff z_coeff[r,c] from the inner step.
            # The outer step places Z into C blocks with coeff c_blocks[br,bc].
            # So C[br,bc][r,c] gets coeff c_blocks[br,bc] * z_coeff[r,c].
            # W index is k*n + i.
            # C[br,bc] corresponds to global i = 2*br + r, k = 2*bc + c.
            for (br, bc), val_c in c_blocks.items():
                for (r, c), val_z in z_coeff.items():
                    global_i = br * 2 + r
                    global_k = bc * 2 + c
                    W[global_k * n + global_i][current_rank] += val_c * val_z
            
            current_rank += 1
            if current_rank >= max_rank:
                break
        if current_rank >= max_rank:
            break
    
    # Part 2: 4x4x2 using Schoolbook (Rank 32)
    # Columns 4 and 5 of B (indices 4, 5)
    # C[i, k] = sum_j A[i, j] B[j, k]
    for k in [4, 5]:
        for i in range(n):
            for j in range(m):
                if current_rank >= max_rank: break
                U[i * m + j][current_rank] = 1
                V[j * p + k][current_rank] = 1
                W[k * n + i][current_rank] = 1
                current_rank += 1
            if current_rank >= max_rank: break
        if current_rank >= max_rank: break
        if time.time() > deadline - 5:
            break
    
    # Trim to actual rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    # Verification
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        return schoolbook(n, m, p)
        
    return {"U": U, "V": V, "W": W}

def schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_mat = np.array(U).T # (rank, n*m)
    V_mat = np.array(V).T # (rank, m*p)
    W_mat = np.array(W).T # (rank, p*n)
    
    recon = np.zeros((n, m, p), dtype=int)
    for c in range(rank):
        u_vec = U_mat[c].reshape(n, m)
        v_vec = V_mat[c].reshape(m, p)
        w_vec = W_mat[c].reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
                    
    if not np.all(recon == 1):
        return False
    return True
