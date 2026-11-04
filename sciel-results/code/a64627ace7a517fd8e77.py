import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    # Build the target tensor for 2x3x6 multiplication
    # T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i

    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))

    # Schoolbook solution: rank = n*m*p = 36
    best_rank = n * m * p
    rank = best_rank
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Rectangular Strassen: partition A (2x3) into A1 (2x2) and A2 (2x1)
    # Partition B (3x6) into B1 (2x6) and B2 (1x6)
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 x 2x6 -> 2x6, use 3 Strassen blocks (each 2x2x2) = rank 21
    # A2*B2: 2x1 x 1x6 -> 2x6, rank 12
    # Total: 33
    total_rank = 33
    U_rect = np.zeros((num_A, total_rank))
    V_rect = np.zeros((num_B, total_rank))
    W_rect = np.zeros((num_C, total_rank))
    col_offset = 0

    # A1*B1: 2x2 x 2x6 -> 2x6
    # A1 indices: {0,1,3,4} (a00,a01,a10,a11)
    # B1 indices: rows 0,1 of B -> {0..11}
    # C indices: all 12 (k=0..5, i=0,1)
    
    # Split B1 into 3 blocks of 2 columns each
    for block in range(3):
        # B1 block columns: 2*block, 2*block+1
        k0 = 2 * block
        k1 = 2 * block + 1
        # B1a indices: B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        b_map = [0 * 6 + k0, 0 * 6 + k1, 1 * 6 + k0, 1 * 6 + k1]
        # C indices for this block: c = k*2+i for k=k0,k1
        c_map = [k0 * 2 + 0, k0 * 2 + 1, k1 * 2 + 0, k1 * 2 + 1]
        a_map = [0, 1, 3, 4]  # a00, a01, a10, a11
        
        # Strassen 2x2x2
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (a00+a11)(b00+b11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # M2 = (a10+a11)b00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1
        
        # M3 = a00(b10-b11)
        U_s[0, 2] = 1
        V_s[2, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = -1
        
        # M4 = a11(b00-b10)
        U_s[3, 3] = 1
        V_s[0, 3] = 1; V_s[2, 3] = -1
        W_s[0, 3] = -1; W_s[2, 3] = 1
        
        # M5 = (a00+a01)b11
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[1, 4] = 1; W_s[3, 4] = -1
        
        # M6 = (a01-a10)(b00+b10)
        U_s[1, 5] = 1; U_s[2, 5] = -1
        V_s[0, 5] = 1; V_s[2, 5] = 1
        W_s[0, 5] = -1; W_s[2, 5] = -1
        
        # M7 = (a10-a01)(b11+b01)
        U_s[2, 6] = 1; U_s[1, 6] = -1
        V_s[3, 6] = 1; V_s[1, 6] = 1
        W_s[0, 6] = 1; W_s[1, 6] = 1
        
        for r in range(7):
            for li, gi in enumerate(a_map):
                U_rect[gi, col_offset + r] += U_s[li, r]
            for li, gi in enumerate(b_map):
                V_rect[gi, col_offset + r] += V_s[li, r]
            for li, gi in enumerate(c_map):
                W_rect[gi, col_offset + r] += W_s[li, r]
        col_offset += 7

    # A2*B2: 2x1 x 1x6 -> 2x6, rank 12
    # A2 indices: {2, 5} (a02, a12)
    # B2 indices: row 2 of B -> {12..17}
    # C indices: all 12
    
    a_map2 = [2, 5]
    for k in range(6):
        b_idx = 2 * 6 + k  # B[2,k]
        c0_idx = k * 2 + 0  # C[k,0]
        c1_idx = k * 2 + 1  # C[k,1]
        
        U_rect[a_map2[0], col_offset] = 1
        V_rect[b_idx, col_offset] = 1
        W_rect[c0_idx, col_offset] = 1
        col_offset += 1
        
        U_rect[a_map2[1], col_offset] = 1
        V_rect[b_idx, col_offset] = 1
        W_rect[c1_idx, col_offset] = 1
        col_offset += 1

    # Verify rank-33 solution
    max_err = 0
    for (a, b, c) in valid_triples:
        val = np.sum(U_rect[a, :] * V_rect[b, :] * W_rect[c, :])
        max_err = max(max_err, abs(val - 1.0))
    
    for a in range(num_A):
        for b in range(num_B):
            for c in range(num_C):
                if (a, b, c) not in valid_triples:
                    val = np.sum(U_rect[a, :] * V_rect[b, :] * W_rect[c, :])
                    max_err = max(max_err, abs(val))
    
    if max_err < 1e-10:
        best_rank = 33
        best_U, best_V, best_W = U_rect.copy(), V_rect.copy(), W_rect.copy()

    # Try to find lower rank using gradient descent on the rank-33 solution as initialization
    for target_rank in [28, 29, 30, 31, 32]:
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(30):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(12345 + target_rank * 100 + restart)
            U_curr = best_U[:, :target_rank].copy() + np.random.randn(num_A, target_rank) * 0.1
            V_curr = best_V[:, :target_rank].copy() + np.random.randn(num_B, target_rank) * 0.1
            W_curr = best_W[:, :target_rank].copy() + np.random.randn(num_C, target_rank) * 0.1
            
            lr = 0.005
            for step in range(500):
                if time.time() > deadline - 2:
                    break
                    
                grad_U = np.zeros_like(U_curr)
                grad_V = np.zeros_like(V_curr)
                grad_W = np.zeros_like(W_curr)
                residual_sum = 0
                
                for (a, b, c) in valid_triples:
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    err = val - 1.0
                    residual_sum += err * err
                    
                    for r in range(target_rank):
                        grad_U[a, r] += err * V_curr[b, r] * W_curr[c, r]
                        grad_V[b, r] += err * U_curr[a, r] * W_curr[c, r]
                        grad_W[c, r] += err * U_curr[a, r] * V_curr[b, r]
                
                U_curr -= lr * grad_U
                V_curr -= lr * grad_V
                W_curr -= lr * grad_W
                lr *= 0.995
                
                if residual_sum < 1e-8:
                    found = True
                    break
            
            if found:
                max_err = 0
                for (a, b, c) in valid_triples:
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                    break
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
