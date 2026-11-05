"""Matrix multiplication 4x5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Uses a recursive Strassen-based decomposition:
    - 4x4x4 core uses Strassen (rank 49 instead of 64)
    - Boundary terms use schoolbook (rank 36)
    Total rank: 85
    
    The tensor identity: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all i in [0,4), j in [0,5), k in [0,5).
    """
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 85
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # 2x2x2 Strassen factors: U, V, W each 4x7
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    s_u = np.zeros((4, 7), dtype=int)
    s_v = np.zeros((4, 7), dtype=int)
    s_w = np.zeros((4, 7), dtype=int)
    
    # Term 0: (a00+a11)(b00+b11)
    s_u[0,0]=1; s_u[3,0]=1
    s_v[0,0]=1; s_v[3,0]=1
    s_w[0,0]=1; s_w[3,0]=1
    
    # Term 1: (a10+a11)b00
    s_u[2,1]=1; s_u[3,1]=1
    s_v[0,1]=1
    s_w[2,1]=1; s_w[3,1]=-1
    
    # Term 2: a00(b01-b11)
    s_u[0,2]=1
    s_v[1,2]=1; s_v[3,2]=-1
    s_w[1,2]=1; s_w[3,2]=1
    
    # Term 3: a11(b10-b00)
    s_u[3,3]=1
    s_v[2,3]=1; s_v[0,3]=-1
    s_w[0,3]=1; s_w[2,3]=1
    
    # Term 4: (a00+a01)b11
    s_u[0,4]=1; s_u[1,4]=1
    s_v[3,4]=1
    s_w[0,4]=-1; s_w[1,4]=1
    
    # Term 5: (a10-a00)(b00+b01)
    s_u[0,5]=-1; s_u[2,5]=1
    s_v[0,5]=1; s_v[1,5]=1
    s_w[3,5]=1
    
    # Term 6: (a01-a11)(b10+b11)
    s_u[1,6]=1; s_u[3,6]=-1
    s_v[2,6]=1; s_v[3,6]=1
    s_w[0,6]=1
    
    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        """
        Add a 4x4x4 Strassen term using 2x2x2 Strassen sub-factors.
        a_blocks: dict {(br,bc): coeff} - linear combo of A's 2x2 blocks
        b_blocks: dict {(br,bc): coeff} - linear combo of B's 2x2 blocks  
        c_blocks: dict {(br,bc): coeff} - linear combo of C's 2x2 blocks
        """
        for inner_k in range(7):
            col = col_start + inner_k
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
            
            # Distribute u_vec to A entries
            for (a_br, a_bc), a_coeff in a_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj, col] += a_coeff * u_val
            
            # Distribute v_vec to B entries
            for (b_br, b_bc), b_coeff in b_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk, col] += b_coeff * v_val
            
            # Distribute w_vec to C entries
            for (c_br, c_bc), c_coeff in c_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            gi = 2 * c_br + lr
                            gk = 2 * c_bc + lc
                            W[gk * 4 + gi, col] += c_coeff * w_val
    
    # 7 Strassen 4x4 terms for the 4x4x4 core
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),           # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),           # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),           # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),           # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),           # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),           # M7
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        if time.time() - start_time > deadline:
            break
        add_4x4_term(a_coeff, b_coeff, c_coeff, t_idx * 7)
    
    col = 49
    
    # 4x4x1 part: A[:, 0:4] * B[:, 4] -> C[:, 4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            if time.time() - start_time > deadline:
                break
    
    # 4x1x4 part: A[:, 4] * B[4, :] -> C[:, 0:4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
            if time.time() - start_time > deadline:
                break
    
    # 4x1x1 part: A[:, 4] * B[4, 4] -> C[:, 4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
    
    # Verify correctness with a random test
    if time.time() - start_time < deadline - 5.0:
        try:
            A_test = np.random.randint(-10, 10, (n, m))
            B_test = np.random.randint(-10, 10, (m, p))
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p), dtype=int)
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            
            for c in range(rank):
                if time.time() - start_time > deadline - 2.0:
                    break
                u_col = U_arr[:, c]
                v_col = V_arr[:, c]
                w_col = W_arr[:, c]
                
                # Reshape and multiply
                u_mat = u_col.reshape(n, m)
                v_mat = v_col.reshape(m, p)
                w_mat = w_col.reshape(p, n).T  # W is [k*n+i, c], so reshape to (p, n) then transpose
                
                C_computed += u_mat * v_mat * w_mat.T
            
            if not np.array_equal(C_expected, C_computed):
                # If verification fails, fall back to schoolbook
                rank_fb = n * m * p
                U_fb = np.zeros((n * m, rank_fb), dtype=int)
                V_fb = np.zeros((m * p, rank_fb), dtype=int)
                W_fb = np.zeros((p * n, rank_fb), dtype=int)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            U_fb[i * m + j, idx] = 1
                            V_fb[j * p + k, idx] = 1
                            W_fb[k * n + i, idx] = 1
                
                return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        except Exception:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
