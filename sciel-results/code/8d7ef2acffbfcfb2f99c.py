"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Rank 189 decomposition using Strassen (rank 7) on 2x2 blocks of 3x3 matrices
    # Inner 3x3 multiplication is schoolbook (rank 27)
    # Total rank = 7 * 27 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block structure
    # m_idx: 0 to 6
    # Blocks: (0,0), (0,1), (1,0), (1,1)
    
    # alpha[m_idx][block_row, block_col] for P_m
    # beta[m_idx][block_row, block_col] for Q_m
    # gamma[m_idx][block_row, block_col] for C block coefficients
    
    alpha = [dict() for _ in range(7)]
    beta = [dict() for _ in range(7)]
    gamma = [dict() for _ in range(7)]
    
    # m=0: M0 = (A00 + A11)(B00 + B11)
    alpha[0][(0,0)] = 1
    alpha[0][(1,1)] = 1
    beta[0][(0,0)] = 1
    beta[0][(1,1)] = 1
    
    # m=1: M1 = (A10 + A11)B00
    alpha[1][(1,0)] = 1
    alpha[1][(1,1)] = 1
    beta[1][(0,0)] = 1
    
    # m=2: M2 = A00(B01 - B11)
    alpha[2][(0,0)] = 1
    beta[2][(0,1)] = 1
    beta[2][(1,1)] = -1
    
    # m=3: M3 = A11(B10 - B00)
    alpha[3][(1,1)] = 1
    beta[3][(1,0)] = 1
    beta[3][(0,0)] = -1
    
    # m=4: M4 = (A00 + A01)B11
    alpha[4][(0,0)] = 1
    alpha[4][(0,1)] = 1
    beta[4][(1,1)] = 1
    
    # m=5: M5 = (A10 - A00)(B00 + B01)
    alpha[5][(1,0)] = 1
    alpha[5][(0,0)] = -1
    beta[5][(0,0)] = 1
    beta[5][(0,1)] = 1
    
    # m=6: M6 = (A01 - A11)(B10 + B11)
    alpha[6][(0,1)] = 1
    alpha[6][(1,1)] = -1
    beta[6][(1,0)] = 1
    beta[6][(1,1)] = 1
    
    # Gamma: Coefficients for C blocks
    # C00 = M0 + M3 - M4 + M6
    gamma[0][(0,0)] = 1
    gamma[3][(0,0)] = 1
    gamma[4][(0,0)] = -1
    gamma[6][(0,0)] = 1
    
    # C01 = M2 + M4
    gamma[2][(0,1)] = 1
    gamma[4][(0,1)] = 1
    
    # C10 = M1 + M3
    gamma[1][(1,0)] = 1
    gamma[3][(1,0)] = 1
    
    # C11 = M0 - M1 + M2 + M5
    gamma[0][(1,1)] = 1
    gamma[1][(1,1)] = -1
    gamma[2][(1,1)] = 1
    gamma[5][(1,1)] = 1
    
    # Construct U, V, W
    # Column c = 27 * m_idx + u
    # u corresponds to schoolbook 3x3 term (r_l, t_l, s_l)
    
    for m_idx in range(7):
        for u in range(27):
            c = m_idx * 27 + u
            r_l = u // 9
            t_l = (u // 3) % 3
            s_l = u % 3
            
            # Iterate over block indices
            for i_b in range(2):
                for j_b in range(2):
                    for k_b in range(2):
                        
                        # U: A[i, j]
                        # i = 3*i_b + r_l
                        # j = 3*j_b + t_l
                        i = 3 * i_b + r_l
                        j = 3 * j_b + t_l
                        u_idx = i * m + j
                        val_u = alpha[m_idx].get((i_b, j_b), 0)
                        if val_u != 0:
                            U[u_idx][c] += val_u
                        
                        # V: B[j, k]
                        # j = 3*j_b + t_l
                        # k = 3*k_b + s_l
                        k = 3 * k_b + s_l
                        v_idx = j * p + k
                        val_v = beta[m_idx].get((j_b, k_b), 0)
                        if val_v != 0:
                            V[v_idx][c] += val_v
                        
                        # W: C[k, i]
                        # k = 3*k_b + s_l
                        # i = 3*i_b + r_l
                        w_idx = k * n + i
                        val_w = gamma[m_idx].get((i_b, k_b), 0)
                        if val_w != 0:
                            W[w_idx][c] += val_w

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
