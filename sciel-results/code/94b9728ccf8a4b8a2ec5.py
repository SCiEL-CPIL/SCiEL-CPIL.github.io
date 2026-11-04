import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 145
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank

    # Try rank 51 decomposition
    # Block decomposition for 2x3 x 3x3 with rank 17
    # Repeated 3 times for the 9 columns
    
    rank_block = 17
    U_b = np.zeros((6, rank_block))
    V_b = np.zeros((9, rank_block))
    W_b = np.zeros((6, rank_block))
    
    col = 0
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            U_b[idx, col] += val
        for idx, val in v_vals.items():
            V_b[idx, col] += val
        for idx, val in w_vals.items():
            W_b[idx, col] += val
        col += 1
        
    # Strassen terms for j,k in {0,1}
    # M1
    add_term({0:1, 3:1}, {0:1, 4:1}, {0:1, 3:1})
    # M2
    add_term({3:1, 4:1}, {0:1}, {1:1, 3:-1})
    # M3
    add_term({0:1}, {1:1, 4:-1}, {2:1, 3:1})
    # M4
    add_term({4:1}, {3:1, 0:-1}, {0:1, 1:1})
    # M5
    add_term({0:1, 1:1}, {4:1}, {0:-1, 2:1})
    # M6
    add_term({3:1, 0:-1}, {0:1, 1:1}, {3:1})
    # M7
    add_term({1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Schoolbook for j in {0,1}, k=2
    add_term({0:1}, {2:1}, {4:1}) # a00 b02 -> c02
    add_term({1:1}, {5:1}, {4:1}) # a01 b12 -> c02
    add_term({3:1}, {2:1}, {5:1}) # a10 b02 -> c12
    add_term({4:1}, {5:1}, {5:1}) # a11 b12 -> c12
    
    # Schoolbook for j=2, k in {0,1,2}
    add_term({2:1}, {6:1}, {0:1}) # a02 b20 -> c00
    add_term({2:1}, {7:1}, {2:1}) # a02 b21 -> c01
    add_term({2:1}, {8:1}, {4:1}) # a02 b22 -> c02
    add_term({5:1}, {6:1}, {1:1}) # a12 b20 -> c10
    add_term({5:1}, {7:1}, {3:1}) # a12 b21 -> c11
    add_term({5:1}, {8:1}, {5:1}) # a12 b22 -> c12
    
    # Verify block
    err = 0.0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a = i*3+j
                b = j*3+k
                c = k*2+i
                val = np.dot(U_b[a,:], V_b[b,:] * W_b[c,:])
                err += (val - 1)**2
    for a in range(6):
        for b in range(9):
            for c in range(6):
                i = a // 3
                j = a % 3
                j2 = b // 3
                k = b % 3
                k2 = c // 2
                i2 = c % 2
                if not (j == j2 and k == k2 and i == i2):
                    val = np.dot(U_b[a,:], V_b[b,:] * W_b[c,:])
                    err += val**2
    
    if err < 1e-10:
        full_rank = 51
        U_full = np.zeros((6, full_rank))
        V_full = np.zeros((27, full_rank))
        W_full = np.zeros((18, full_rank))
        
        for block_idx in range(3):
            k_offset = block_idx * 3
            for c in range(17):
                full_c = block_idx * 17 + c
                U_full[:, full_c] = U_b[:, c]
                
                for j in range(3):
                    for k in range(3):
                        v_block_idx = j * 3 + k
                        v_full_idx = j * 9 + (k_offset + k)
                        V_full[v_full_idx, full_c] = V_b[v_block_idx, c]
                        
                for k in range(3):
                    for i in range(2):
                        w_block_idx = k * 2 + i
                        w_full_idx = (k_offset + k) * 2 + i
                        W_full[w_full_idx, full_c] = W_b[w_block_idx, c]
        
        # Verify full
        valid = True
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    a = i*3+j
                    b = j*9+k
                    c = k*2+i
                    val = np.dot(U_full[a,:], V_full[b,:] * W_full[c,:])
                    if abs(val - 1) > 1e-10:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            # Check invalid terms
            for a in range(6):
                for b in range(27):
                    for c in range(18):
                        i = a // 3
                        j = a % 3
                        j2 = b // 9
                        k = b % 9
                        k2 = c // 2
                        i2 = c % 2
                        if not (j == j2 and k == k2 and i == i2):
                            val = np.dot(U_full[a,:], V_full[b,:] * W_full[c,:])
                            if abs(val) > 1e-10:
                                valid = False
                                break
                    if not valid: break
                if not valid: break
                
            if valid:
                best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                best_rank = full_rank

    return best_result
