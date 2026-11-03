import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U: i*2 + j
    V: j*2 + k
    W: k*2 + i
    """
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express A and B entries as linear combinations for U and V.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # U vectors (rows correspond to A entries: A00, A01, A10, A11)
    # V vectors (rows correspond to B entries: B00, B01, B10, B11)
    # W vectors (rows correspond to C entries: C00, C01, C10, C11)
    
    # Term 1: (A00+A11)(B00+B11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: C00 += 1, C11 += 1 => [1, 0, 0, 1]
    
    # Term 2: (A10+A11)B00
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: C10 += 1 => [0, 0, 1, 0]
    
    # Term 3: A00(B01-B11)
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: C01 += 1 => [0, 1, 0, 0]
    
    # Term 4: A11(B10-B00)
    # U: [0, 0, 0, 1]
    # V: [0, 0, 1, -1]
    # W: C00 += 1, C10 += 1 => [1, 0, 1, 0]
    
    # Term 5: (A00+A01)B11
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: C00 -= 1, C01 += 1 => [-1, 1, 0, 0]
    
    # Term 6: (A10-A00)(B00+B01)
    # U: [-1, 0, 1, 0]
    # V: [1, 1, 0, 0]
    # W: C11 += 1 => [0, 0, 0, 1]
    
    # Term 7: (A01-A11)(B10+B11)
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # W: C00 += 1, C11 += 1 => [1, 0, 0, 1]
    
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]).T # Shape (4, 7)
    
    V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T # Shape (4, 7)
    
    W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]).T # Shape (4, 7)
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Partition indices
    I_blocks = [[0, 1], [2]]
    J_blocks = [[0, 1], [2, 3], [4]]
    K_blocks = [[0, 1], [2, 3], [4]]
    
    U_list = []
    V_list = []
    W_list = []
    
    for i_idx, i_block in enumerate(I_blocks):
        for j_idx, j_block in enumerate(J_blocks):
            for k_idx, k_block in enumerate(K_blocks):
                ni = len(i_block)
                nj = len(j_block)
                nk = len(k_block)
                
                # Determine if we can use Strassen
                if ni == 2 and nj == 2 and nk == 2:
                    U_s, V_s, W_s = _strassen_2x2x2()
                    # Map local indices to global
                    # U local index: li * 2 + lj
                    # Global U index: i_block[li] * m + j_block[lj]
                    # V local index: lj * 2 + lk
                    # Global V index: j_block[lj] * p + k_block[lk]
                    # W local index: lk * 2 + li
                    # Global W index: k_block[lk] * n + i_block[li]
                    
                    for r in range(7):
                        u_col = [0] * (n * m)
                        v_col = [0] * (m * p)
                        w_col = [0] * (p * n)
                        
                        for li in range(2):
                            for lj in range(2):
                                u_local = li * 2 + lj
                                u_global = i_block[li] * m + j_block[lj]
                                u_col[u_global] = U_s[u_local, r]
                                
                        for lj in range(2):
                            for lk in range(2):
                                v_local = lj * 2 + lk
                                v_global = j_block[lj] * p + k_block[lk]
                                v_col[v_global] = V_s[v_local, r]
                                
                        for lk in range(2):
                            for li in range(2):
                                w_local = lk * 2 + li
                                w_global = k_block[lk] * n + i_block[li]
                                w_col[w_global] = W_s[w_local, r]
                                
                        U_list.append(u_col)
                        V_list.append(v_col)
                        W_list.append(w_col)
                        
                else:
                    # Schoolbook for this block
                    for li in range(ni):
                        for lj in range(nj):
                            for lk in range(nk):
                                u_col = [0] * (n * m)
                                v_col = [0] * (m * p)
                                w_col = [0] * (p * n)
                                
                                u_global = i_block[li] * m + j_block[lj]
                                v_global = j_block[lj] * p + k_block[lk]
                                w_global = k_block[lk] * n + i_block[li]
                                
                                u_col[u_global] = 1
                                v_col[v_global] = 1
                                w_col[w_global] = 1
                                
                                U_list.append(u_col)
                                V_list.append(v_col)
                                W_list.append(w_col)
                                
    rank = len(U_list)
    
    # Convert to lists of lists as required
    U = U_list
    V = V_list
    W = W_list
    
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
