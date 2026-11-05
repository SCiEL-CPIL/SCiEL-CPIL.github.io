import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W factors for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    A: 00=0, 01=1, 10=2, 11=3
    B: 00=0, 01=1, 10=2, 11=3
    C: 00=0, 01=1, 10=2, 11=3 (Note: C index is k*n+i, so 00=0, 01=1, 10=2, 11=3 is correct for n=2)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1 # C00, C10
    
    # Term 2: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C10, C11
    
    # Term 3: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[3, 2] = 1 # C11
    
    # Term 4: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1 # C00, C01
    
    # Term 5: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1 # C01, C11
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1 # C00
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1 # C00, C11
    
    return U, V, W

def _rank2_2x1x2():
    """
    Returns U, V, W factors for 2x1x2 matrix multiplication with rank 2.
    A: 2x1, B: 1x2, C: 2x2
    U: (2, 2), V: (2, 2), W: (4, 2)
    Indices:
    A: 0=0, 1=1
    B: 0=0, 1=1
    C: 00=0, 01=1, 10=2, 11=3 (k*n+i)
    """
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((4, 2))
    
    # Term 0: i'=0
    U[0, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1 # C00, C10 (k=0,i=0 and k=1,i=0)
    
    # Term 1: i'=1
    U[1, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1 # C01, C11 (k=0,i=1 and k=1,i=1)
    
    return U, V, W

def _build_block_strassen(n, m, p):
    """
    Build block Strassen decomposition for 4x5x6.
    """
    if (n, m, p) != (4, 5, 6):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    total_rank = 96
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    SU, SV, SW = _strassen_2x2x2()
    RU, RV, RW = _rank2_2x1x2()
    
    # Row groups for A (and C): [0:2], [2:4]
    a_row_groups = [(0, 2), (2, 4)]
    # Col groups for A: [0:2], [2:4], [4:5]
    a_col_groups = [(0, 2), (2, 4), (4, 5)]
    # Row groups for B: [0:2], [2:4], [4:5]
    b_row_groups = [(0, 2), (2, 4), (4, 5)]
    # Col groups for B (and C): [0:2], [2:4], [4:6]
    b_col_groups = [(0, 2), (2, 4), (4, 6)]
    
    col_offset = 0
    
    for ri, (r_start, r_end) in enumerate(a_row_groups):
        for ci, (c_start, c_end) in enumerate(a_col_groups):
            bi = ci
            k_start, k_end = b_row_groups[bi]
            
            a_size_r = r_end - r_start
            a_size_c = c_end - c_start
            b_size_r = k_end - k_start
            assert a_size_c == b_size_r
            
            for di, (d_start, d_end) in enumerate(b_col_groups):
                b_size_c = d_end - d_start
                
                if a_size_r == 2 and a_size_c == 2 and b_size_c == 2:
                    block_rank = 7
                    for lr in range(7):
                        # Map A
                        for la in range(4):
                            local_i = la // 2
                            local_j = la % 2
                            global_i = r_start + local_i
                            global_j = c_start + local_j
                            U[global_i * m + global_j, col_offset + lr] = SU[la, lr]
                        
                        # Map B
                        for lb in range(4):
                            local_k = lb // 2
                            local_l = lb % 2
                            global_k = k_start + local_k
                            global_l = d_start + local_l
                            V[global_k * p + global_l, col_offset + lr] = SV[lb, lr]
                        
                        # Map C (W)
                        for lc in range(4):
                            local_kc = lc // 2
                            local_ic = lc % 2
                            global_k = k_start + local_kc
                            global_i = r_start + local_ic
                            W[global_k * n + global_i, col_offset + lr] = SW[lc, lr]
                
                elif a_size_r == 2 and a_size_c == 1 and b_size_c == 2:
                    block_rank = 2
                    for lr in range(2):
                        # Map A
                        for la in range(2):
                            local_i = la
                            global_i = r_start + local_i
                            global_j = c_start
                            U[global_i * m + global_j, col_offset + lr] = RU[la, lr]
                        
                        # Map B
                        for lb in range(2):
                            local_l = lb
                            global_k = k_start
                            global_l = d_start + local_l
                            V[global_k * p + global_l, col_offset + lr] = RV[lb, lr]
                        
                        # Map C (W)
                        for lc in range(4):
                            local_kc = lc // 2
                            local_ic = lc % 2
                            global_k = k_start + local_kc
                            global_i = r_start + local_ic
                            W[global_k * n + global_i, col_offset + lr] = RW[lc, lr]
                
                col_offset += block_rank
    
    assert col_offset == total_rank, f"col_offset={col_offset}, total_rank={total_rank}"
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_block_strassen(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
