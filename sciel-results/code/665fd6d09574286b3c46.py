import time
import numpy as np

def construct():
    """
    Constructs factor matrices U, V, W for 5x5x5 matrix multiplication tensor.
    Uses block decomposition: partition [3, 2] for rows, cols, and output cols.
    3x3x3 block: schoolbook (rank 9)
    2x2x2 block: Strassen (rank 7)
    Cross blocks: schoolbook
    Total rank: 9 + 7 + 4*(3*2) = 66
    """
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Block partition: sizes [3, 2], starts [0, 3]
    block_sizes = [3, 2]
    block_starts = [0, 3]
    
    terms = []  # (i, j, k, u_coeff, v_coeff, w_coeff)
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append((i, j, k, 1, 1, 1))
    
    def add_strassen(i_start, j_start, k_start):
        """
        Add 7 Strassen rank-1 tensor terms for a 2x2x2 block.
        M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=(A00+A01)B11,
        M4=A11(B01-B11), M5=A00(B00-B10), M6=A10(B11-B10), M7=A01(B01-B00)
        C00=M1+M4-M7, C01=M3+M7, C10=M2+M4, C11=M1-M2+M3+M5
        """
        s = [
            (0, 0, 0, 1,  1,  1),
            (0, 1, 0, 1,  1, -1),
            (0, 0, 1, 1,  1,  1),
            (0, 1, 1, 1,  1,  1),
            (1, 0, 0, 1,  1,  1),
            (1, 1, 0, 1, -1,  1),
            (1, 0, 1, 1,  1,  1),
            (1, 1, 1, 1,  1, -1),
            (0, 0, 1, 1,  1,  1),
            (1, 0, 1, 1,  1,  1),
            (1, 1, 0, 1, -1,  1),
            (0, 1, 1, 1,  1,  1),
            (1, 1, 1, 1,  1, -1),
            (0, 0, 0, 1,  1,  1),
            (1, 0, 1, 1,  1,  1),
            (1, 1, 0, 1,  1,  1),
            (1, 1, 1, 1,  1,  1),
        ]
        for (li, lj, lk, cu, cv, cw) in s:
            terms.append((i_start + li, j_start + lj, k_start + lk, cu, cv, cw))
    
    # Process all block combinations
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                if time.time() > deadline:
                    break
                i_s = block_starts[ib]
                i_e = i_s + block_sizes[ib]
                j_s = block_starts[jb]
                j_e = j_s + block_sizes[jb]
                k_s = block_starts[kb]
                k_e = k_s + block_sizes[kb]
                
                if block_sizes[ib] == 2 and block_sizes[jb] == 2 and block_sizes[kb] == 2:
                    add_strassen(i_s, j_s, k_s)
                else:
                    add_schoolbook(i_s, i_e, j_s, j_e, k_s, k_e)
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    for c, (i, j, k, cu, cv, cw) in enumerate(terms):
        idx_u = i * m + j      # row i, col j of A
        idx_v = j * p + k      # row j, col k of B
        idx_w = k * n + i      # row k, col i of C (trace convention W[k*n+i, r])
        U[idx_u, c] = cu
        V[idx_v, c] = cv
        W[idx_w, c] = cw
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
