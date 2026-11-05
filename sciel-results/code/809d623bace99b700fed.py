"""Program for 5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5 matrix multiplication
    using a block-partition strategy with Strassen's algorithm for 2x2x2 sub-blocks.
    
    Strategy:
    1. Try multiple partitions of dimension 5.
    2. For each partition, iterate over all sub-blocks.
    3. For 2x2x2 sub-blocks, use Strassen's rank-7 decomposition.
    4. For other sub-blocks, use the schoolbook rank.
    5. Return the decomposition with the minimum rank.
    """
    start_time = time.time()
    deadline = 145.0  # seconds, safety margin from 150s limit
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Try different partitions
    partitions = [
        [2, 2, 1],
        [3, 1, 1],
        [1, 2, 2],
        [1, 1, 3],
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for parts in partitions:
        if time.time() > deadline - 5.0:
            break
            
        offsets = [0]
        for i in range(1, len(parts)):
            offsets.append(offsets[-1] + parts[i-1])
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        def add_col(u_vec, v_vec, w_vec):
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
        
        for a_idx in range(len(parts)):
            for b_idx in range(len(parts)):
                for c_idx in range(len(parts)):
                    size_a = parts[a_idx]
                    size_b = parts[b_idx]
                    size_c = parts[c_idx]
                    
                    off_a = offsets[a_idx]
                    off_b = offsets[b_idx]
                    off_c = offsets[c_idx]
                    
                    if size_a == 2 and size_b == 2 and size_c == 2:
                        # Strassen block for 2x2x2
                        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
                        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                        u_vec[(off_a+0)*5 + (off_b+0)] = 1
                        u_vec[(off_a+1)*5 + (off_b+1)] = 1
                        v_vec[(off_b+0)*5 + (off_c+0)] = 1
                        v_vec[(off_b+1)*5 + (off_c+1)] = 1
                        w_vec[(off_c+0)*5 + (off_a+0)] = 1
                        w_vec[(off_c+1)*5 + (off_a+1)] = 1
                        add_col(u_vec, v_vec, w_vec)
                        
                        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
                        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                        u_vec[(off_a+1)*5 + (off_b+0)] = 1
                        u_vec[(off_a+1)*5 + (off_b+1)] = 1
                        v_vec[(off_b+0)*5 + (off_c+0)] = 1
                        w_vec[(off_c+0)*5 + (off_a+1)] = 1
                        w_vec[(off_c+1)*5 + (off_a+1)] = -1
                        add_col(u_vec, v_vec, w_vec)
                        
                        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
                        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                        u_vec[(off_a+0)*5 + (off_b+0)] = 1
                        v_vec[(off_b+0)*5 + (off_c+1)] = 1
                        v_vec[(off_b+1)*5 + (off_c+1)] = -1
                        w_vec[(off_c+1)*5 + (off_a+0)] = 1
                        w_vec[(off_c+1)*5 + (off_a+1)] = 1
                        add_col(u_vec, v_vec, w_vec)
                        
                        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
                        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                        u_vec[(off_a+1)*5 + (off_b+1)] = 1
                        v_vec[(off_b+1)*5 + (off_c+0)] = 1
                        v_vec[(off_b+0)*5 + (off_c+0)] = -1
                        w_vec[(off_c+0)*5 + (off_a+0)] = 1
                        w_vec[(off_c+0)*5 + (off_a+1)] = 1
                        add_col(u_vec, v_vec, w_vec)
                        
                        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
                        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                        u_vec[(off_a+0)*5 + (off_b+0)] = 1
                        u_vec[(off_a+0)*5 + (off_b+1)] = 1
                        v_vec[(off_b+1)*5 + (off_c+1)] = 1
                        w_vec[(off_c+0)*5 + (off_a+0)] = -1
                        w_vec[(off_c+1)*5 + (off_a+0)] = 1
                        add_col(u_vec, v_vec, w_vec)
                        
                        # M6 = (A10-A00)(B00+B01) -> C11 += M6
                        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                        u_vec[(off_a+1)*5 + (off_b+0)] = 1
                        u_vec[(off_a+0)*5 + (off_b+0)] = -1
                        v_vec[(off_b+0)*5 + (off_c+0)] = 1
                        v_vec[(off_b+0)*5 + (off_c+1)] = 1
                        w_vec[(off_c+1)*5 + (off_a+1)] = 1
                        add_col(u_vec, v_vec, w_vec)
                        
                        # M7 = (A01-A11)(B10+B11) -> C00 += M7
                        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                        u_vec[(off_a+0)*5 + (off_b+1)] = 1
                        u_vec[(off_a+1)*5 + (off_b+1)] = -1
                        v_vec[(off_b+1)*5 + (off_c+0)] = 1
                        v_vec[(off_b+1)*5 + (off_c+1)] = 1
                        w_vec[(off_c+0)*5 + (off_a+0)] = 1
                        add_col(u_vec, v_vec, w_vec)
                        
                    else:
                        # Schoolbook block
                        for u in range(size_a):
                            for v in range(size_b):
                                for w in range(size_c):
                                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                                    gi = off_a + u
                                    gj = off_b + v
                                    gk = off_c + w
                                    
                                    u_vec[gi*5 + gj] = 1
                                    v_vec[gj*5 + gk] = 1
                                    w_vec[gk*5 + gi] = 1
                                    add_col(u_vec, v_vec, w_vec)
        
        rank = len(U_cols)
        
        if rank < best_rank:
            best_rank = rank
            U = [[0]*rank for _ in range(25)]
            V = [[0]*rank for _ in range(25)]
            W = [[0]*rank for _ in range(25)]
            
            for col_idx in range(rank):
                u_col = U_cols[col_idx]
                v_col = V_cols[col_idx]
                w_col = W_cols[col_idx]
                for row_idx in range(25):
                    U[row_idx][col_idx] = u_col[row_idx]
                    V[row_idx][col_idx] = v_col[row_idx]
                    W[row_idx][col_idx] = w_col[row_idx]
            
            best_result = {"U": U, "V": V, "W": W}
    
    if best_result is None:
        # Fallback: pure schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[i*5 + j] = 1
                    v_vec[j*5 + k] = 1
                    w_vec[k*5 + i] = 1
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
        
        rank = 125
        U = [[0]*rank for _ in range(25)]
        V = [[0]*rank for _ in range(25)]
        W = [[0]*rank for _ in range(25)]
        
        for col_idx in range(rank):
            for row_idx in range(25):
                U[row_idx][col_idx] = U_cols[col_idx][row_idx]
                V[row_idx][col_idx] = V_cols[col_idx][row_idx]
                W[row_idx][col_idx] = W_cols[col_idx][row_idx]
        
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
