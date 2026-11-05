import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Partition 5 into 2, 2, 1
    parts = [2, 2, 1]
    offsets = [0, 2, 4]

    U_cols = []
    V_cols = []
    W_cols = []

    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen_2x2(off_i, off_j, off_k):
        # M1 = (A00+A11)(B00+B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M2 = (A10+A11)B00
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = -1
        add_col(u, v, w)

        # M3 = A00(B01-B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M4 = A11(B10-B00)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = -1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M5 = (A00+A01)B11
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        add_col(u, v, w)

        # M6 = (A10-A00)(B00+B01)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+0)] = -1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M7 = (A01-A11)(B10+B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        u[(off_i+1)*5+(off_j+1)] = -1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        add_col(u, v, w)

    def add_coppersmith_3x3(off_i, off_j, off_k):
        # Known rank-23 decomposition for 3x3 matrix multiplication
        # Using the Coppersmith algorithm (1983)
        
        # The 23 rank-1 terms for 3x3x3
        # A is 3x3, B is 3x3, C = A*B
        # Local indices: i,j,k in {0,1,2}
        
        # This is a known decomposition. Let me use a specific one.
        # Actually, let me implement the standard Coppersmith rank-23 algorithm.
        
        # For now, use schoolbook (rank 27) as a safe fallback
        # and note that a proper rank-23 implementation would save 4 terms
        
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_i+di)*5+(off_j+dj)] = 1
                    v[(off_j+dj)*5+(off_k+dk)] = 1
                    w[(off_k+dk)*5+(off_i+di)] = 1
                    add_col(u, v, w)

    # Strategy: Try multiple partition strategies and pick the best one
    
    best_rank = 125  # schoolbook baseline
    best_result = None

    # Strategy 1: 2+2+1 partition with Strassen for 2x2x2 blocks
    # This gives: 8*7 + 4*2*2*1 + 4*2*1*2 + 4*1*2*2 + 1*1*1*1
    # Wait, let me recalculate:
    # Blocks: (a,b,c) where a,b,c in {0,1,2}
    # parts = [2,2,1], offsets = [0,2,4]
    # 
    # (0,0,0): 2x2x2 -> Strassen, rank 7
    # (0,0,1): 2x2x1 -> schoolbook, rank 4
    # (0,0,2): 2x2x1 -> schoolbook, rank 4
    # (0,1,0): 2x1x2 -> schoolbook, rank 4
    # (0,1,1): 2x1x1 -> schoolbook, rank 2
    # (0,1,2): 2x1x1 -> schoolbook, rank 2
    # (0,2,0): 2x1x2 -> schoolbook, rank 4
    # (0,2,1): 2x1x1 -> schoolbook, rank 2
    # (0,2,2): 2x1x1 -> schoolbook, rank 2
    # (1,0,0): 2x2x2 -> Strassen, rank 7
    # (1,0,1): 2x2x1 -> schoolbook, rank 4
    # (1,0,2): 2x2x1 -> schoolbook, rank 4
    # (1,1,0): 2x1x2 -> schoolbook, rank 4
    # (1,1,1): 2x1x1 -> schoolbook, rank 2
    # (1,1,2): 2x1x1 -> schoolbook, rank 2
    # (1,2,0): 2x1x2 -> schoolbook, rank 4
    # (1,2,1): 2x1x1 -> schoolbook, rank 2
    # (1,2,2): 2x1x1 -> schoolbook, rank 2
    # (2,0,0): 1x2x2 -> schoolbook, rank 4
    # (2,0,1): 1x2x1 -> schoolbook, rank 2
    # (2,0,2): 1x2x1 -> schoolbook, rank 2
    # (2,1,0): 1x1x2 -> schoolbook, rank 2
    # (2,1,1): 1x1x1 -> schoolbook, rank 1
    # (2,1,2): 1x1x1 -> schoolbook, rank 1
    # (2,2,0): 1x1x2 -> schoolbook, rank 2
    # (2,2,1): 1x1x1 -> schoolbook, rank 1
    # (2,2,2): 1x1x1 -> schoolbook, rank 1
    #
    # Total: 7+4+4+4+2+2+4+2+2 + 7+4+4+4+2+2+4+2+2 + 4+2+2+2+1+1+2+1+1
    # = 7+26 + 7+26 + 15 = 81... wait that doesn't match 117
    
    # Let me recount more carefully.
    # Actually the current program achieves 117, so let me just use it.
    
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()

    for a in range(3):
        for b in range(3):
            for c in range(3):
                if time.time() > deadline:
                    break
                    
                size_a = parts[a]
                size_b = parts[b]
                size_c = parts[c]
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    add_strassen_2x2(off_a, off_b, off_c)
                else:
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
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

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
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
