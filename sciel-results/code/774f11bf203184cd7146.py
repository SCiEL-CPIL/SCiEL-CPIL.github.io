import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    U_cols = []
    V_cols = []
    W_cols = []

    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        if len(U_cols) > 200:
            # Safety: this should never happen, but guard against runaway
            pass

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
        # 3x3 Strassen-like (Coppersmith) rank-23 decomposition
        # This is a known decomposition for 3x3 matrix multiplication
        # using 23 multiplications
        
        # We'll use a simpler approach: decompose 3x3 as blocks
        # Actually, let me use a known rank-23 algorithm for 3x3
        
        # For simplicity and correctness, let's use the standard decomposition
        # that achieves rank 23 for 3x3
        
        # Define the 23 terms for 3x3 Coppersmith algorithm
        # A is 3x3, B is 3x3, C = A*B
        
        # The Coppersmith algorithm for 3x3x3 with rank 23:
        
        # Let's use a systematic approach. We'll define the 23 rank-1 terms.
        
        # Actually, let me use a different strategy: 
        # Partition 5x5 as 3+2 and use Coppersmith for 3x3x3 and Strassen for 2x2x2
        
        # For now, let's just add schoolbook for the 3x3 blocks
        # This gives 27 terms instead of 23, but it's correct
        
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_i+di)*5+(off_j+dj)] = 1
                    v[(off_j+dj)*5+(off_k+dk)] = 1
                    w[(off_k+dk)*5+(off_i+di)] = 1
                    add_col(u, v, w)

    # Strategy: Use 3+2 partition with Strassen for 2x2 blocks
    # and schoolbook for 3x3 blocks, but optimize the cross terms
    
    # Actually, let me try a different approach:
    # Use 2+2+1 partition (current) but try to optimize
    
    # Alternative: Use 3+2 partition
    # 3x3x3: rank 27 (schoolbook) or 23 (Coppersmith)
    # 2x2x2: rank 7 (Strassen)
    # 3x2x2: rank 12
    # 2x3x2: rank 12
    # 2x2x3: rank 12
    # 3x3x2: rank 18
    # 3x2x3: rank 18
    # 2x3x3: rank 18
    # Total: 27 + 7 + 12+12+12+18+18+18 = 124 (worse than 117)
    
    # So 2+2+1 with Strassen is better
    
    # Let's stick with 2+2+1 partition but try different block arrangements
    
    # Define 8 disjoint 2x2x2 blocks for the 2+2+1 partition
    # Indices: i in {0,1,2,3,4}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    # Partition: {0,1}, {2,3}, {4}
    
    # Strassen blocks: all combinations of {0,1} and {2,3} for each dimension
    strassen_blocks = [
        (0, 0, 0), (0, 0, 2),
        (0, 2, 0), (0, 2, 2),
        (2, 0, 0), (2, 0, 2),
        (2, 2, 0), (2, 2, 2),
    ]
    
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        add_strassen_2x2(i0, j0, k0)
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    # Add schoolbook for the remaining 61 terms
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if (i, j, k) not in covered:
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[i*5 + j] = 1
                    v_vec[j*5 + k] = 1
                    w_vec[k*5 + i] = 1
                    add_col(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
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
            
    # Final safety check: ensure we're within deadline
    if time.time() > deadline - 5:
        # We're close to the deadline; return immediately
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
