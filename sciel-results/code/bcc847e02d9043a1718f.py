import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Partition 5 into 4, 1
    parts = [4, 1]
    offsets = [0, 4]

    U_cols = []
    V_cols = []
    W_cols = []

    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Helper to add Strassen factors for a 2x2 block
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

    # Helper to add Recursive Strassen factors for a 4x4 block
    def add_strassen_4x4(off_i, off_j, off_k):
        # Split 4x4 into four 2x2 blocks
        # A00 at (0,0), A01 at (0,2), A10 at (2,0), A11 at (2,2)
        # Same for B and C.
        
        # We need to implement the 7 Strassen products for the 4x4 case.
        # Each product is a 2x2 matrix multiplication.
        # We will generate the factors for each of the 7 products.
        
        # Product 1: M1 = (A00+A11)(B00+B11)
        # This is not a single 2x2 multiplication. It is a sum of 4.
        # But we can't just add 4 Strassen calls.
        # We must expand the linear combinations.
        
        # Actually, the rank 49 algorithm for 4x4 is NOT simply 7 calls to rank-7 2x2.
        # It is a specific set of 49 factors.
        # I will fall back to Schoolbook for the 4x4 block if I can't find the exact 49 factors.
        # Rank 64 for 4x4.
        # Total rank 125. No improvement.
        
        # Let's use the 3+2 partition instead.
        pass

    # Revised Plan: 3+2 Partition
    # Reset lists
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    parts = [3, 2]
    offsets = [0, 3]
    
    # Hardcode Rank 23 factors for 3x3x3?
    # This is too long to hardcode manually here without error.
    
    # Let's stick to the 2+2+1 partition but ensure it is correct.
    # The current code IS 2+2+1.
    # I will try a different partition: 1+2+2? Same as 2+2+1.
    
    # What if I use Strassen for the 2x2 blocks in a 1+1+1+1+1 partition? No.
    
    # I will return the current 2+2+1 logic but with a clean implementation.
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
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
                                gi = off_a + u
                                gj = off_b + v
                                gk = off_c + w
                                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                                u_vec[gi*5 + gj] = 1
                                v_vec[gj*5 + gk] = 1
                                w_vec[gk*5 + gi] = 1
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
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
