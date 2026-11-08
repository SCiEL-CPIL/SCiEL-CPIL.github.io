import numpy as np
import time
import hashlib

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Rank calculation for 2x3 * 3x7 matrix multiplication:
    # Split into A[:,0:2] * B[0:2,:] (2x2 * 2x7) and A[:,2] * B[2,:] (2x1 * 1x7)
    # For 2x2 * 2x7: split B columns into pairs (0-1, 2-3, 4-5) and single (6)
    #   3 Strassen blocks (rank 7 each) = 21
    #   1 Schoolbook block (rank 4) = 4
    #   Subtotal: 25
    # For 2x1 * 1x7: 14 schoolbook terms
    # Total rank: 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 39
    V = [[0] * rank for _ in range(m * p)]   # 21 x 39
    W = [[0] * rank for _ in range(p * n)]   # 14 x 39
    
    col = 0
    
    def add_signed_term(u_indices, v_indices, w_indices, u_signs=None, v_signs=None, w_signs=None):
        """Add a rank-1 term with optional signs for each factor."""
        nonlocal col
        if u_signs is None:
            u_signs = [1] * len(u_indices)
        if v_signs is None:
            v_signs = [1] * len(v_indices)
        if w_signs is None:
            w_signs = [1] * len(w_indices)
        
        for idx, s in zip(u_indices, u_signs):
            U[idx][col] += s
        for idx, s in zip(v_indices, v_signs):
            V[idx][col] += s
        for idx, s in zip(w_indices, w_signs):
            W[idx][col] += s
        col += 1
    
    def add_strassen_block(k_start):
        """
        Add Strassen decomposition for 2x2 * 2x2 block.
        A indices: A00=0, A01=1, A10=3, A11=4 (global i*3+j)
        B indices: rows 0-1, cols k_start, k_start+1 (global j*7+k)
        C indices: cols k_start, k_start+1, rows 0-1 (global k*2+i)
        """
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_signed_term([a00, a11], [b00, b11], [c00, c11])
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_signed_term([a10, a11], [b00], [c10, c11],
                        w_signs=[1, -1])
        
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        add_signed_term([a00], [b01, b11], [c01, c11],
                        v_signs=[1, -1])
        
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        add_signed_term([a11], [b10, b00], [c00, c10],
                        v_signs=[1, -1])
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        add_signed_term([a00, a01], [b11], [c00, c01],
                        w_signs=[-1, 1])
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        add_signed_term([a10, a00], [b00, b01], [c11],
                        u_signs=[1, -1])
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        add_signed_term([a01, a11], [b10, b11], [c00],
                        u_signs=[1, -1])
    
    # 3 Strassen blocks for column pairs (0,1), (2,3), (4,5)
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Schoolbook block for column 6 (2x2 * 2x1)
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_signed_term([0], [b06], [c06])
    add_signed_term([1], [b16], [c06])
    add_signed_term([3], [b06], [c16])
    add_signed_term([4], [b16], [c16])
    
    # 2x1 * 1x7 part: A[:,2] * B[2,:]
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_signed_term([2], [b2k], [c0k])
        add_signed_term([5], [b2k], [c1k])
    
    # Quick verification using numpy for speed
    if time.time() < deadline - 2:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Verify a subset of entries for correctness
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.sum(U_arr[a_idx, :] * V_arr[b_idx, :] * W_arr[c_idx, :])
                    if abs(val - 1.0) > 1e-10:
                        errors += 1
        
        if errors > 0:
            raise ValueError(f"Verification failed with {errors} errors")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
