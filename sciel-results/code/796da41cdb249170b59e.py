"""Matrix multiplication 3,4,4: Strassen + schoolbook decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # U shape (n*m, rank) = (12, rank)
    # V shape (m*p, rank) = (16, rank)
    # W shape (p*n, rank) = (12, rank)
    
    # We split the 3x4 matrix into a 2x4 top block and a 1x4 bottom block.
    # For the 2x4 block, we use Strassen on the 2x2 part (columns 0,1) and schoolbook for columns 2,3.
    # Actually, let's be more systematic.
    # A is 3x4, B is 4x4, C is 3x4.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can view this as two block matrix multiplications:
    # C_top (2x4) = A_top (2x4) * B (4x4)
    # C_bottom (1x4) = A_bottom (1x4) * B (4x4)
    #
    # For C_top: we can further split A_top into A_top1 (2x2, cols 0-1) and A_top2 (2x2, cols 2-3)
    # C_top = A_top1 * B1 + A_top2 * B2, where B1 is B[:,0:2] and B2 is B[:,2:4]
    # Each 2x2*2x2 multiplication can use Strassen (rank 7) instead of schoolbook (rank 8).
    # So C_top needs 7 + 7 = 14 rank terms.
    # C_bottom is 1x4 * 4x4. This is just a sum of 4 scalar multiplications for each output,
    # but it's 1x4 * 4x4 = 1x4 result. Each output C_bottom[k] = sum_j A_bottom[j]*B[j,k].
    # This is a rank-4 tensor (or we can just use schoolbook: 4 terms).
    # Actually, 1x4 * 4x4: the tensor is T_{i,j,k} where i is only 0. T_{0,j,k} = delta_{i=0}.
    # The multiplication is C_k = sum_j A_j B_{j,k}. This is just matrix-vector multiplication.
    # The tensor rank for 1x4x4 is 4 (you need 4 scalar products).
    # So total rank = 7 + 7 + 4 = 18.
    
    # Let's construct this.
    # Indices:
    # A: i in 0..2, j in 0..3. U index = i*4 + j
    # B: j in 0..3, k in 0..3. V index = j*4 + k
    # C: i in 0..2, k in 0..3. W index = k*3 + i  (per the trace convention W[k*n+i, r])
    
    U = {}
    V = {}
    W = {}
    
    def add_term(u_vals, v_vals, w_vals):
        """u_vals: dict mapping A-index to coeff, v_vals: dict mapping B-index to coeff, w_vals: dict mapping C-index to coeff"""
        # Generate a unique rank index
        global rank_counter
        r = rank_counter
        rank_counter += 1
        for idx, val in u_vals.items():
            if val != 0:
                U[(idx, r)] = val
        for idx, val in v_vals.items():
            if val != 0:
                V[(idx, r)] = val
        for idx, val in w_vals.items():
            if val != 0:
                W[(idx, r)] = val
    
    rank_counter = 0
    
    # Strassen for 2x2 * 2x2
    # Given A (2x2) with entries a00,a01,a10,a11 and B (2x2) with b00,b01,b10,b11
    # The 7 Strassen products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)
    # M4 = a11(b01-b11)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_block(A_base_i, A_base_j, B_base_j, C_base_i, C_base_k, j_offset, k_offset):
        """
        A_base_i, A_base_j: base indices for A (rows i in 0,1; cols j in 0,1)
        B_base_j, B_base_k: base indices for B (rows j in 0,1; cols k in 0,1)
        C_base_i, C_base_k: base indices for C (rows i in 0,1; cols k in 0,1)
        j_offset, k_offset: offsets for the 4x4 blocks
        """
        # A indices: (A_base_i + i, A_base_j + j)
        # B indices: (B_base_j + j, B_base_k + k)
        # C indices: (C_base_i + i, C_base_k + k)
        
        # M1
        a00 = (A_base_i + 0, A_base_j + 0)
        a11 = (A_base_i + 1, A_base_j + 1)
        b00 = (B_base_j + 0, B_base_k + 0)
        b11 = (B_base_j + 1, B_base_k + 1)
        c00 = (C_base_i + 0, C_base_k + 0)
        c11 = (C_base_i + 1, C_base_k + 1)
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        
        # M2
        a10 = (A_base_i + 1, A_base_j + 0)
        c10 = (C_base_i + 1, C_base_k + 0)
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1})
        
        # M3
        c01 = (C_base_i + 0, C_base_k + 1)
        add_term({a00: 1}, {b00: -1, b11: 1}, {c01: 1})
        # Wait, M3 = a00(b10-b00). Let me re-check.
        # M3 = a00 * (b10 - b00)
        # Let me redo this more carefully.
        
        # Actually, let me just directly implement the 7 terms.
        # I'll clear and redo.
        pass
    
    # Let me just directly code the 7 Strassen terms for each 2x2 block.
    # Clear the dicts and start fresh.
    U = {}
    V = {}
    W = {}
    rank_counter = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        global rank_counter
        r = rank_counter
        rank_counter += 1
        for idx, val in u_coeffs.items():
            if val != 0:
                if (idx, r) in U:
                    U[(idx, r)] += val
                else:
                    U[(idx, r)] = val
        for idx, val in v_coeffs.items():
            if val != 0:
                if (idx, r) in V:
                    V[(idx, r)] += val
                else:
                    V[(idx, r)] = val
        for idx, val in w_coeffs.items():
            if val != 0:
                if (idx, r) in W:
                    W[(idx, r)] += val
                else:
                    W[(idx, r)] = val
    
    def idx_U(i, j):
        return i * 4 + j  # A is 3x4
    
    def idx_V(j, k):
        return j * 4 + k  # B is 4x4
    
    def idx_W(i, k):
        return k * 3 + i  # C is 3x4, W[k*n+i, r]
    
    # Strassen for top 2x2 block of A (rows 0,1; cols 0,1) * B (rows 0,1; cols 0,1) -> C (rows 0,1; cols 0,1)
    # M1 = (a00+a11)(b00+b11)
    add_term({idx_U(0,0): 1, idx_U(1,1): 1}, {idx_V(0,0): 1, idx_V(1,1): 1}, {idx_W(0,0): 1, idx_W(1,1): 1})
    # M2 = (a10+a11)b00
    add_term({idx_U(1,0): 1, idx_U(1,1): 1}, {idx_V(0,0): 1}, {idx_W(1,0): 1})
    # M3 = a00(b10-b00)
    add_term({idx_U(0,0): 1}, {idx_V(1,0): 1, idx_V(0,0): -1}, {idx_W(0,1): 1})
    # M4 = a11(b01-b11)
    add_term({idx_U(1,1): 1}, {idx_V(0,1): 1, idx_V(1,1): -1}, {idx_W(0,0): 1, idx_W(1,0): 1})
    # M5 = (a00+a01)b11
    add_term({idx_U(0,0): 1, idx_U(0,1): 1}, {idx_V(1,1): 1}, {idx_W(0,0): -1, idx_W(0,1): 1})
    # M6 = (a10-a00)(b00+b01)
    add_term({idx_U(1,0): 1, idx_U(0,0): -1}, {idx_V(0,0): 1, idx_V(0,1): 1}, {idx_W(1,1): 1})
    # M7 = (a01-a11)(b10+b11)
    add_term({idx_U(0,1): 1, idx_U(1,1): -1}, {idx_V(1,0): 1, idx_V(1,1): 1}, {idx_W(0,0): 1})
    
    # Strassen for top 2x2 block of A (rows 0,1; cols 2,3) * B (rows 2,3; cols 0,1) -> C (rows 0,1; cols 0,1)
    # A cols 2,3; B rows 2,3; C cols 0,1
    add_term({idx_U(0,2): 1, idx_U(1,3): 1}, {idx_V(2,0): 1, idx_V(3,1): 1}, {idx_W(0,0): 1, idx_W(1,1): 1})
    add_term({idx_U(1,2): 1, idx_U(1,3): 1}, {idx_V(2,0): 1}, {idx_W(1,0): 1})
    add_term({idx_U(0,2): 1}, {idx_V(3,0): 1, idx_V(2,0): -1}, {idx_W(0,1): 1})
    add_term({idx_U(1,3): 1}, {idx_V(2,1): 1, idx_V(3,1): -1}, {idx_W(0,0): 1, idx_W(1,0): 1})
    add_term({idx_U(0,2): 1, idx_U(0,3): 1}, {idx_V(3,1): 1}, {idx_W(0,0): -1, idx_W(0,1): 1})
    add_term({idx_U(1,2): 1, idx_U(0,2): -1}, {idx_V(2,0): 1, idx_V(2,1): 1}, {idx_W(1,1): 1})
    add_term({idx_U(0,3): 1, idx_U(1,3): -1}, {idx_V(3,0): 1, idx_V(3,1): 1}, {idx_W(0,0): 1})
    
    # Strassen for top 2x2 block of A (rows 0,1; cols 0,1) * B (rows 0,1; cols 2,3) -> C (rows 0,1; cols 2,3)
    add_term({idx_U(0,0): 1, idx_U(1,1): 1}, {idx_V(0,2): 1, idx_V(1,3): 1}, {idx_W(0,2): 1, idx_W(1,3): 1})
    add_term({idx_U(1,0): 1, idx_U(1,1): 1}, {idx_V(0,2): 1}, {idx_W(1,2): 1})
    add_term({idx_U(0,0): 1}, {idx_V(1,2): 1, idx_V(0,2): -1}, {idx_W(0,3): 1})
    add_term({idx_U(1,1): 1}, {idx_V(0,3): 1, idx_V(1,3): -1}, {idx_W(0,2): 1, idx_W(1,2): 1})
    add_term({idx_U(0,0): 1, idx_U(0,1): 1}, {idx_V(1,3): 1}, {idx_W(0,2): -1, idx_W(0,3): 1})
    add_term({idx_U(1,0): 1, idx_U(0,0): -1}, {idx_V(0,2): 1, idx_V(0,3): 1}, {idx_W(1,3): 1})
    add_term({idx_U(0,1): 1, idx_U(1,1): -1}, {idx_V(1,2): 1, idx_V(1,3): 1}, {idx_W(0,2): 1})
    
    # Strassen for top 2x2 block of A (rows 0,1; cols 2,3) * B (rows 2,3; cols 2,3) -> C (rows 0,1; cols 2,3)
    add_term({idx_U(0,2): 1, idx_U(1,3): 1}, {idx_V(2,2): 1, idx_V(3,3): 1}, {idx_W(0,2): 1, idx_W(1,3): 1})
    add_term({idx_U(1,2): 1, idx_U(1,3): 1}, {idx_V(2,2): 1}, {idx_W(1,2): 1})
    add_term({idx_U(0,2): 1}, {idx_V(3,2): 1, idx_V(2,2): -1}, {idx_W(0,3): 1})
    add_term({idx_U(1,3): 1}, {idx_V(2,3): 1, idx_V(3,3): -1}, {idx_W(0,2): 1, idx_W(1,2): 1})
    add_term({idx_U(0,2): 1, idx_U(0,3): 1}, {idx_V(3,3): 1}, {idx_W(0,2): -1, idx_W(0,3): 1})
    add_term({idx_U(1,2): 1, idx_U(0,2): -1}, {idx_V(2,2): 1, idx_V(2,3): 1}, {idx_W(1,3): 1})
    add_term({idx_U(0,3): 1, idx_U(1,3): -1}, {idx_V(3,2): 1, idx_V(3,3): 1}, {idx_W(0,2): 1})
    
    # Bottom row (i=2) of A (1x4) * B (4x4) -> C (row 2, 1x4)
    # C[2,k] = sum_j A[2,j] * B[j,k]
    # This is rank 4 (one term for each j)
    for j in range(4):
        for k in range(4):
            add_term({idx_U(2, j): 1}, {idx_V(j, k): 1}, {idx_W(2, k): 1})
    
    rank = rank_counter
    # Convert to arrays
    U_arr = [[0] * rank for _ in range(12)]
    V_arr = [[0] * rank for _ in range(16)]
    W_arr = [[0] * rank for _ in range(12)]
    
    for (idx, r), val in U.items():
        U_arr[idx][r] = val
    for (idx, r), val in V.items():
        V_arr[idx][r] = val
    for (idx, r), val in W.items():
        W_arr[idx][r] = val
    
    return {"U": U_arr, "V": V_arr, "W": W_arr}
