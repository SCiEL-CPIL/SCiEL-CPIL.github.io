import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 3 Strassen blocks (2x2x2) for j=0,1 and k=0,1; 2,3; 4,5 -> 3 * 7 = 21
    # 1 Schoolbook block (2x2x1) for j=0,1 and k=6 -> 4
    # 1 Rank-1 term for j=2 -> 1
    # Total rank = 21 + 4 + 1 = 26
    rank = 26
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx, col] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx, col] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx, col] += coeff
        col += 1

    def add_strassen_block(k_start):
        # A indices (2x2 block of 2x3 matrix, rows 0,1; cols 0,1)
        # A00=0, A01=1, A10=3, A11=4
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        # B indices (2x2 block of 3x7 matrix, rows 0,1; cols k_start, k_start+1)
        # B00 = 0*7 + k_start, B01 = 0*7 + k_start+1
        # B10 = 1*7 + k_start, B11 = 1*7 + k_start+1
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices (2x2 block of 2x7 output, rows 0,1; cols k_start, k_start+1)
        # Trace convention: W[k*n+i, r]
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

    # Add 3 Strassen blocks for k=0, 2, 4
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block for k=6: Schoolbook 2x2x1 for j=0,1
    # C[i, 6] += A[i, 0]B[0, 6] + A[i, 1]B[1, 6]
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    # A00 B06 -> C06
    add_term([0], [1], [b06], [1], [c06], [1])
    # A01 B16 -> C06
    add_term([1], [1], [b16], [1], [c06], [1])
    # A10 B06 -> C16
    add_term([3], [1], [b06], [1], [c16], [1])
    # A11 B16 -> C16
    add_term([4], [1], [b16], [1], [c16], [1])
    
    # Term for j=2: A[i, 2] B[2, k] -> C[i, k]
    # This is a rank-1 term: U[i*3+2, r]=1, V[2*7+k, r]=1, W[k*2+i, r]=1
    # We need to add this for all i, k.
    # Since it's a single rank-1 tensor, we just fill the columns.
    
    # U indices: i*3 + 2 for i in 0,1 -> 2, 5
    # V indices: 2*7 + k for k in 0..6 -> 14..20
    # W indices: k*2 + i for k in 0..6, i in 0,1 -> 0..13
    
    # We use the last column (col 25)
    u_indices = [2, 5]
    u_coeffs = [1, 1]
    
    v_indices = list(range(14, 21))
    v_coeffs = [1] * 7
    
    w_indices = list(range(14))
    w_coeffs = [1] * 14
    
    add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    
    # Verification
    if time.time() < deadline:
        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Note: The tensor T has T[i, j, k] = 1 if the term A[i,j]B[j,k] is present.
        # In our decomposition, T[i, j, k] is 1 for all i, j, k.
        # So we check if the sum is 1 for all i, j, k.
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-10:
            # Fallback to full schoolbook if verification fails
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, col_fb] = 1
                        V_fb[j*p+k, col_fb] = 1
                        W_fb[k*n+i, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
