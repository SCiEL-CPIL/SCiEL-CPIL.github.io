"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Strategy: Embed 5x5x5 into 7x7x7, use Strassen-based rank-21 decomposition for 7x7,
    # then project back to 5x5 subspace.
    #
    # The 7x7 Strassen decomposition has rank 21. We embed the 5x5 multiplication tensor
    # into the 7x7 tensor by padding with zeros, then extract the relevant components.
    #
    # Key insight: The 7x7 Strassen decomposition uses 7 scalar multiplications per 2x2 block.
    # For 7x7, we can think of it as having structure that allows rank-21.
    #
    # Actually, let me use a more direct approach:
    # Use the known fact that 5x5x5 matrix multiplication can be done with rank 100 or less
    # by using a combination of Strassen and schoolbook.
    #
    # Better approach: Use the 4+1 decomposition but apply Strassen to the 4x4 part.
    # 4x4 * 4x4 = 4x4 can be done with rank 49 (Strassen recursively: 7*7).
    # Then handle the cross terms.
    
    U = {}  # (i*m+j) -> list of (col, val)
    V = {}  # (j*p+k) -> list of (col, val)
    W = {}  # (k*n+i) -> list of (col, val)
    
    next_col = 0
    
    def add_term(a_idx, b_idx, c_idx, val=1):
        """Add a rank-1 term: U[a_idx] += val, V[b_idx] += val, W[c_idx] += val"""
        nonlocal next_col
        if a_idx not in U:
            U[a_idx] = []
        if b_idx not in V:
            V[b_idx] = []
        if c_idx not in W:
            W[c_idx] = []
        U[a_idx].append((next_col, val))
        V[b_idx].append((next_col, val))
        W[c_idx].append((next_col, val))
        next_col += 1
    
    # Use Strassen for the 4x4 * 4x4 = 4x4 part.
    # Split 4x4 into four 2x2 blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B and C.
    #
    # Strassen for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    #
    # Each M_i is a 2x2 * 2x2 = 2x2 product, done with rank 7 (Strassen).
    # Total: 7 * 7 = 49 terms for the 4x4 part.
    
    # Define Strassen 2x2 decomposition terms.
    # For multiplying X (2x2) by Y (2x2) to get Z (2x2):
    # X indices: X[0,0]=x00, X[0,1]=x01, X[1,0]=x10, X[1,1]=x11
    # Y indices: Y[0,0]=y00, Y[0,1]=y01, Y[1,0]=y10, Y[1,1]=y11
    # Z indices: Z[0,0]=z00, Z[0,1]=z01, Z[1,0]=z10, Z[1,1]=z11
    #
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y01-y11)
    # m4 = x11(y10-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y01)
    # m7 = (x01-x11)(y10+y11)
    #
    # z00 = m1 - m4 + m6
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m3 + m7
    
    def strassen_2x2_decomp(x_base_i, x_base_j, y_base_i, y_base_j, z_base_i, z_base_j):
        """Generate the 7 Strassen terms for a 2x2 block multiplication.
        x_base_i, x_base_j: base row, col for X in the full A matrix
        y_base_i, y_base_j: base row, col for Y in the full B matrix  
        z_base_i, z_base_j: base row, col for Z in the full C matrix
        """
        # X elements:
        # x00 = A[x_base_i, x_base_j]
        # x01 = A[x_base_i, x_base_j+1]
        # x10 = A[x_base_i+1, x_base_j]
        # x11 = A[x_base_i+1, x_base_j+1]
        
        # Y elements:
        # y00 = B[y_base_i, y_base_j]
        # y01 = B[y_base_i, y_base_j+1]
        # y10 = B[y_base_i+1, y_base_j]
        # y11 = B[y_base_i+1, y_base_j+1]
        
        # Z elements:
        # z00 = C[z_base_i, z_base_j]
        # z01 = C[z_base_i, z_base_j+1]
        # z10 = C[z_base_i+1, z_base_j]
        # z11 = C[z_base_i+1, z_base_j+1]
        
        # Helper to add a scalar product term with coefficients
        def add_scalar_term(x_coeffs, y_coeffs, z_coeffs):
            """x_coeffs: dict {A_index: coeff}, y_coeffs: dict {B_index: coeff}, z_coeffs: dict {C_index: coeff}"""
            nonlocal next_col
            # For each combination of A, B, C indices with non-zero coefficients
            for a_idx, a_val in x_coeffs.items():
                for b_idx, b_val in y_coeffs.items():
                    for c_idx, c_val in z_coeffs.items():
                        if a_val != 0 and b_val != 0 and c_val != 0:
                            # The coefficient for this rank-1 term
                            # We need U[a_idx] to have the product of all A coefficients
                            # Actually, this approach doesn't work directly for linear combinations.
                            # Let me use a different approach.
                            pass
        
        # Better approach: directly enumerate the 7 scalar multiplications
        # m1 = (x00+x11)(y00+y11)
        # This expands to: x00*y00 + x00*y11 + x11*y00 + x11*y11
        # Each is a rank-1 term in the overall tensor.
        
        # For m1:
        # U contributions: A[x_base_i, x_base_j] * 1, A[x_base_i+1, x_base_j+1] * 1
        # V contributions: B[y_base_i, y_base_j] * 1, B[y_base_i, y_base_j+1] * 1, B[y_base_i+1, y_base_j] * 1, B[y_base_i+1, y_base_j+1] * 1
        # Wait, that's not right either.
        
        # Let me think more carefully. In a rank-1 decomposition:
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        #
        # For the term m1 = (x00+x11)(y00+y11), this contributes to z00, z01, z10, z11.
        # Specifically:
        # z00 += m1, z01 += 0 (from m1 alone, but m1 contributes to z00 and z11)
        # Actually: z00 = m1 - m4 + m6, so m1 contributes +1 to z00.
        # z11 = m1 - m3 + m7, so m1 contributes +1 to z11.
        # z01 and z10 don't get contributions from m1.
        
        # So m1 is a single scalar that gets added to z00 and z11.
        # But m1 itself is a product of two linear combinations.
        # (x00+x11)(y00+y11) = x00*y00 + x00*y11 + x11*y00 + x11*y11
        #
        # This is 4 rank-1 terms in the A*B -> C tensor, but they all contribute to the same output positions.
        #
        # Actually, I think the cleanest way is to just directly generate all the terms.
        
        # Let me define the 7 Strassen combinations explicitly:
        # Each combination is: (list of (A_idx, coeff)), (list of (B_idx, coeff)), (list of (C_idx, coeff))
        # where the term is: sum_{a,c} U[a_idx]*c * V[b_idx]*d * W[c_idx]*e for all valid a,b,c
        
        # This is getting complicated. Let me just use schoolbook for the 4x4 part.
        # 4*4*4 = 64 terms.
        
        # A[i,j] * B[j,k] for i,j,k in the 4x4 block
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    a_idx = (x_base_i + di) * m + (x_base_j + dj)
                    b_idx = (y_base_i + dj) * p + (y_base_j + dk)
                    c_idx = (z_base_i + dk) * n + (z_base_i + di)
                    add_term(a_idx, b_idx, c_idx)
    
    # Hmm, this is getting too complex. Let me just use the schoolbook 4+1 approach
    # which gives rank 113, and see if I can improve it.
    
    # Actually, let me try a different approach: use the known rank-100 decomposition
    # for 5x5x5 matrix multiplication.
    #
    # One known approach: use 3x3 Strassen (rank 23) combined with padding.
    # But 5 is not a power of 2, so this is tricky.
    #
    # Let me try: split 5 = 3 + 2.
    # A = [[A11, A12], [A21, A22]] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2
    # B = [[B11, B12], [B21, B22]] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2
    # C = [[C11, C12], [C21, C22]] where C11 is 3x3, C12 is 3x2, C21 is 2x3, C22 is 2x2
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # A11*B11: 3x3*3x3, use Strassen (rank 23) or schoolbook (rank 27)
    # A12*B21: 3x2*2x3, schoolbook: 3*2*3 = 18
    # A11*B12: 3x3*3x2, schoolbook: 3*3*2 = 18
    # A12*B22: 3x2*2x2, schoolbook: 3*2*2 = 12
    # A21*B11: 2x3*3x3, schoolbook: 2*3*3 = 18
    # A22*B21: 2x2*2x3, schoolbook: 2*2*3 = 12
    # A21*B12: 2x3*3x2, schoolbook: 2*3*2 = 12
    # A22*B22: 2x2*2x2, Strassen (rank 7) or schoolbook (rank 8)
    #
    # Total with schoolbook for 3x3: 27 + 18 + 18 + 12 + 18 + 12 + 12 + 8 = 125
    # Total with Strassen for 3x3: 23 + 18 + 18 + 12 + 18 + 12 + 12 + 8 = 121
    # Total with Strassen for 3x3 and 2x2: 23 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 120
    
    # That's not better than 113. Let me stick with the 4+1 split.
    
    # 4+1 split with schoolbook:
    # A11*B11: 4*4*4 = 64
    # A12*B21: 4*1*4 = 16
    # A11*B12: 4*4*1 = 16
    # A12*B22: 4*1*1 = 4
    # A21*B11: 1*4*4 = 16
    # A22*B21: 1*1*4 = 4
    # A21*B12: 1*4*1 = 4
    # A22*B22: 1*1*1 = 1
    # Total: 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125
    
    # Wait, that's 125, not 113. Let me recount the original.
    # Original had:
    # A11*B11: 64
    # A12*B21: 4 (not 16!) - because A12 is 4x1 and B21 is 1x4, so 4*1*4 = 16? No...
    
    # Actually A12 is A[0:4, 4] which is 4x1, B21 is B[4, 0:4] which is 1x4.
    # Product: 4x4, with 4*1*4 = 16 terms.
    
    # Hmm, but the original code only had 4 terms for A12*B21. Let me re-examine.
    
    # Original code:
    # for i in range(4):
    #     for k in range(4):
    #         a_idx = i * m + 4
    #         b_idx = 4 * p + k
    #         c_idx = k * n + i
    #         add_term(a_idx, b_idx, c_idx)
    # That's 16 terms, not 4.
    
    # Let me recount:
    # A11*B11: 4*4*4 = 64
    # A12*B21: 4*1*4 = 16
    # A11*B12: 4*4*1 = 16
    # A12*B22: 4*1*1 = 4
    # A21*B11: 1*4*4 = 16
    # A22*B21: 1*1*4 = 4
    # A21*B12: 1*4*1 = 4
    # A22*B22: 1*1*1 = 1
    # Total: 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125
    
    # So the original is actually 125, same as schoolbook. The comment said 113 but the code does 125.
    
    # Let me just implement the 4+1 split with schoolbook, which is rank 125.
    # Then try to improve.
    
    # A11*B11: A[i,j] * B[j,k] for i,j,k in 0..3 -> C[i,k]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                add_term(a_idx, b_idx, c_idx)
    
    # A12*B21: A[i,4] * B[4,k] for i,k in 0..3 -> C[i,k]
    for i in range(4):
        for k in range(4):
            a_idx = i * m + 4
            b_idx = 4 * p + k
            c_idx = k * n + i
            add_term(a_idx, b_idx, c_idx)
    
    # A11*B12: A[i,j] * B[j,4] for i,j in 0..3 -> C[i,4]
    for i in range(4):
        for j in range(4):
            a_idx = i * m + j
            b_idx = j * p + 4
            c_idx = 4 * n + i
            add_term(a_idx, b_idx, c_idx)
    
    # A12*B22: A[i,4] * B[4,4] for i in 0..3 -> C[i,4]
    for i in range(4):
        a_idx = i * m + 4
        b_idx = 4 * p + 4
        c_idx = 4 * n + i
        add_term(a_idx, b_idx, c_idx)
    
    # A21*B11: A[4,j] * B[j,k] for j,k in 0..3 -> C[4,k]
    for j in range(4):
        for k in range(4):
            a_idx = 4 * m + j
            b_idx = j * p + k
            c_idx = k * n + 4
            add_term(a_idx, b_idx, c_idx)
    
    # A22*B21: A[4,4] * B[4,k] for k in 0..3 -> C[4,k]
    for k in range(4):
        a_idx = 4 * m + 4
        b_idx = 4 * p + k
        c_idx = k * n + 4
        add_term(a_idx, b_idx, c_idx)
    
    # A21*B12: A[4,j] * B[j,4] for j in 0..3 -> C[4,4]
    for j in range(4):
        a_idx = 4 * m + j
        b_idx = j * p + 4
        c_idx = 4 * n + 4
        add_term(a_idx, b_idx, c_idx)
    
    # A22*B22: A[4,4] * B[4,4] -> C[4,4]
    add_term(4 * m + 4, 4 * p + 4, 4 * n + 4)
    
    rank = next_col
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for idx, terms in U.items():
        for col, val in terms:
            U_mat[idx][col] = val
    for idx, terms in V.items():
        for col, val in terms:
            V_mat[idx][col] = val
    for idx, terms in W.items():
        for col, val in terms:
            W_mat[idx][col] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
