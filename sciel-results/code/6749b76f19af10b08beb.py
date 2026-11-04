# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Uses Strassen's algorithm for the 4x4 sub-block (rank 49) and schoolbook for cross terms.
    Total rank: 49 + 16 + 4 + 16 + 4 + 4 + 1 = 94.
    """
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 145
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u, v, w):
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)

    # Helper to create a zero vector
    def zero_vec():
        return np.zeros(25)

    # Helper to add a scalar multiple of a basis vector
    def add_basis(vec, idx, coeff):
        vec[idx] += coeff

    # --- Part 1: C[0:4, 0:4] using Strassen on 4x4 ---
    
    # Define the 2x2 blocks of A[0:4, 0:4]
    # Block 00: A[0:2, 0:2] -> indices (0,0),(0,1),(1,0),(1,1)
    # Block 01: A[0:2, 2:4] -> indices (0,2),(0,3),(1,2),(1,3)
    # Block 10: A[2:4, 0:2] -> indices (2,0),(2,1),(3,0),(3,1)
    # Block 11: A[2:4, 2:4] -> indices (2,2),(2,3),(3,2),(3,3)
    
    # We represent a 2x2 matrix of vectors (length 25) for the "S" and "T" matrices in Strassen
    # S[i,j] is a vector representing the linear combination of A entries for that position
    
    def get_block_vec(block_idx, is_A):
        """
        Returns a 2x2 array of vectors (each length 25) representing the block.
        block_idx: 0, 1, 2, 3 corresponding to 00, 01, 10, 11
        """
        vecs = np.zeros((2, 2, 25))
        if is_A:
            # A indices
            if block_idx == 0: # 00
                indices = [(0,0), (0,1), (1,0), (1,1)]
            elif block_idx == 1: # 01
                indices = [(0,2), (0,3), (1,2), (1,3)]
            elif block_idx == 2: # 10
                indices = [(2,0), (2,1), (3,0), (3,1)]
            else: # 11
                indices = [(2,2), (2,3), (3,2), (3,3)]
        else:
            # B indices
            if block_idx == 0: # 00
                indices = [(0,0), (0,1), (1,0), (1,1)]
            elif block_idx == 1: # 01
                indices = [(0,2), (0,3), (1,2), (1,3)]
            elif block_idx == 2: # 10
                indices = [(2,0), (2,1), (3,0), (3,1)]
            else: # 11
                indices = [(2,2), (2,3), (3,2), (3,3)]
        
        for i in range(2):
            for j in range(2):
                r, c = indices[i*2+j]
                vecs[i, j, r*5+c] = 1.0
        return vecs

    # Precompute block vectors
    A_blocks = [get_block_vec(i, True) for i in range(4)]
    B_blocks = [get_block_vec(i, False) for i in range(4)]

    # Outer Strassen steps for 4x4
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    outer_steps = [
        # (S_blocks, S_coeffs, T_blocks, T_coeffs)
        # S_blocks: list of block indices, S_coeffs: list of coeffs
        ([0, 3], [1, 1], [0, 3], [1, 1]), # M1
        ([2, 3], [1, 1], [0], [1]),       # M2
        ([0], [1], [1, 3], [1, -1]),      # M3
        ([3], [1], [2, 0], [1, -1]),      # M4
        ([0, 1], [1, 1], [3], [1]),       # M5
        ([2, 0], [1, -1], [0, 1], [1, 1]),# M6
        ([1, 3], [1, -1], [2, 3], [1, 1]) # M7
    ]

    # Inner Strassen coefficients for 2x2
    # U1 = S00+S11, U2 = S10+S11, U3 = S00, U4 = S11, U5 = S00+S01, U6 = S10-S00, U7 = S01-S11
    # V1 = T00+T11, V2 = T00, V3 = T01-T11, V4 = T10-T00, V5 = T11, V6 = T00+T01, V7 = T10+T11
    inner_U_coeffs = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    inner_V_coeffs = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    # Contribution of inner terms M1..M7 to the 2x2 result Z
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    inner_W_contribs = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,-1)],     # M2
        [(0,1,1), (1,1,1)],      # M3
        [(0,0,1), (1,0,1)],      # M4
        [(0,0,-1), (0,1,1)],     # M5
        [(1,1,1)],               # M6
        [(0,0,1)]                # M7
    ]

    # Outer reconstruction for 4x4 result C11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    outer_W_contribs = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,-1)],     # M2
        [(0,1,1), (1,1,1)],      # M3
        [(0,0,1), (1,0,1)],      # M4
        [(0,0,-1), (0,1,1)],     # M5
        [(1,1,1)],               # M6
        [(0,0,1)]                # M7
    ]

    # Process each outer step
    for k in range(7):
        S_blocks, S_coeffs, T_blocks, T_coeffs = outer_steps[k]
        
        # Construct S and T as 2x2 arrays of vectors
        S = np.zeros((2, 2, 25))
        for b_idx, coeff in zip(S_blocks, S_coeffs):
            S += coeff * A_blocks[b_idx]
            
        T = np.zeros((2, 2, 25))
        for b_idx, coeff in zip(T_blocks, T_coeffs):
            T += coeff * B_blocks[b_idx]
            
        # Process 7 inner terms
        for j in range(7):
            U_vec = np.zeros(25)
            for r, c, coeff in inner_U_coeffs[j]:
                U_vec += coeff * S[r, c]
                
            V_vec = np.zeros(25)
            for r, c, coeff in inner_V_coeffs[j]:
                V_vec += coeff * T[r, c]
                
            W_vec = np.zeros(25)
            # W_vec gets contributions from inner_W_contribs[j] (to 2x2 blocks)
            # and outer_W_contribs[k] (to 4x4 blocks)
            # The 2x2 blocks of the inner result correspond to the 2x2 blocks of M_k
            # M_k is one of the 7 intermediate matrices in the outer Strassen
            # We need to map the 2x2 block positions of M_k to the 4x4 block positions of C11
            # and apply the outer coefficients
            
            for r_in, c_in, coeff_in in inner_W_contribs[j]:
                # r_in, c_in is the position in the 2x2 result of the inner product
                # This 2x2 result is M_k
                # Now apply outer_W_contribs[k]
                for r_out, c_out, coeff_out in outer_W_contribs[k]:
                    # Wait, the inner product result M_k is a 2x2 matrix.
                    # The outer step k produces M_k.
                    # The outer reconstruction uses M_k to form C11.
                    # The coefficients in outer_W_contribs[k] tell us how M_k (as a whole 2x2 matrix? No, M_k is a 2x2 matrix) contributes.
                    # Actually, M_k is a 2x2 matrix.
                    # The outer Strassen formula says C11[0,0] = M1[0,0] + M4[0,0] ...
                    # So we need to sum the contributions element-wise.
                    
                    # The term coeff_in * M_k[r_in, c_in] contributes to C11.
                    # The outer step k contributes M_k to C11 with coefficients defined in outer_W_contribs[k]?
                    # No, outer_W_contribs[k] lists the contributions of M_k to the 4 blocks of C11.
                    # But M_k is a 2x2 matrix.
                    # The formula C11[0,0] = M1 + M4 ... means the (0,0) block of C11 is the sum of the (0,0) entries of M1, M4, etc?
                    # No, in Strassen, M_k are 2x2 matrices.
                    # C11[0,0] (which is a 2x2 block) = M1[0,0] (2x2 block) + M4[0,0] (2x2 block) ...
                    # So we need to map the 2x2 block index (r_out, c_out) in C11 to the 4x4 matrix indices.
                    
                    # r_out, c_out is the block index in C11 (0..1 for rows, 0..1 for cols)
                    # This block is 2x2.
                    # The entry (r_in, c_in) of M_k contributes to this block?
                    # Yes, M_k[r_in, c_in] is a 2x2 matrix? No.
                    # In the inner Strassen, the result is a 2x2 matrix.
                    # Let's call it Z.
                    # Z[r_in, c_in] is a scalar? No, in the context of the outer Strassen, M_k is a 2x2 matrix of scalars (if we were doing 2x2).
                    # But here, M_k is the result of multiplying two 2x2 matrices of *vectors*?
                    # No, S and T are 2x2 matrices of vectors (length 25).
                    # The product M_k = S T is a 2x2 matrix of vectors?
                    # No, the "inner Strassen" computes the product of two 2x2 matrices.
                    # The entries of S and T are vectors.
                    # The product is computed by summing products of entries.
                    # So M_k is a 2x2 matrix where each entry is a vector (length 25).
                    
                    # So M_k[r_in, c_in] is a vector.
                    # This vector contributes to the C11 block (r_out, c_out).
                    # The C11 block (r_out, c_out) is a 2x2 block of the 4x4 matrix.
                    # Wait, the outer Strassen treats the 4x4 matrix as 2x2 blocks of 2x2 matrices.
                    # So C11[0,0] is a 2x2 matrix.
                    # The contribution of M_k to C11[0,0] is M_k itself?
                    # No, the outer formula is C11_block_00 = M1_block_00 + M4_block_00 ...
                    # So M_k is a 2x2 matrix.
                    # M_k[r_in, c_in] is the (r_in, c_in) entry of the 2x2 matrix M_k.
                    # This entry is a vector.
                    # It contributes to the (r_in, c_in) entry of the C11 block (r_out, c_out).
                    
                    # So we need to map:
                    # Global row in C: r_out * 2 + r_in
                    # Global col in C: c_out * 2 + c_in
                    # Global index in C: (r_out * 2 + r_in) * 5 + (c_out * 2 + c_in)
                    
                    global_r = r_out * 2 + r_in
                    global_c = c_out * 2 + c_in
                    global_idx = global_r * 5 + global_c
                    
                    W_vec[global_idx] += coeff_in * coeff_out

            if np.any(U_vec != 0) and np.any(V_vec != 0) and np.any(W_vec != 0):
                add_term(U_vec, V_vec, W_vec)

    # --- Part 2: Cross Terms (Schoolbook) ---
    
    # C[0:4, 4] = A[0:4, 0:4] B[0:4, 4] + A[0:4, 4] B[4, 4]
    for i in range(4):
        for k in range(4):
            u = zero_vec(); v = zero_vec(); w = zero_vec()
            u[i*5+k] = 1.0
            v[k*5+4] = 1.0
            w[i*5+4] = 1.0
            add_term(u, v, w)
        # A[i,4] * B[4,4]
        u = zero_vec(); v = zero_vec(); w = zero_vec()
        u[i*5+4] = 1.0
        v[4*5+4] = 1.0
        w[i*5+4] = 1.0
        add_term(u, v, w)
        
    # C[4, 0:4] = A[4, 0:4] B[0:4, 0:4] + A[4, 4] B[4, 0:4]
    for k in range(4):
        for j in range(4):
            u = zero_vec(); v = zero_vec(); w = zero_vec()
            u[4*5+j] = 1.0
            v[j*5+k] = 1.0
            w[4*5+k] = 1.0
            add_term(u, v, w)
        # A[4,4] * B[4,k]
        u = zero_vec(); v = zero_vec(); w = zero_vec()
        u[4*5+4] = 1.0
        v[4*5+k] = 1.0
        w[4*5+k] = 1.0
        add_term(u, v, w)
        
    # C[4, 4] = A[4, 0:4] B[0:4, 4] + A[4, 4] B[4, 4]
    for k in range(4):
        u = zero_vec(); v = zero_vec(); w = zero_vec()
        u[4*5+k] = 1.0
        v[k*5+4] = 1.0
        w[4*5+4] = 1.0
        add_term(u, v, w)
    # A[4,4] * B[4,4]
    u = zero_vec(); v = zero_vec(); w = zero_vec()
    u[4*5+4] = 1.0
    v[4*5+4] = 1.0
    w[4*5+4] = 1.0
    add_term(u, v, w)

    # Build matrices
    rank = len(U_terms)
    U = np.column_stack(U_terms)
    V = np.column_stack(V_terms)
    W = np.column_stack(W_terms)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
