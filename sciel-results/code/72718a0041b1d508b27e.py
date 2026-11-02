import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Strassen 2x2 x 2x2 rank-7 decomposition
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # The 7 terms (u, v, w) for 2x2 x 2x2
    strassen_2x2 = [
        # M1 = (a+d)(e+h)
        ([1,0,0,1], [1,1,1,1], [1,0,0,-1]),
        # M2 = (c+d)e
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # M3 = a(f-h)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        # M4 = d(g-e)
        ([0,0,0,1], [0,0,1,-1], [0,0,1,0]),
        # M5 = (a+b)h
        ([1,1,0,0], [0,0,0,1], [0,0,-1,0]),
        # M6 = (c-a)(e+f)
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # M7 = (b-d)(g+h)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Verify Strassen 2x2
    def verify_2x2():
        A = np.random.randn(4)
        B = np.random.randn(4)
        C_expected = np.zeros(4)
        A_mat = A.reshape(2,2)
        B_mat = B.reshape(2,2)
        C_mat = A_mat @ B_mat
        C_expected = C_mat.flatten()
        
        C_computed = np.zeros(4)
        for u, v, w in strassen_2x2:
            u = np.array(u)
            v = np.array(v)
            w = np.array(w)
            C_computed += np.dot(u, A) * np.dot(v, B) * w
        
        return np.allclose(C_computed, C_expected)
    
    # Build 4x4 x 4x4 decomposition using tensor product of Strassen
    # 4x4 = 2x2 blocks, so we use 2x2 Strassen at block level
    # Each block multiplication uses Strassen again
    
    # For 4x4 x 4x4:
    # A is 4x4, viewed as 2x2 blocks of 2x2
    # B is 4x4, viewed as 2x2 blocks of 2x2
    # C is 4x4, viewed as 2x2 blocks of 2x2
    
    # The block-level Strassen gives 7 terms, each involving products of 2x2 blocks
    # Each 2x2 block product uses the 2x2 Strassen decomposition
    
    # So total rank = 7 * 7 = 49
    
    # For 4x4 x 4x8:
    # Split B into B1 (4x4) and B2 (4x4)
    # C1 = A @ B1, C2 = A @ B2
    # Each has rank 49, total rank 98
    
    # Let's build the 49 rank-1 terms for 4x4 x 4x4
    # A indices for 4x4: row-major, so (i,j) -> i*4+j
    # Block (bi, bj) of A contains elements (2*bi+i, 2*bj+j) for i,j in {0,1}
    
    # For the tensor product approach:
    # Block-level Strassen term (u_block, v_block, w_block) where each is 4-dimensional (2x2 blocks)
    # Each u_block[i,j] corresponds to a 2x2 block of A
    # The actual A vector is 16-dimensional
    
    # Let me construct the 49 terms directly
    
    def block_to_full(block_vec, block_size=2, full_size=4):
        """Convert a 4-dim block vector (2x2) to a 16-dim full vector (4x4)"""
        full = np.zeros(full_size * full_size)
        block = block_vec.reshape(2, 2)
        for bi in range(2):
            for bj in range(2):
                # Block (bi, bj) starts at row 2*bi, col 2*bj
                for i in range(2):
                    for j in range(2):
                        full[(2*bi+i)*full_size + (2*bj+j)] = block[bi, bj]
        return full
    
    # Build 49 terms for 4x4 x 4x4
    terms_4x4 = []
    for u_b, v_b, w_b in strassen_2x2:
        u_b = np.array(u_b)
        v_b = np.array(v_b)
        w_b = np.array(w_b)
        for u_b2, v_b2, w_b2 in strassen_2x2:
            u_b2 = np.array(u_b2)
            v_b2 = np.array(v_b2)
            w_b2 = np.array(w_b2)
            
            # The combined term: u_full = u_b ⊗ u_b2 (Kronecker product in block sense)
            # Actually, for the tensor product of two 2x2 Strassen decompositions:
            # The block-level term has u_b[i,j] for block (i,j)
            # Each block multiplication uses the inner Strassen term
            
            # The full u vector: for each block (bi,bj), the contribution is u_b[bi,bj] * u_b2
            # So u_full is a 16-dim vector
            
            u_full = np.zeros(16)
            v_full = np.zeros(16)
            w_full = np.zeros(16)
            
            u_b_mat = u_b.reshape(2,2)
            v_b_mat = v_b.reshape(2,2)
            w_b_mat = w_b.reshape(2,2)
            
            for bi in range(2):
                for bj in range(2):
                    coeff_u = u_b_mat[bi, bj]
                    coeff_v = v_b_mat[bi, bj]
                    coeff_w = w_b_mat[bi, bj]
                    
                    # The inner Strassen term u_b2 is 4-dim, corresponding to a 2x2 block
                    # Place it in the appropriate block of the 4x4 matrix
                    for i in range(2):
                        for j in range(2):
                            row = 2*bi + i
                            col = 2*bj + j
                            u_full[row*4+col] = coeff_u * u_b2[i*2+j]
                            v_full[row*4+col] = coeff_v * v_b2[i*2+j]
                            w_full[row*4+col] = coeff_w * w_b2[i*2+j]
            
            terms_4x4.append((u_full, v_full, w_full))
    
    # Verify 4x4 x 4x4 decomposition
    def verify_4x4():
        A = np.random.randn(16)
        B = np.random.randn(16)
        A_mat = A.reshape(4,4)
        B_mat = B.reshape(4,4)
        C_expected = (A_mat @ B_mat).flatten()
        
        C_computed = np.zeros(16)
        for u, v, w in terms_4x4:
            C_computed += np.dot(u, A) * np.dot(v, B) * w
        
        return np.allclose(C_computed, C_expected)
    
    # Check time
    if time.time() > deadline:
        # Return trivial decomposition
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        r = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify
    if not verify_4x4():
        # Fallback to trivial
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        r = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 4x4 x 4x8 decomposition
    # Split B into two 4x4 parts: B[:, 0:4] and B[:, 4:8]
    # C[:, 0:4] = A @ B[:, 0:4]
    # C[:, 4:8] = A @ B[:, 4:8]
    
    # For C[:, 0:4]: use the 49 terms, with V vectors only in the first half
    # For C[:, 4:8]: use the 49 terms, with V vectors only in the second half
    
    # A is 4x4 (16-dim), B is 4x8 (32-dim), C is 4x8 (32-dim)
    # B index: (j, k) where j in [0,4), k in [0,8)
    # B[:, 0:4] has indices (j, k) for j in [0,4), k in [0,4) -> 16 elements
    # B[:, 4:8] has indices (j, k) for j in [0,4), k in [4,8) -> 16 elements
    
    # C index: (i, k) where i in [0,4), k in [0,8)
    # C[:, 0:4] has indices (i, k) for i in [0,4), k in [0,4) -> 16 elements
    # C[:, 4:8] has indices (i, k) for i in [0,4), k in [4,8) -> 16 elements
    
    R = 98
    U = np.zeros((16, R))
    V = np.zeros((32, R))
    W = np.zeros((32, R))
    
    r = 0
    # First 49 terms: C[:, 0:4] = A @ B[:, 0:4]
    for u, v, w in terms_4x4:
        # u is 16-dim (A)
        # v is 16-dim (B[:, 0:4]) -> place in first 16 of 32-dim V
        # w is 16-dim (C[:, 0:4]) -> place in first 16 of 32-dim W
        
        U[:, r] = u
        V[:16, r] = v  # B[:, 0:4]
        W[:16, r] = w  # C[:, 0:4]
        r += 1
    
    # Next 49 terms: C[:, 4:8] = A @ B[:, 4:8]
    for u, v, w in terms_4x4:
        U[:, r] = u
        V[16:, r] = v  # B[:, 4:8]
        W[16:, r] = w  # C[:, 4:8]
        r += 1
    
    # Verify
    def verify_full():
        A = np.random.randn(16)
        B = np.random.randn(32)
        A_mat = A.reshape(4,4)
        B_mat = B.reshape(4,8)
        C_expected = (A_mat @ B_mat).flatten()
        
        C_computed = np.zeros(32)
        for r_idx in range(R):
            C_computed += np.dot(U[:, r_idx], A) * np.dot(V[:, r_idx], B) * W[:, r_idx]
        
        return np.allclose(C_computed, C_expected)
    
    if not verify_full():
        # Fallback to trivial
        R_trivial = n * m * p
        U_t = np.zeros((n * m, R_trivial))
        V_t = np.zeros((m * p, R_trivial))
        W_t = np.zeros((p * n, R_trivial))
        r_t = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    U_t[i * m + j, r_t] = 1
                    V_t[j * p + k, r_t] = 1
                    W_t[k * n + i, r_t] = 1
                    r_t += 1
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
