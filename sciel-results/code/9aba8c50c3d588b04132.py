# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 145.0

    n, m, p = DIMENSIONS

    # Attempt 3x3 Winograd (Rank 76)
    # Verify if it works, if not, fallback to 2x2 Strassen (Rank 189)
    
    def build_3x3_winograd():
        # Returns U_3, V_3, W_3 of shape (9, 19)
        U_3 = [[0]*19 for _ in range(9)]
        V_3 = [[0]*19 for _ in range(9)]
        W_3 = [[0]*19 for _ in range(9)]
        
        # Helper to set coeff
        def set_u(i, j, r, val): U_3[i*3+j][r] += val
        def set_v(i, j, r, val): V_3[i*3+j][r] += val
        def set_w(i, k, r, val): W_3[k*3+i][r] += val # Trace convention W[k*n+i]
        
        # 19 Products
        # 0
        set_u(0,0,0,1); set_u(1,1,0,1); set_u(2,2,0,1)
        set_v(0,0,0,1); set_v(1,1,0,1); set_v(2,2,0,1)
        set_w(0,0,0,1); set_w(1,1,0,1); set_w(2,2,0,1)
        
        # 1
        set_u(0,1,1,1); set_u(1,1,1,1); set_u(2,1,1,1)
        set_v(0,0,1,1); set_v(1,0,1,1); set_v(2,0,1,1)
        set_w(0,1,1,1); set_w(1,1,1,1); set_w(2,1,1,1)
        
        # 2
        set_u(0,2,2,1); set_u(1,2,2,1); set_u(2,2,2,1)
        set_v(0,0,2,1); set_v(1,0,2,1); set_v(2,0,2,1)
        set_w(0,2,2,1); set_w(1,2,2,1); set_w(2,2,2,1)
        
        # 3
        set_u(0,0,3,1); set_u(1,1,3,1); set_u(2,2,3,-1)
        set_v(0,0,3,1); set_v(1,1,3,1); set_v(2,2,3,-1)
        set_w(0,0,3,1); set_w(1,1,3,1); set_w(2,2,3,-1)
        
        # 4
        set_u(0,1,4,1); set_u(1,1,4,1); set_u(2,1,4,-1)
        set_v(0,0,4,1); set_v(1,0,4,1); set_v(2,0,4,-1)
        set_w(0,1,4,1); set_w(1,1,4,1); set_w(2,1,4,-1)
        
        # 5
        set_u(0,2,5,1); set_u(1,2,5,1); set_u(2,2,5,-1)
        set_v(0,0,5,1); set_v(1,0,5,1); set_v(2,0,5,-1)
        set_w(0,2,5,1); set_w(1,2,5,1); set_w(2,2,5,-1)
        
        # 6
        set_u(0,0,6,1); set_u(1,1,6,-1); set_u(2,2,6,1)
        set_v(0,0,6,1); set_v(1,1,6,-1); set_v(2,2,6,1)
        set_w(0,0,6,1); set_w(1,1,6,-1); set_w(2,2,6,1)
        
        # 7
        set_u(0,1,7,1); set_u(1,1,7,-1); set_u(2,1,7,1)
        set_v(0,0,7,1); set_v(1,0,7,-1); set_v(2,0,7,1)
        set_w(0,1,7,1); set_w(1,1,7,-1); set_w(2,1,7,1)
        
        # 8
        set_u(0,2,8,1); set_u(1,2,8,-1); set_u(2,2,8,1)
        set_v(0,0,8,1); set_v(1,0,8,-1); set_v(2,0,8,1)
        set_w(0,2,8,1); set_w(1,2,8,-1); set_w(2,2,8,1)
        
        # 9-18: Correction terms
        # 9: (A00-A20)*B20 -> C00
        set_u(0,0,9,1); set_u(2,0,9,-1)
        set_v(2,0,9,1)
        set_w(0,0,9,1)
        
        # 10: (A10-A20)*B20 -> C10
        set_u(1,0,10,1); set_u(2,0,10,-1)
        set_v(2,0,10,1)
        set_w(1,0,10,1)
        
        # 11: (A01-A21)*B21 -> C01
        set_u(0,1,11,1); set_u(2,1,11,-1)
        set_v(2,1,11,1)
        set_w(0,1,11,1)
        
        # 12: (A11-A21)*B21 -> C11
        set_u(1,1,12,1); set_u(2,1,12,-1)
        set_v(2,1,12,1)
        set_w(1,1,12,1)
        
        # 13: (A02-A22)*B22 -> C02
        set_u(0,2,13,1); set_u(2,2,13,-1)
        set_v(2,2,13,1)
        set_w(0,2,13,1)
        
        # 14: (A12-A22)*B22 -> C12
        set_u(1,2,14,1); set_u(2,2,14,-1)
        set_v(2,2,14,1)
        set_w(1,2,14,1)
        
        # 15: (A00-A02)*B02 -> C00
        set_u(0,0,15,1); set_u(0,2,15,-1)
        set_v(0,2,15,1)
        set_w(0,0,15,1)
        
        # 16: (A10-A12)*B02 -> C10
        set_u(1,0,16,1); set_u(1,2,16,-1)
        set_v(0,2,16,1)
        set_w(1,0,16,1)
        
        # 17: (A20-A22)*B02 -> C20
        set_u(2,0,17,1); set_u(2,2,17,-1)
        set_v(0,2,17,1)
        set_w(2,0,17,1)
        
        # 18: (A21-A22)*B12? No, need to cover C21, C22
        # Let's check coverage.
        # C21 needs A20B01 + A21B11 + A22B21.
        # We have terms with B21 (11,12) covering C01, C11.
        # We need C21.
        # Maybe 18 is (A21-A22)*B12?
        # This set might be incomplete or wrong.
        
        return U_3, V_3, W_3

    U_3, V_3, W_3 = build_3x3_winograd()
    
    # Verify 3x3
    import numpy as np
    valid = True
    for _ in range(5):
        A = np.random.randint(-5, 5, (3,3))
        B = np.random.randint(-5, 5, (3,3))
        C_direct = A @ B
        C_factor = np.zeros((3,3))
        for r in range(19):
            z = np.sum(np.array(U_3[:,r]) * A) * np.sum(np.array(V_3[:,r]) * B)
            C_factor += np.array(W_3[:,r]).reshape(3,3) * z
            # Note: W_3 is stored as W[k*3+i], so reshape is tricky
            # W_3 index: k*3+i. 
            # We need C[i,k].
            # My set_w used W[k*3+i].
            # So W_3[k*3+i] is coeff for C[i,k].
            # So C[i,k] += W_3[k*3+i] * z.
            # My loop above `C_factor += ...` is wrong.
            
    # Correct Verification:
    valid = True
    for _ in range(10):
        A = np.random.randint(-5, 5, (3,3))
        B = np.random.randint(-5, 5, (3,3))
        C_direct = A @ B
        C_factor = np.zeros((3,3))
        for r in range(19):
            u_vec = np.array(U_3[:,r])
            v_vec = np.array(V_3[:,r])
            w_vec = np.array(W_3[:,r])
            z = np.sum(u_vec * A) * np.sum(v_vec * B)
            # w_vec index k*3+i corresponds to C[i,k]
            # So C[i,k] += w_vec[k*3+i] * z
            for i in range(3):
                for k in range(3):
                    C_factor[i,k] += w_vec[k*3+i] * z
        if not np.allclose(C_direct, C_factor):
            valid = False
            break

    if valid:
        # Build 6x6 from 3x3
        rank = 76
        U = [[0]*rank for _ in range(36)]
        V = [[0]*rank for _ in range(36)]
        W = [[0]*rank for _ in range(36)]
        
        for bi in range(2):
            for bj in range(2):
                for bk in range(2):
                    for r in range(19):
                        col = (bi*2 + bj)*19 + (bk*2 + 0)*19 + r # Wrong indexing
                        # We need 4 blocks.
                        # Block (bi, bj, bk) ?
                        # A is 6x6. Blocks: A[0:3, 0:3], A[0:3, 3:6], etc.
                        # Multiplication C = AB.
                        # C[bi*3:(bi+1)*3, bk*3:(bk+1)*3] = sum_j A[bi*3:..., bj*3:...] B[bj*3:..., bk*3:...]
                        # So we need 2x2x2 = 8 blocks?
                        # No, 6/3 = 2.
                        # Bi, Bj, Bk are 0 or 1.
                        # For each (Bi, Bj, Bk), we have a 3x3 multiplication.
                        # Total 8 multiplications?
                        # Wait, 6x6 matrix multiplication.
                        # C[i,k] = sum_j A[i,j] B[j,k].
                        # If we split i,j,k into 2 blocks of size 3.
                        # There are 2x2x2 = 8 independent 3x3 multiplications?
                        # No, the sum is over j.
                        # C[Bi, Bk] = A[Bi, B0]B[B0, Bk] + A[Bi, B1]B[B1, Bk].
                        # This is not independent.
                        # So we cannot just tile 3x3.
                        # We need to handle the summation.
                        # This means the rank is NOT 4*19.
                        # It is higher.
                        
        # Since tiling 3x3 to 6x6 is not straightforward due to summation over j,
        # the rank 76 assumption is likely WRONG for simple tiling.
        # The rank of 6x6 is likely higher.
        
        # Fallback to Strassen 2x2 (Rank 189) is safer.
        pass

    # Use Strassen 2x2 (Rank 189)
    block_size = 2
    blocks_per_dim = 3
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    offsets = [i * block_size for i in range(blocks_per_dim)]
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        off_I = offsets[I]
        for J in range(blocks_per_dim):
            off_J = offsets[J]
            for K in range(blocks_per_dim):
                off_K = offsets[K]
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = off_I + pi
                        c = off_J + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = off_J + pi
                        c = off_K + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = off_K + pj
                        i = off_I + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                if time.time() > _deadline:
                    break
            if time.time() > _deadline:
                break
        if time.time() > _deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
