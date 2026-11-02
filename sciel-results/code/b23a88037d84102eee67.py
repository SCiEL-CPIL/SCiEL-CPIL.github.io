import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return U, V, W

    # Helper to create Strassen decomposition for 2x2x2
    def strassen_2x2():
        # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1 # A11, A22
        V[0, 0] = 1; V[3, 0] = 1 # B11, B22
        W[0, 0] = 1; W[3, 0] = 1 # C11, C22
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1 # A21, A22
        V[0, 1] = 1 # B11
        W[2, 1] = 1 # C21
        
        # M3
        U[0, 2] = 1 # A11
        V[2, 2] = 1; V[3, 2] = -1 # B21, -B22
        W[1, 2] = 1 # C12
        
        # M4
        U[3, 3] = 1 # A22
        V[1, 3] = 1; V[0, 3] = -1 # B12, -B11
        W[0, 3] = 1; W[2, 3] = 1 # C11, C21
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1 # A11, A12
        V[3, 4] = 1 # B22
        W[0, 4] = -1; W[1, 4] = 1 # -C11, C12
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1 # A21, -A11
        V[0, 5] = 1; V[1, 5] = 1 # B11, B12
        W[3, 5] = 1 # C22
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1 # A12, -A22
        V[2, 6] = 1; V[3, 6] = 1 # B21, B22
        W[0, 6] = 1; W[3, 6] = 1 # C11, C22
        
        return U, V, W

    # Helper to create 4x4x4 Strassen decomposition
    def strassen_4x4():
        # Split into 2x2 blocks
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # U, V, W from strassen_2x2 operate on 2x2 blocks
        # We need to map block indices to full indices
        # Block (bi, bj) corresponds to rows 2*bi .. 2*bi+1, cols 2*bj .. 2*bj+1
        
        U_2x2, V_2x2, W_2x2 = strassen_2x2()
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # The 7 terms of the outer Strassen
        # For each outer term t (0..6), we have 7 inner terms
        # Total 49 terms
        inner_idx = 0
        for t_outer in range(7):
            # Extract coefficients for A, B, C blocks for this outer term
            # U_2x2 is (4, 7). Rows are 0,1,2,3 corresponding to A11, A12, A21, A22
            # V_2x2 is (4, 7). Rows are 0,1,2,3 corresponding to B11, B12, B21, B22
            # W_2x2 is (4, 7). Rows are 0,1,2,3 corresponding to C11, C12, C21, C22
            
            for t_inner in range(7):
                # Coefficients
                u_coeff = U_2x2[:, t_outer] # shape (4,)
                v_coeff = V_2x2[:, t_outer] # shape (4,)
                w_coeff = W_2x2[:, t_outer] # shape (4,)
                
                # Inner decomposition for the 2x2 block multiplications
                # We need to sum over the 4 block positions?
                # No, the outer Strassen combines the 4 blocks into 7 "super" blocks.
                # Then each "super" block multiplication is a 2x2x2 multiplication.
                # So for each outer term, we effectively perform a 2x2x2 multiplication
                # on the linear combinations.
                
                # Wait, the structure is:
                # M_outer_t = (Sum_a U_a A_block_a) * (Sum_b V_b B_block_b)
                # This is a product of two 2x2 matrices.
                # We decompose this 2x2x2 product using strassen_2x2().
                # The result is added to C blocks according to W_2x2.
                
                # So for a fixed outer term t_outer:
                # We generate 7 inner terms.
                # For each inner term t_inner:
                # The U part is the linear combination of A blocks defined by u_coeff,
                # further decomposed by U_2x2[:, t_inner].
                
                # Let's trace indices.
                # A_block_a is 2x2.
                # A_full indices for block a:
                # a=0 (11): rows 0,1 cols 0,1 -> idx 0,1,4,5
                # a=1 (12): rows 0,1 cols 2,3 -> idx 2,3,6,7
                # a=2 (21): rows 2,3 cols 0,1 -> idx 8,9,12,13
                # a=3 (22): rows 2,3 cols 2,3 -> idx 10,11,14,15
                
                # Similarly for B and C.
                
                # The inner decomposition strassen_2x2 gives U_inner, V_inner, W_inner.
                # U_inner[i, t_inner] is coeff for A_local[i].
                # A_local is the 2x2 matrix formed by the linear combination.
                
                # So U_full[i_full, global_idx] += u_coeff[a] * U_inner[i_local, t_inner]
                # where i_full corresponds to i_local in block a.
                
                U_in, V_in, W_in = strassen_2x2()
                
                for a in range(4):
                    if u_coeff[a] != 0:
                        # Map local indices to full indices
                        # Block a local (r, c) -> full (2*block_r + r, 2*block_c + c)
                        # a=0: br=0, bc=0
                        # a=1: br=0, bc=1
                        # a=2: br=1, bc=0
                        # a=3: br=1, bc=1
                        br = a // 2
                        bc = a % 2
                        for r in range(2):
                            for c in range(2):
                                local_idx = r * 2 + c
                                full_idx = (2 * br + r) * 4 + (2 * bc + c)
                                U[full_idx, inner_idx] += u_coeff[a] * U_in[local_idx, t_inner]
                
                for b in range(4):
                    if v_coeff[b] != 0:
                        br = b // 2
                        bc = b % 2
                        for r in range(2):
                            for c in range(2):
                                local_idx = r * 2 + c
                                full_idx = (2 * br + r) * 4 + (2 * bc + c)
                                V[full_idx, inner_idx] += v_coeff[b] * V_in[local_idx, t_inner]
                
                for c in range(4):
                    if w_coeff[c] != 0:
                        br = c // 2
                        bc = c % 2
                        for r in range(2):
                            for c_local in range(2):
                                local_idx = r * 2 + c_local
                                full_idx = (2 * br + r) * 4 + (2 * bc + c_local)
                                W[full_idx, inner_idx] += w_coeff[c] * W_in[local_idx, t_inner]
                
                inner_idx += 1
        
        return U, V, W

    # Build the 5x5 decomposition
    # Split 5 = 4 + 1
    # Indices 0..3 are block, 4 is tail
    
    U_44, V_44, W_44 = strassen_4x4()
    
    # Schoolbook for rectangular parts
    # 4x1x4 (A_col B_row)
    U_414, V_414, W_414 = schoolbook(4, 1, 4)
    # 4x4x1 (A_block B_col)
    U_441, V_441, W_441 = schoolbook(4, 4, 1)
    # 1x4x4 (A_row B_block)
    U_144, V_144, W_144 = schoolbook(1, 4, 4)
    # 4x1x1 (A_col B_scalar)
    U_411, V_411, W_411 = schoolbook(4, 1, 1)
    # 1x1x4 (A_scalar B_row)
    U_114, V_114, W_114 = schoolbook(1, 1, 4)
    # 1x4x1 (A_row B_col)
    U_141, V_141, W_141 = schoolbook(1, 4, 1)
    # 1x1x1 (A_scalar B_scalar)
    U_111, V_111, W_111 = schoolbook(1, 1, 1)
    
    # Assemble
    # Total rank = 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 110
    
    total_rank = 110
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    offset = 0
    
    # 1. C_block (0:4, 0:4) from A_block B_block
    # Maps to U[0:16, offset:offset+49], etc.
    U[0:16, offset:offset+49] = U_44
    V[0:16, offset:offset+49] = V_44
    W[0:16, offset:offset+49] = W_44
    offset += 49
    
    # 2. C_block (0:4, 0:4) from A_col B_row
    # A_col is (0:4, 4) -> U indices 4, 8, 12, 16?
    # U index for A[i, 4] is i*5 + 4.
    # i=0..3 -> 4, 9, 14, 19.
    # B_row is (4, 0:4) -> V indices 4*5+0..3 = 20..23.
    # C_block is (0:4, 0:4) -> W indices k*5+i.
    # i=0..3, k=0..3 -> 0..3, 5..8, 10..13, 15..18.
    
    # We need to map the schoolbook(4,1,4) factors to these indices.
    # schoolbook(4,1,4) assumes A is 4x1, B is 1x4, C is 4x4.
    # U_sb[i*1+j, t] = U_sb[i, t]. Here j is always 0.
    # So U_sb row i corresponds to A_col row i.
    # We need to place these in U[4, 9, 14, 19, offset...].
    
    for i in range(4):
        U[i*5 + 4, offset:offset+16] = U_414[i]
        
    for j in range(4): # j is col of C, which is k in B_row?
        # B_row is 1x4. V_sb[j*4+k, t]. Here j=0.
        # So V_sb[k, t] corresponds to B_row[0, k].
        # B_row index in V is 4*5 + k = 20 + k.
        V[20 + j, offset:offset+16] = V_414[j] # V_414 is (4, 16). Row j is k=j.
        
    for i in range(4):
        for k in range(4):
            W[k*5 + i, offset:offset+16] = W_414[i*4 + k] # W_414 is (16, 16). Row i*4+k is (i,k).
            
    offset += 16
    
    # 3. C_col (0:4, 4) from A_block B_col
    # A_block (0:4, 0:4) -> U 0..15.
    # B_col (0:4, 4) -> V 4, 9, 14, 19.
    # C_col (0:4, 4) -> W 4, 9, 14, 19.
    
    # schoolbook(4,4,1)
    # U_sb[i*4+j, t].
    for i in range(4):
        for j in range(4):
            U[i*5 + j, offset:offset+16] = U_441[i*4 + j]
            
    for j in range(4):
        V[j*5 + 4, offset:offset+16] = V_441[j*4 + 0] # V_441 is (16, 16). j*4+0 is (j,0).
        
    for i in range(4):
        W[4*5 + i, offset:offset+16] = W_441[0*4 + i] # W_441 is (4, 16). Row i is (0,i).
        
    offset += 16
    
    # 4. C_row (4, 0:4) from A_row B_block
    # A_row (4, 0:4) -> U 20..23.
    # B_block (0:4, 0:4) -> V 0..15.
    # C_row (4, 0:4) -> W 20..23.
    
    # schoolbook(1,4,4)
    # U_sb[0*4+j, t] = U_sb[j, t].
    for j in range(4):
        U[20 + j, offset:offset+16] = U_144[j]
        
    for i in range(4):
        for j in range(4):
            V[i*5 + j, offset:offset+16] = V_144[i*4 + j]
            
    for k in range(4):
        W[4*5 + k, offset:offset+16] = W_144[0*4 + k]
        
    offset += 16
    
    # 5. C_col (0:4, 4) from A_col B_scalar
    # A_col (0:4, 4) -> U 4, 9, 14, 19.
    # B_scalar (4, 4) -> V 24.
    # C_col (0:4, 4) -> W 4, 9, 14, 19.
    
    # schoolbook(4,1,1)
    for i in range(4):
        U[i*5 + 4, offset:offset+4] = U_411[i]
        
    V[24, offset:offset+4] = V_411[0]
        
    for i in range(4):
        W[4*5 + i, offset:offset+4] = W_411[i]
        
    offset += 4
    
    # 6. C_row (4, 0:4) from A_scalar B_row
    # A_scalar (4, 4) -> U 24.
    # B_row (4, 0:4) -> V 20..23.
    # C_row (4, 0:4) -> W 20..23.
    
    # schoolbook(1,1,4)
    U[24, offset:offset+4] = U_114[0]
    
    for k in range(4):
        V[20 + k, offset:offset+4] = V_114[k]
        
    for k in range(4):
        W[4*5 + k, offset:offset+4] = W_114[k]
        
    offset += 4
    
    # 7. C_scalar (4, 4) from A_row B_col
    # A_row (4, 0:4) -> U 20..23.
    # B_col (0:4, 4) -> V 4, 9, 14, 19.
    # C_scalar (4, 4) -> W 24.
    
    # schoolbook(1,4,1)
    for j in range(4):
        U[20 + j, offset:offset+4] = U_141[j]
        
    for j in range(4):
        V[j*5 + 4, offset:offset+4] = V_141[j*4 + 0]
        
    W[24, offset:offset+4] = W_141[0]
        
    offset += 4
    
    # 8. C_scalar (4, 4) from A_scalar B_scalar
    # U 24, V 24, W 24.
    
    U[24, offset:offset+1] = U_111[0]
    V[24, offset:offset+1] = V_111[0]
    W[24, offset:offset+1] = W_111[0]
    
    offset += 1
    
    # Convert to lists of floats (or ints)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

