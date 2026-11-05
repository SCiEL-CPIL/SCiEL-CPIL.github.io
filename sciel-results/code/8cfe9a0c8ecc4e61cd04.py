"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def get_2x2x1_decomp():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        # U: 4x4, V: 2x4, W: 2x4
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # Terms: (i,j) * (j,k) -> (i,k)
        # k=0
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W

    def get_2x2x2_decomp():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # Strassen terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U rows: 0=A11, 1=A12, 2=A21, 3=A22
        # V rows: 0=B11, 1=B12, 2=B21, 3=B22
        # W rows: 0=C11, 1=C12, 2=C21, 3=C22
        
        # M1
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        # M2
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        # M3
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        # M4
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        # M5
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        # M6
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        # M7
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        
        return U, V, W

    def get_2x2x3_decomp():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3."""
        # Split 2x3 into 2x1 and 2x2
        U1, V1, W1 = get_2x2x1_decomp()
        U2, V2, W2 = get_2x2x2_decomp()
        
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Copy 2x2x1 part (cols 0-3)
        # V rows for 2x1 part: 0,1 (cols 0,1 of B)
        # W rows for 2x1 part: 0,1 (cols 0,1 of C)
        U[:, 0:4] = U1
        V[0:2, 0:4] = V1
        W[0:2, 0:4] = W1
        
        # Copy 2x2x2 part (cols 4-10)
        # V rows for 2x2 part: 2,3,4,5 (cols 2,3 of B? No, B is 2x3.
        # B = [B_col0, B_col1, B_col2]
        # 2x1 part uses B_col0, B_col1?
        # Wait, 2x2x1 is 2x2 * 2x1.
        # If we split 2x3 into 2x1 and 2x2:
        # B = [B1 (2x1) | B2 (2x2)]
        # C = [C1 (2x1) | C2 (2x2)]
        # B1 uses cols 0,1 of B? No, B is 2x3.
        # Let's say B1 is first 1 column, B2 is last 2 columns.
        # V1 is 2x4. Rows 0,1 correspond to B1.
        # V2 is 4x7. Rows 0-3 correspond to B2.
        # In V (6x11), rows 0,1 are B1. Rows 2-5 are B2.
        # W1 is 2x4. Rows 0,1 correspond to C1.
        # W2 is 4x7. Rows 0-3 correspond to C2.
        # In W (6x11), rows 0,1 are C1. Rows 2-5 are C2.
        
        U[:, 4:11] = U2
        V[2:6, 4:11] = V2
        W[2:6, 4:11] = W2
        
        return U, V, W

    def get_4x4x6_decomp():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        U_sub, V_sub, W_sub = get_2x2x3_decomp()
        rank_sub = 11
        total_rank = 7 * rank_sub
        
        U = np.zeros((16, total_rank))
        V = np.zeros((24, total_rank))
        W = np.zeros((24, total_rank))
        
        # Strassen 7 terms for 4x4x4 block structure
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]] (where B blocks are 2x3)
        # C = [[C11, C12], [C21, C22]] (where C blocks are 2x3)
        
        # U_sub is 4x11 (2x2 A blocks)
        # V_sub is 6x11 (2x3 B blocks)
        # W_sub is 6x11 (2x3 C blocks)
        
        # Mapping of local indices to global indices
        # U local: 0=A11, 1=A12, 2=A21, 3=A22
        # V local: 0=B11_row0, 1=B11_row1, 2=B12_row0, 3=B12_row1, 4=B21_row0, 5=B21_row1, 6=B22_row0, 7=B22_row1?
        # No, V_sub is 6x11.
        # B is 2x3.
        # B11 is 2x3? No.
        # B is 4x6.
        # B = [[B11 (2x3), B12 (2x3)], [B21 (2x3), B22 (2x3)]]
        # V_sub corresponds to a 2x3 matrix.
        # So V_sub rows 0-5 correspond to one 2x3 block.
        
        # Strassen terms:
        # 1. (A11+A22)(B11+B22) -> C11, C22
        # 2. (A21+A22)B11 -> C21, C22
        # 3. A11(B21-B22) -> C12, C22
        # 4. A22(B12-B11) -> C11, C21
        # 5. (A11+A12)B22 -> C11, C12
        # 6. (A21-A11)(B11+B12) -> C22
        # 7. (A12-A22)(B21+B22) -> C11
        
        # We need to map the 2x2 Strassen coefficients to the 4x4 blocks.
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # Each coeff is a list of (local_block, multiplier)
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]}, # M1
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},       # M2
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},       # M3
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},       # M4
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},       # M5
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},       # M6
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},       # M7
        ]
        
        # Offsets for U (4x4 -> 2x2 blocks)
        # U local 0 -> A11 (rows 0-1, cols 0-1)
        # U local 1 -> A12 (rows 0-1, cols 2-3)
        # U local 2 -> A21 (rows 2-3, cols 0-1)
        # U local 3 -> A22 (rows 2-3, cols 2-3)
        u_offsets = {
            0: (0, 0),
            1: (0, 2),
            2: (2, 0),
            3: (2, 2)
        }
        
        # Offsets for V (4x6 -> 2x3 blocks)
        # V local 0-5 -> B11 (rows 0-1, cols 0-2)
        # V local 0-5 -> B12 (rows 0-1, cols 3-5)
        # V local 0-5 -> B21 (rows 2-3, cols 0-2)
        # V local 0-5 -> B22 (rows 2-3, cols 3-5)
        v_offsets = {
            0: (0, 0), # B11
            1: (0, 3), # B12
            2: (2, 0), # B21
            3: (2, 3)  # B22
        }
        
        # Offsets for W (4x6 -> 2x3 blocks)
        # W local 0-5 -> C11 (rows 0-1, cols 0-2)
        # W local 0-5 -> C12 (rows 0-1, cols 3-5)
        # W local 0-5 -> C21 (rows 2-3, cols 0-2)
        # W local 0-5 -> C22 (rows 2-3, cols 3-5)
        w_offsets = {
            0: (0, 0), # C11
            1: (0, 3), # C12
            2: (2, 0), # C21
            3: (2, 3)  # C22
        }
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                # Process U
                for u_local in range(4):
                    u_val = U_sub[u_local, c]
                    if u_val == 0: continue
                    i_l, j_l = u_local // 2, u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offsets[block]
                        U[(i_l+ro)*4 + (j_l+co), base_col+c] += u_val * coeff
                        
                # Process V
                for v_local in range(6):
                    v_val = V_sub[v_local, c]
                    if v_val == 0: continue
                    # v_local 0-5 maps to 2x3 block
                    # row = v_local // 3, col = v_local % 3
                    i_l, j_l = v_local // 3, v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offsets[block]
                        V[(i_l+ro)*6 + (j_l+co), base_col+c] += v_val * coeff
                        
                # Process W
                for w_local in range(6):
                    w_val = W_sub[w_local, c]
                    if w_val == 0: continue
                    # w_local 0-5 maps to 2x3 block
                    # row = w_local // 3, col = w_local % 3
                    i_l, j_l = w_local // 3, w_local % 3
                    for block, coeff in term['C']:
                        ro, co = w_offsets[block]
                        W[(i_l+ro)*4 + (j_l+co), base_col+c] += w_val * coeff
                        
        return U, V, W

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    prod = U_arr[i*4+j, :] * V_arr[j*6+k, :] * W_arr[k*4+i, :]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    # Generate decomposition
    U, V, W = get_4x4x6_decomp()
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 4*4*6
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
                    if time.time() > deadline:
                        break
                if time.time() > deadline: break
            if time.time() > deadline: break
        if time.time() > deadline:
            # Return whatever we have
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
