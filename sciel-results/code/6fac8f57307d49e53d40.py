import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Local indices:
    U[i*2+j, r] for A[i,j]
    V[j*2+k, r] for B[j,k]
    W[k*2+i, r] for C[k,i] -> Wait, task says W[k*n+i]. For 2x2, n=2, so W[k*2+i].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U[0, 0] = 1; U[3, 0] = 1 # A00 (0,0), A11 (1,1) -> idx 0, 3
    V[0, 0] = 1; V[3, 0] = 1 # B00 (0,0), B11 (1,1) -> idx 0, 3
    W[0, 0] = 1; W[3, 0] = 1 # C00 (k=0,i=0 -> 0), C11 (k=1,i=1 -> 3)
    
    # Term 2: (A10+A11)B00 -> C10 += 1, C11 -= 1
    U[2, 1] = 1; U[3, 1] = 1 # A10 (1,0 -> 2), A11 (1,1 -> 3)
    V[0, 1] = 1             # B00 (0,0 -> 0)
    W[2, 1] = 1; W[3, 1] = -1 # C10 (k=1,i=0 -> 2), C11 (k=1,i=1 -> 3)
    
    # Term 3: A00(B01-B11) -> C01 += 1, C11 += 1
    U[0, 2] = 1             # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01 (0,1 -> 1), B11 (1,1 -> 3)
    W[1, 2] = 1; W[3, 2] = 1  # C01 (k=0,i=1 -> 1), C11 (k=1,i=1 -> 3)
    
    # Term 4: A11(B10-B00) -> C00 += 1, C10 += 1
    U[3, 3] = 1             # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10 (1,0 -> 2), B00 (0,0 -> 0)
    W[0, 3] = 1; W[2, 3] = 1  # C00 (k=0,i=0 -> 0), C10 (k=1,i=0 -> 2)
    
    # Term 5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    U[0, 4] = 1; U[1, 4] = 1 # A00 (0,0), A01 (0,1 -> 1)
    V[3, 4] = 1             # B11 (1,1 -> 3)
    W[0, 4] = -1; W[1, 4] = 1 # C00 (k=0,i=0), C01 (k=0,i=1 -> 1)
    
    # Term 6: (A10-A00)(B00+B01) -> C11 += 1
    U[2, 5] = 1; U[0, 5] = -1 # A10 (1,0 -> 2), A00 (0,0)
    V[0, 5] = 1; V[1, 5] = 1  # B00 (0,0), B01 (0,1 -> 1)
    W[3, 5] = 1             # C11 (k=1,i=1 -> 3)
    
    # Term 7: (A01-A11)(B10+B11) -> C00 += 1
    U[1, 6] = 1; U[3, 6] = -1 # A01 (0,1 -> 1), A11 (1,1 -> 3)
    V[2, 6] = 1; V[3, 6] = 1  # B10 (1,0 -> 2), B11 (1,1 -> 3)
    W[0, 6] = 1             # C00 (k=0,i=0 -> 0)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 4, 4, 6
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    # Total rank: 49 (for 4x4x4) + 28 (for 4x4x2) = 77
    total_rank = 77
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U2, V2, W2 = _get_strassen_2x2x2()
    
    # Helper to add a 2x2x2 block contribution
    # a_row_off, a_col_off: offset of the 2x2 A block in the 4x4 A
    # b_row_off, b_col_off: offset of the 2x2 B block in the 4x6 B
    # c_row_off, c_col_off: offset of the 2x2 C block in the 4x6 C
    # Note: B is 4x6, C is 4x6.
    # A is 4x4.
    # Local indices in 2x2x2:
    # A: i,j in 0..1
    # B: j,k in 0..1
    # C: k,i in 0..1 (but W is indexed k*n+i, here n=2 locally? No, W maps to global C)
    
    def add_block(a_r, a_c, b_r, b_c, c_r, c_c, col_start):
        # U part: A[i,j] -> global idx (a_r+i)*4 + (a_c+j)
        for i in range(2):
            for j in range(2):
                local_u_idx = i*2 + j
                global_u_idx = (a_r + i) * 4 + (a_c + j)
                U[global_u_idx, col_start:col_start+7] += U2[local_u_idx, :]
                
        # V part: B[j,k] -> global idx (b_r+j)*6 + (b_c+k)
        for j in range(2):
            for k in range(2):
                local_v_idx = j*2 + k
                global_v_idx = (b_r + j) * 6 + (b_c + k)
                V[global_v_idx, col_start:col_start+7] += V2[local_v_idx, :]
                
        # W part: C[k,i] -> global idx (c_r+k)*4 + (c_c+i)
        # Wait, W is indexed [k*n+i, r]. n=4.
        # Local C is 2x2. k,i in 0..1.
        # Global C is 4x6.
        # c_r, c_c are offsets in rows (k) and cols (i)?
        # C[k,i] corresponds to row k, col i.
        # So global row = c_r + k, global col = c_c + i.
        # Global W idx = (c_r + k) * 4 + (c_c + i).
        for k in range(2):
            for i in range(2):
                local_w_idx = k*2 + i
                global_w_idx = (c_r + k) * 4 + (c_c + i)
                W[global_w_idx, col_start:col_start+7] += W2[local_w_idx, :]

    # Part 1: 4x4x4 (Columns 0-3 of B and C)
    # We use 7 Strassen terms on the 2x2 blocks of A and B (cols 0-3).
    # A blocks:
    # A00: r=0,c=0; A01: r=0,c=2; A10: r=2,c=0; A11: r=2,c=2
    # B blocks (cols 0-3):
    # B00: r=0,c=0; B01: r=0,c=2; B10: r=2,c=0; B11: r=2,c=2
    # C blocks (cols 0-3):
    # C00: r=0,c=0; C01: r=0,c=2; C10: r=2,c=0; C11: r=2,c=2
    
    # Strassen 4x4x4 decomposition:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # We define the terms as (A_blocks, B_blocks, C_blocks)
    # Each block is (row_off, col_off, coeff)
    
    terms_4x4x4 = [
        # M1
        ([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)]),
        # M2
        ([(2,0,1), (2,2,1)], [(0,0,1)], [(2,0,1), (2,2,-1)]),
        # M3
        ([(0,0,1)], [(0,2,1), (2,2,-1)], [(0,2,1), (2,2,1)]),
        # M4
        ([(2,2,1)], [(2,0,1), (0,0,-1)], [(0,0,1), (2,0,1)]),
        # M5
        ([(0,0,1), (0,2,1)], [(2,2,1)], [(0,0,-1), (0,2,1)]),
        # M6
        ([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], [(2,2,1)]),
        # M7
        ([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,0,1)]),
    ]
    
    col = 0
    for a_blocks, b_blocks, c_blocks in terms_4x4x4:
        # This term represents a linear combination of 2x2x2 multiplications.
        # We need to sum the contributions of each (A_block, B_block, C_block) triple?
        # No, the Strassen algorithm computes M_k = A'_k B'_k where A'_k and B'_k are sums of blocks.
        # Then C blocks are sums/differences of M_k.
        # The tensor decomposition sums the products of the linear forms.
        # So for each term k, we have one U vector, one V vector, one W vector.
        # The U vector is the sum of the A block contributions.
        # The V vector is the sum of the B block contributions.
        # The W vector is the sum of the C block contributions.
        
        U_term = np.zeros((16, 7))
        V_term = np.zeros((24, 7))
        W_term = np.zeros((24, 7))
        
        for a_r, a_c, coeff in a_blocks:
            for i in range(2):
                for j in range(2):
                    local_u_idx = i*2 + j
                    global_u_idx = (a_r + i) * 4 + (a_c + j)
                    U_term[global_u_idx, :] += coeff * U2[local_u_idx, :]
                    
        for b_r, b_c, coeff in b_blocks:
            for j in range(2):
                for k in range(2):
                    local_v_idx = j*2 + k
                    global_v_idx = (b_r + j) * 6 + (b_c + k)
                    V_term[global_v_idx, :] += coeff * V2[local_v_idx, :]
                    
        for c_r, c_c, coeff in c_blocks:
            for k in range(2):
                for i in range(2):
                    local_w_idx = k*2 + i
                    global_w_idx = (c_r + k) * 4 + (c_c + i)
                    W_term[global_w_idx, :] += coeff * W2[local_w_idx, :]
                    
        U[:, col:col+7] = U_term
        V[:, col:col+7] = V_term
        W[:, col:col+7] = W_term
        col += 7
        
    # Part 2: 4x4x2 (Columns 4-5 of B and C)
    # We decompose this into 4 independent 2x2x2 multiplications.
    # 1. A00 (r0,c0) * B_0:2,4:6 (r0,c4) -> C_0:2,4:6 (r0,c4)
    # 2. A01 (r0,c2) * B_2:4,4:6 (r2,c4) -> C_0:2,4:6 (r0,c4)
    # 3. A10 (r2,c0) * B_0:2,4:6 (r0,c4) -> C_2:4,4:6 (r2,c4)
    # 4. A11 (r2,c2) * B_2:4,4:6 (r2,c4) -> C_2:4,4:6 (r2,c4)
    
    terms_4x4x2 = [
        (0, 0, 0, 4, 0, 4), # a_r, a_c, b_r, b_c, c_r, c_c
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]
    
    for a_r, a_c, b_r, b_c, c_r, c_c in terms_4x4x2:
        # Each is a pure 2x2x2 multiplication with coeff 1
        U_term = np.zeros((16, 7))
        V_term = np.zeros((24, 7))
        W_term = np.zeros((24, 7))
        
        for i in range(2):
            for j in range(2):
                local_u_idx = i*2 + j
                global_u_idx = (a_r + i) * 4 + (a_c + j)
                U_term[global_u_idx, :] = U2[local_u_idx, :]
                
        for j in range(2):
            for k in range(2):
                local_v_idx = j*2 + k
                global_v_idx = (b_r + j) * 6 + (b_c + k)
                V_term[global_v_idx, :] = V2[local_v_idx, :]
                
        for k in range(2):
            for i in range(2):
                local_w_idx = k*2 + i
                global_w_idx = (c_r + k) * 4 + (c_c + i)
                W_term[global_w_idx, :] = W2[local_w_idx, :]
                
        U[:, col:col+7] = U_term
        V[:, col:col+7] = V_term
        W[:, col:col+7] = W_term
        col += 7
        
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                # Ensure exact integer/half-integer
                rounded = round(val * 2) / 2
                r.append(rounded)
            result.append(r)
        return result
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
