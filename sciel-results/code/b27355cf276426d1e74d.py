import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    DEADLINE = start_time + 135

    def verify(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        np.random.seed(42)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_mat[:, r].reshape(n, m)
            v_r = V_mat[:, r].reshape(m, p)
            w_r = W_mat[:, r].reshape(p, n)
            
            for i in range(n):
                for j in range(p):
                    ua = np.dot(u_r[i, :], A_test[i, :])
                    vb = np.dot(v_r[:, j], B_test[:, j])
                    C_computed[i, j] += w_r[j, i] * ua * vb
        
        return np.allclose(C_computed, C_expected, atol=1e-10)

    # Try rank-28: 4 independent Strassen 2x2 multiplications
    # A is 2x4, B is 4x8 = four 4x2 blocks
    # Each block: C_block = A * B_block (2x2 result)
    # Strassen for 2x2: 7 multiplications per block, 4 blocks = 28 total
    
    rank_28 = 28
    U = np.zeros((8, rank_28))
    V = np.zeros((32, rank_28))
    W = np.zeros((16, rank_28))
    
    col = 0
    for block in range(4):
        # B_block has rows 0-3, cols block*2 to block*2+1
        # A has rows 0-1, cols 0-3
        # C_block has rows 0-1, cols block*2 to block*2+1
        
        # Strassen 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # A indices: A[i, j] -> U[i*4+j]
        # B indices: B[j, k] -> V[j*8+k]
        # C indices: C[i, k] -> W[k*2+i] (trace convention W[k*n+i, r])
        
        b00 = 0*8 + block*2
        b01 = 0*8 + block*2 + 1
        b10 = 1*8 + block*2
        b11 = 1*8 + block*2 + 1
        b20 = 2*8 + block*2
        b21 = 2*8 + block*2 + 1
        b30 = 3*8 + block*2
        b31 = 3*8 + block*2 + 1
        
        c00 = (block*2)*2 + 0
        c01 = (block*2+1)*2 + 0
        c10 = (block*2)*2 + 1
        c11 = (block*2+1)*2 + 1
        
        # M1 = (A00+A11)*(B00+B11)
        # A00 = A[0,0], A11 = A[1,1]
        # B00 = B[0, block*2], B11 = B[1, block*2+1]
        U[0*4+0, col] += 1  # A[0,0]
        U[1*4+1, col] += 1  # A[1,1]
        V[b00, col] += 1     # B[0, block*2]
        V[b11, col] += 1     # B[1, block*2+1]
        W[c00, col] += 1     # C[0, block*2]
        W[c11, col] += 1     # C[1, block*2+1]
        col += 1
        
        # M2 = (A10+A11)*B00
        U[1*4+0, col] += 1  # A[1,0]
        U[1*4+1, col] += 1  # A[1,1]
        V[b00, col] += 1     # B[0, block*2]
        W[c10, col] += 1     # C[1, block*2]
        W[c11, col] += -1    # C[1, block*2+1]
        col += 1
        
        # M3 = A00*(B01-B11)
        U[0*4+0, col] += 1  # A[0,0]
        V[b01, col] += 1     # B[0, block*2+1]
        V[b11, col] += -1    # B[1, block*2+1]
        W[c01, col] += 1     # C[0, block*2+1]
        W[c11, col] += 1     # C[1, block*2+1]
        col += 1
        
        # M4 = A11*(B10-B00)
        U[1*4+1, col] += 1  # A[1,1]
        V[b10, col] += 1     # B[1, block*2]
        V[b00, col] += -1    # B[0, block*2]
        W[c00, col] += 1     # C[0, block*2]
        W[c10, col] += 1     # C[1, block*2]
        col += 1
        
        # M5 = (A00+A01)*B11
        U[0*4+0, col] += 1  # A[0,0]
        U[0*4+1, col] += 1  # A[0,1]
        V[b11, col] += 1     # B[1, block*2+1]
        W[c00, col] += -1    # C[0, block*2]
        W[c01, col] += 1     # C[0, block*2+1]
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[1*4+0, col] += 1  # A[1,0]
        U[0*4+0, col] += -1 # A[0,0]
        V[b00, col] += 1     # B[0, block*2]
        V[b01, col] += 1     # B[0, block*2+1]
        W[c11, col] += 1     # C[1, block*2+1]
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[0*4+1, col] += 1  # A[0,1]
        U[1*4+1, col] += -1 # A[1,1]
        V[b10, col] += 1     # B[1, block*2]
        V[b11, col] += 1     # B[1, block*2+1]
        W[c00, col] += 1     # C[0, block*2]
        col += 1
    
    assert col == 28, f"Expected 28, got {col}"
    
    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to rank-56
    rank_56 = 56
    U56 = np.zeros((8, rank_56))
    V56 = np.zeros((32, rank_56))
    W56 = np.zeros((16, rank_56))
    
    col = 0
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0*4 + a_col_off
        a01 = 0*4 + a_col_off + 1
        a10 = 1*4 + a_col_off
        a11 = 1*4 + a_col_off + 1
        
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        
        U56[a00, col] += 1
        U56[a11, col] += 1
        V56[b00, col] += 1
        V56[b11, col] += 1
        W56[c00, col] += 1
        W56[c11, col] += 1
        col += 1
        
        U56[a10, col] += 1
        U56[a11, col] += 1
        V56[b00, col] += 1
        W56[c10, col] += 1
        W56[c11, col] += -1
        col += 1
        
        U56[a00, col] += 1
        V56[b01, col] += 1
        V56[b11, col] += -1
        W56[c01, col] += 1
        W56[c11, col] += 1
        col += 1
        
        U56[a11, col] += 1
        V56[b10, col] += 1
        V56[b00, col] += -1
        W56[c00, col] += 1
        W56[c10, col] += 1
        col += 1
        
        U56[a00, col] += 1
        U56[a01, col] += 1
        V56[b11, col] += 1
        W56[c00, col] += -1
        W56[c01, col] += 1
        col += 1
        
        U56[a10, col] += 1
        U56[a00, col] += -1
        V56[b00, col] += 1
        V56[b01, col] += 1
        W56[c11, col] += 1
        col += 1
        
        U56[a01, col] += 1
        U56[a11, col] += -1
        V56[b10, col] += 1
        V56[b11, col] += 1
        W56[c00, col] += 1
        col += 1
    
    assert col == 56
    
    if not verify(U56, V56, W56):
        return {
            "U": U56.tolist(),
            "V": V56.tolist(),
            "W": W56.tolist()
        }
    
    return {
        "U": U56.tolist(),
        "V": V56.tolist(),
        "W": W56.tolist()
    }
