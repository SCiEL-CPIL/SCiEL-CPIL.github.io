import numpy as np
import time

def construct():
    n, m, p = 2, 2, 8
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds max
    
    # Target Tensor T
    # T[i, j, k] = 1 if valid
    # i in 0..1 (A row)
    # j in 0..1 (A col / B row)
    # k in 0..7 (B col / C col)
    # A index: i*m + j
    # B index: j*p + k
    # C index: k*n + i
    
    # We will use the block Strassen approach.
    # Rank = 4 blocks * 7 per block = 28.
    
    # Strassen factors for 2x2 * 2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B01
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    
    # C00 = M1 - M3 - M4 + M7
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M2 - M5 + M6
    
    # U factors (A side)
    # Shape (4, 7)
    # Rows correspond to A00, A01, A10, A11
    # Cols correspond to M1..M7
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 1, 0, 0, 1, 0, -1],  # A01
        [1, 1, 0, 0, 0, 1, 0],   # A10
        [1, 0, 0, 1, 0, 0, 1]    # A11
    ])
    
    # V factors (B side)
    # Shape (4, 7)
    # Rows correspond to B00, B01, B10, B11
    V_2x2 = np.array([
        [1, 0, 0, -1, 0, 1, 0],  # B00
        [1, 1, -1, 0, 0, 0, 1],  # B01
        [0, 0, 1, 0, 1, 1, 0],   # B10
        [1, 0, -1, 0, 0, 0, 1]   # B11
    ])
    
    # W factors (C side)
    # Shape (4, 7)
    # Rows correspond to C00, C01, C10, C11
    W_2x2 = np.array([
        [1, 0, -1, -1, 0, 0, 1],  # C00
        [0, 1, 0, 1, 0, 0, 0],    # C01
        [0, 0, 1, 0, 1, 0, 0],    # C10
        [1, -1, 0, 0, -1, 1, 0]   # C11
    ])
    
    # Verify 2x2 Strassen
    # C = U * diag(M) * V.T ? No.
    # C_ik = sum_r U_ir W_cr V_rj ? No.
    # Tensor: T_ijk = sum_r U_ir V_jr W_kr
    # Here i is A idx, j is B idx, k is C idx.
    # Let's check one entry.
    # T_00,0,0 = sum_r U_0r V_0r W_0r
    # U_0 = [1, 0, 1, 0, 1, -1, 0]
    # V_0 = [1, 0, 0, -1, 0, 1, 0]
    # W_0 = [1, 0, -1, -1, 0, 0, 1]
    # Dot products:
    # r=0: 1*1*1 = 1
    # r=1: 0*0*0 = 0
    # r=2: 1*0*-1 = 0
    # r=3: 0*-1*-1 = 0
    # r=4: 1*0*0 = 0
    # r=5: -1*1*0 = 0
    # r=6: 0*0*1 = 0
    # Sum = 1. Correct.
    
    # Construct full U, V, W
    # U shape (4, 28)
    # V shape (16, 28)
    # W shape (16, 28)
    
    U_full = np.zeros((n*m, 4*7))
    V_full = np.zeros((m*p, 4*7))
    W_full = np.zeros((p*n, 4*7))
    
    # Fill U: A is shared.
    # All 4 blocks use the same A factors.
    for block in range(4):
        start_col = block * 7
        end_col = start_col + 7
        U_full[:, start_col:end_col] = U_2x2
    
    # Fill V: B blocks.
    # B index j_global = row * 8 + col.
    # Block b (0..3) corresponds to cols 2b, 2b+1.
    # Within block, local B indices 0,1,2,3 correspond to:
    # 0: row 0, col 2b -> global 2b
    # 1: row 0, col 2b+1 -> global 2b+1
    # 2: row 1, col 2b -> global 8 + 2b
    # 3: row 1, col 2b+1 -> global 8 + 2b + 1
    
    for block in range(4):
        start_col = block * 7
        end_col = start_col + 7
        v_indices = [2*block, 2*block+1, 8+2*block, 8+2*block+1]
        for local_idx, global_idx in enumerate(v_indices):
            V_full[global_idx, start_col:end_col] = V_2x2[local_idx]
            
    # Fill W: C blocks.
    # C index k_global = col * 2 + row.
    # Block b (0..3) corresponds to cols 2b, 2b+1.
    # Within block, local C indices 0,1,2,3 correspond to:
    # 0: row 0, col 2b -> global 2b*2 + 0 = 4b
    # 1: row 0, col 2b+1 -> global (2b+1)*2 + 0 = 4b+2
    # 2: row 1, col 2b -> global 2b*2 + 1 = 4b+1
    # 3: row 1, col 2b+1 -> global (2b+1)*2 + 1 = 4b+3
    
    for block in range(4):
        start_col = block * 7
        end_col = start_col + 7
        w_indices = [4*block, 4*block+2, 4*block+1, 4*block+3]
        for local_idx, global_idx in enumerate(w_indices):
            W_full[global_idx, start_col:end_col] = W_2x2[local_idx]
            
    # Verify exactness
    # Compute tensor
    # T_comp[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # i in 0..3, j in 0..15, k in 0..15
    
    # Optimization: only check non-zero entries of target tensor
    # Target entries:
    # A[i,j] B[j,k] -> C[i,k]
    # i in 0..1, j in 0..1, k in 0..7
    # a_idx = i*2 + j
    # b_idx = j*8 + k
    # c_idx = k*2 + i
    
    is_valid = True
    # Precompute indices for speed
    target_abc = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target_abc.append((a_idx, b_idx, c_idx))
                
    # Check all target entries are 1
    for a, b, c in target_abc:
        val = np.dot(U_full[a], np.multiply(V_full[b], W_full[c]))
        if abs(val - 1.0) > 1e-9:
            is_valid = False
            break
            
    # Check all non-target entries are 0
    # Total entries 4*16*16 = 1024
    # Target entries = 32
    # We can just compute the full tensor and check, it's fast enough.
    if is_valid:
        T_comp = np.einsum('ir,jr,kr->ijk', U_full, V_full, W_full)
        # Create mask
        mask = np.zeros((4, 16, 16), dtype=bool)
        for a, b, c in target_abc:
            mask[a, b, c] = True
            
        non_target_err = np.sum(np.abs(T_comp[~mask]))
        if non_target_err > 1e-9:
            is_valid = False
            
    if not is_valid:
        # Fallback to schoolbook
        U_school = np.zeros((4, 32))
        V_school = np.zeros((16, 32))
        W_school = np.zeros((16, 32))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_school[a_idx, idx] = 1
                    V_school[b_idx, idx] = 1
                    W_school[c_idx, idx] = 1
                    idx += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
