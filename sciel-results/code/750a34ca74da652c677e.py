import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 6

    # We build a rank-77 decomposition:
    # Part 1: Strassen 4x4x4 (49 terms) for columns 0-3 of B
    # Part 2: 4 Strassen 2x2x2 products (28 terms) for columns 4-5 of B
    #
    # The key structure:
    # - Split A into 4 2x2 blocks: A00, A01, A10, A11
    # - Split B into 2 2x3 blocks vertically: B_top (rows 0-1), B_bot (rows 2-3)
    # - For cols 0-3: use full Strassen 4x4x4
    # - For cols 4-5: 4 independent 2x2@2x2 products via Strassen
    #
    # Index conventions:
    # U: 16 x rank, where U[i, k] with i = row*4 + col of A
    # V: 24 x rank, where V[j, k] with j = row*6 + col of B  
    # W: 24 x rank, where W[idx, k] with idx = col*4 + row of C (trace convention W[k*n+i, r])
    #   Wait, the task says W[k*n+i, r]. Let me re-read.
    #   "The output factor follows the official trace convention W[k*n+i,r]"
    #   So W has shape (p*n, rank) = (24, rank), indexed as W[col*4 + row, term]
    #   This means C[row, col] = sum_k U[row*4+col_a, k] * V[col_a*6+col_b, k] * W[col*4+row, k]
    #   Wait, that's not right either. Let me think more carefully.
    #
    # For matrix multiplication C = A @ B where A is n x m, B is m x p, C is n x p:
    # C[i, j] = sum_k A[i, k] * B[k, j]
    #
    # Tensor decomposition: C[i,j] = sum_t U[i*m + ?, t] * V[?, t] * W[?, t]
    # Actually the standard is:
    # C[i,j] = sum_t U[i, t] * V[j, t] * W[k, t] summed over k too... no.
    #
    # The bilinear form: C[i,j] = sum_t (sum_k U[i*m+k, t] * A[i,k]) * (sum_l V[k*p+l, t] * B[k,l]) ... 
    # No, that's not right either.
    #
    # Standard tensor decomposition for matrix multiplication:
    # C[i,j] = sum_t U[i, t] * V[j, t] * ... no.
    #
    # Actually: C[i,j] = sum_t (U[:, t]^T A) (V[:, t]^T B) ... 
    # Let me use the standard formulation:
    # C[i,j] = sum_t sum_k U[i, k, t] * A[i,k] * V[k, j, t] * B[k,j] * ... 
    #
    # The cleanest: C[i,j] = sum_t (sum_k U[i,k,t] A[i,k]) (sum_l V[k,l,t] B[k,l])
    # No, the indices need to be consistent.
    #
    # Standard: C[i,j] = sum_t (sum_k U[i,k,t] A[i,k]) (sum_k V[k,j,t] B[k,j])
    # where U is n x m x rank, V is m x p x rank.
    # Then C[i,j] = sum_t (sum_k U[i,k,t] A[i,k]) (sum_k V[k,j,t] B[k,j])
    #
    # In the flattened form:
    # U_flat has shape (n*m, rank): U_flat[i*m+k, t] = U[i,k,t]
    # V_flat has shape (m*p, rank): V_flat[k*p+j, t] = V[k,j,t]
    # W_flat has shape (p*n, rank): W_flat[j*n+i, t] = W[i,j,t]
    #
    # And C[i,j] = sum_t W[j*n+i, t] * (sum_k U[i*m+k, t] A[i,k]) * (sum_k V[k*p+j, t] B[k,j])
    #
    # Wait, that gives C[i,j] = sum_t W[j*n+i, t] * U_prod[i,t] * V_prod[j,t]
    # where U_prod[i,t] = sum_k U[i*m+k, t] A[i,k]
    # and V_prod[j,t] = sum_k V[k*p+j, t] B[k,j]
    #
    # So the decomposition is: C[i,j] = sum_t W[j*n+i, t] * U_prod[i,t] * V_prod[j,t]
    #
    # For this to equal sum_k A[i,k] B[k,j], we need the tensor identity.
    #
    # Let me just implement the Strassen decomposition carefully and verify.

    rank = 77
    U = [[0.0]*rank for _ in range(n*m)]  # 16 x 77
    V = [[0.0]*rank for _ in range(m*p)]  # 24 x 77
    W = [[0.0]*rank for _ in range(p*n)]  # 24 x 77

    col_idx = 0

    def add_term(u_entries, v_entries, w_entries):
        nonlocal col_idx
        if col_idx >= rank:
            return
        for idx, coeff in u_entries:
            U[idx][col_idx] += coeff
        for idx, coeff in v_entries:
            V[idx][col_idx] += coeff
        for idx, coeff in w_entries:
            W[idx][col_idx] += coeff
        col_idx += 1

    # Strassen 2x2x2 base terms
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: list of (local_2x2_idx, coeff) where local_2x2_idx = r*2+c
    # v_entries: list of (local_2x2_idx, coeff)
    # w_entries: list of (local_2x2_idx, coeff)
    
    # Strassen 2x2: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    s2x2_u = [
        [(0,1), (3,1)],      # M1: A00+A11
        [(2,1), (3,1)],      # M2: A10+A11
        [(0,1)],             # M3: A00
        [(3,1)],             # M4: A11
        [(0,1), (1,1)],      # M5: A00+A01
        [(2,1), (0,-1)],     # M6: A10-A00
        [(1,1), (3,-1)],     # M7: A01-A11
    ]
    s2x2_v = [
        [(0,1), (3,1)],      # M1: B00+B11
        [(0,1)],             # M2: B00
        [(1,1), (3,-1)],     # M3: B01-B11
        [(2,1), (0,-1)],     # M4: B10-B00
        [(3,1)],             # M5: B11
        [(0,1), (1,1)],      # M6: B00+B01
        [(2,1), (3,1)],      # M7: B10+B11
    ]
    s2x2_w = {
        0: [(0,1), (3,1), (4,-1), (6,1)],   # C00 = M1+M4-M5+M7
        1: [(2,1), (4,1)],                   # C01 = M3+M5
        2: [(1,1), (3,1)],                   # C10 = M2+M4
        3: [(0,1), (1,-1), (2,1), (5,1)],   # C11 = M1-M2+M3+M6
    }

    # Part 1: Strassen 4x4x4 for columns 0-3
    # Split A into 2x2 blocks: A00, A01, A10, A11
    # Split B (cols 0-3) into 2x2 blocks: B00, B01, B10, B11
    # Use Strassen at the block level (7 block-products), each expanded via Strassen 2x2x2 (7 terms)
    # Total: 49 terms

    # Block-level Strassen for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    big_u = [
        [('00',1), ('11',1)],
        [('10',1), ('11',1)],
        [('00',1)],
        [('11',1)],
        [('00',1), ('01',1)],
        [('10',1), ('00',-1)],
        [('01',1), ('11',-1)],
    ]
    big_v = [
        [('00',1), ('11',1)],
        [('00',1)],
        [('01',1), ('11',-1)],
        [('10',1), ('00',-1)],
        [('11',1)],
        [('00',1), ('01',1)],
        [('10',1), ('11',1)],
    ]
    big_w = {
        '00': [(0,1), (3,1), (4,-1), (6,1)],
        '01': [(2,1), (4,1)],
        '10': [(1,1), (3,1)],
        '11': [(0,1), (1,-1), (2,1), (5,1)],
    }

    def a_block_idx(block, r, c):
        """Map block (e.g., '00') and local (r,c) to flat A index."""
        br, bc = int(block[0]), int(block[1])
        return (br*2+r)*4 + (bc*2+c)

    def b_block_idx(block, r, c, col_offset=0):
        """Map block and local (r,c) to flat B index with column offset."""
        br, bc = int(block[0]), int(block[1])
        return (br*2+r)*6 + (bc*2+c+col_offset)

    def c_block_idx(block, r, c, col_offset=0):
        """Map block and local (r,c) to flat C index (W index) with column offset.
        W index = col*4 + row."""
        br, bc = int(block[0]), int(block[1])
        col = bc*2 + c + col_offset
        row = br*2 + r
        return col*4 + row

    for bi in range(7):
        for si in range(7):
            u_entries = []
            for blk, coeff in big_u[bi]:
                for ur, uc_coeff in s2x2_u[si]:
                    local_r, local_c = ur // 2, ur % 2
                    u_entries.append((a_block_idx(blk, local_r, local_c), coeff * uc_coeff))
            
            v_entries = []
            for blk, coeff in big_v[bi]:
                for vr, vc_coeff in s2x2_v[si]:
                    local_r, local_c = vr // 2, vr % 2
                    v_entries.append((b_block_idx(blk, local_r, local_c, 0), coeff * vc_coeff))
            
            w_entries = []
            for cblk, contribs in big_w.items():
                big_coeff = 0
                for mi, mc in contribs:
                    if mi == bi:
                        big_coeff = mc
                        break
                if big_coeff == 0:
                    continue
                for cidx_local, sc_contribs in s2x2_w.items():
                    small_coeff = 0
                    for sk, sc in sc_contribs:
                        if sk == si:
                            small_coeff = sc
                            break
                    if small_coeff == 0:
                        continue
                    local_r, local_c = cidx_local // 2, cidx_local % 2
                    w_entries.append((c_block_idx(cblk, local_r, local_c, 0), big_coeff * small_coeff))
            
            add_term(u_entries, v_entries, w_entries)

    # Part 2: 4x4x2 for columns 4-5 (28 terms)
    # C[0:2, 4:6] = A[0:2,0:2] @ B[0:2,4:6] + A[0:2,2:4] @ B[2:4,4:6]
    # C[2:4, 4:6] = A[2:4,0:2] @ B[0:2,4:6] + A[2:4,2:4] @ B[2:4,4:6]
    # Each 2x2@2x2 via Strassen: 7 terms
    # 4 products * 7 = 28 terms

    products = [
        # (A_block, B_row_block, C_row_block, C_col_offset)
        ('00', 'top', 'top', 4),   # A[0:2,0:2] @ B[0:2,4:6] -> C[0:2,4:6]
        ('01', 'bot', 'top', 4),   # A[0:2,2:4] @ B[2:4,4:6] -> C[0:2,4:6]
        ('10', 'top', 'bot', 4),   # A[2:4,0:2] @ B[0:2,4:6] -> C[2:4,4:6]
        ('11', 'bot', 'bot', 4),   # A[2:4,2:4] @ B[2:4,4:6] -> C[2:4,4:6]
    ]

    for abl, bblock, cpart, col_off in products:
        for si in range(7):
            u_entries = []
            for ur, coeff in s2x2_u[si]:
                local_r, local_c = ur // 2, ur % 2
                u_entries.append((a_block_idx(abl, local_r, local_c), coeff))
            
            v_entries = []
            for vr, coeff in s2x2_v[si]:
                local_r, local_c = vr // 2, vr % 2
                if bblock == 'top':
                    v_entries.append((local_r*6 + (col_off + local_c), coeff))
                else:
                    v_entries.append(((2+local_r)*6 + (col_off + local_c), coeff))
            
            w_entries = []
            for cidx_local, sc_contribs in s2x2_w.items():
                small_coeff = 0
                for sk, sc in sc_contribs:
                    if sk == si:
                        small_coeff = sc
                        break
                if small_coeff == 0:
                    continue
                local_r, local_c = cidx_local // 2, cidx_local % 2
                if cpart == 'top':
                    w_entries.append(((col_off + local_c)*4 + local_r, small_coeff))
                else:
                    w_entries.append(((col_off + local_c)*4 + (2+local_r), small_coeff))
            
            add_term(u_entries, v_entries, w_entries)

    # Convert to numpy arrays
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    return {"U": U_np, "V": V_np, "W": W_np}
