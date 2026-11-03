import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = start_time + 100.0
    
    # We'll build the decomposition and verify it
    # C (4x6) = A (4x5) * B (5x6)
    
    # Split: A = [A1 (4x4), A2 (4x1)], B = [B1 (4x6); B2 (1x6)]
    # B1 = [B1a (4x4), B1b (4x2)]
    # C = A1*B1a + A1*B1b + A2*B2
    
    # Part 1: A1 (4x4) * B1a (4x4) using Strassen
    # Part 2: A1 (4x4) * B1b (4x2) using schoolbook
    # Part 3: A2 (4x1) * B2 (1x6) using schoolbook
    
    # Collect all rank-1 terms as (u_row, v_row, w_row, coeff)
    terms = []
    
    # Part 1: Strassen for 4x4*4x4
    # A1[i,j] -> U row i*5+j for i,j in 0..3
    # B1a[j,k] -> V row j*6+k for j,k in 0..3
    # C[i,k] -> W row k*4+i for i,k in 0..3
    
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For each product, left and right are 2x2 matrices that are linear combos of blocks
    # A00: A[i,j] for i,j in 0,1 -> U rows: 0*5+0, 0*5+1, 1*5+0, 1*5+1 = 0,1,5,6
    # A01: A[i,j+2] for i,j in 0,1 -> U rows: 0*5+2, 0*5+3, 1*5+2, 1*5+3 = 2,3,7,8
    # A10: A[i+2,j] for i,j in 0,1 -> U rows: 2*5+0, 2*5+1, 3*5+0, 3*5+1 = 10,11,15,16
    # A11: A[i+2,j+2] for i,j in 0,1 -> U rows: 2*5+2, 2*5+3, 3*5+2, 3*5+3 = 12,13,17,18
    
    # B00: B[j,k] for j,k in 0,1 -> V rows: 0*6+0, 0*6+1, 1*6+0, 1*6+1 = 0,1,6,7
    # B01: B[j,k+2] for j,k in 0,1 -> V rows: 0*6+2, 0*6+3, 1*6+2, 1*6+3 = 2,3,8,9
    # B10: B[j+2,k] for j,k in 0,1 -> V rows: 2*6+0, 2*6+1, 3*6+0, 3*6+1 = 12,13,18,19
    # B11: B[j+2,k+2] for j,k in 0,1 -> V rows: 2*6+2, 2*6+3, 3*6+2, 3*6+3 = 14,15,20,21
    
    # C00: C[i,k] for i,k in 0,1 -> W rows: 0*4+0, 0*4+1, 1*4+0, 1*4+1 = 0,1,4,5
    # C01: C[i,k+2] for i,k in 0,1 -> W rows: 2*4+0, 2*4+1, 3*4+0, 3*4+1 = 8,9,12,13
    # C10: C[i+2,k] for i,k in 0,1 -> W rows: 0*4+2, 0*4+3, 1*4+2, 1*4+3 = 2,3,6,7
    # C11: C[i+2,k+2] for i,k in 0,1 -> W rows: 2*4+2, 2*4+3, 3*4+2, 3*4+3 = 10,11,14,15
    
    def add_product(left_blocks, right_blocks, w_blocks):
        """
        left_blocks: list of (block_id, coeff) where block_id is 'A00','A01','A10','A11'
        right_blocks: list of (block_id, coeff) where block_id is 'B00','B01','B10','B11'
        w_blocks: list of (block_id, coeff) where block_id is 'C00','C01','C10','C11'
        """
        # Get the U, V, W row indices for each block
        def get_u_rows(block):
            if block == 'A00':
                return [(0,0), (0,1), (1,0), (1,1)]
            elif block == 'A01':
                return [(0,2), (0,3), (1,2), (1,3)]
            elif block == 'A10':
                return [(2,0), (2,1), (3,0), (3,1)]
            elif block == 'A11':
                return [(2,2), (2,3), (3,2), (3,3)]
        
        def get_v_rows(block):
            if block == 'B00':
                return [(0,0), (0,1), (1,0), (1,1)]
            elif block == 'B01':
                return [(0,2), (0,3), (1,2), (1,3)]
            elif block == 'B10':
                return [(2,0), (2,1), (3,0), (3,1)]
            elif block == 'B11':
                return [(2,2), (2,3), (3,2), (3,3)]
        
        def get_w_rows(block):
            if block == 'C00':
                return [(0,0), (0,1), (1,0), (1,1)]
            elif block == 'C01':
                return [(0,2), (0,3), (1,2), (1,3)]
            elif block == 'C10':
                return [(2,0), (2,1), (3,0), (3,1)]
            elif block == 'C11':
                return [(2,2), (2,3), (3,2), (3,3)]
        
        # For each combination of left block entry, right block entry, and output block entry
        # The product L[i,k]*R[k,j] where L[i,k] comes from left block and R[k,j] from right block
        # contributes to P[i,j] which maps to the output block
        
        # For a 2x2*2x2 product, we have 4 output entries (i,j) in {0,1}x{0,1}
        # Each output entry P[i,j] = sum_{k=0,1} L[i,k]*R[k,j]
        
        # L[i,k] is a linear combination of A entries from the left blocks
        # R[k,j] is a linear combination of B entries from the right blocks
        
        # For each (i,k,j) and each left block entry and right block entry:
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    # L[i,k] from left blocks
                    for (lb, lc) in left_blocks:
                        if lc == 0:
                            continue
                        # Get the (row, col) in the 2x2 block for position (i,k)
                        # The block is 2x2, so position (i,k) in the block
                        lb_rows = get_u_rows(lb)
                        # lb_rows is a list of 4 (global_row, global_col) for the 2x2 block
                        # Position (i,k) in the 2x2 block: index = i*2 + k
                        u_idx = i * 2 + k
                        u_row_idx, u_col_idx = lb_rows[u_idx]
                        u_row = u_row_idx * m + u_col_idx
                        
                        for (rb, rc) in right_blocks:
                            if rc == 0:
                                continue
                            rb_rows = get_v_rows(rb)
                            # Position (k,j) in the 2x2 block: index = k*2 + j
                            v_idx = k * 2 + j
                            v_row_idx, v_col_idx = rb_rows[v_idx]
                            v_row = v_row_idx * p + v_col_idx
                            
                            coeff = lc * rc
                            if coeff == 0:
                                continue
                            
                            # This contributes to P[i,j] in the 2x2 product
                            # P[i,j] maps to output blocks via w_blocks
                            for (wb, wc) in w_blocks:
                                if wc == 0:
                                    continue
                                wb_rows = get_w_rows(wb)
                                # Position (i,j) in the 2x2 block: index = i*2 + j
                                w_idx = i * 2 + j
                                w_row_idx, w_col_idx = wb_rows[w_idx]
                                w_row = w_row_idx * n + w_col_idx
                                
                                total_coeff = coeff * wc
                                if total_coeff != 0:
                                    terms.append((u_row, v_row, w_row, total_coeff))
    
    # M1 = (A00+A11)(B00+B11), contributes to C00 and C11
    add_product([('A00', 1), ('A11', 1)], [('B00', 1), ('B11', 1)], [('C00', 1), ('C11', 1)])
    
    # M2 = (A10+A11)B00, contributes to C10
    add_product([('A10', 1), ('A11', 1)], [('B00', 1)], [('C10', 1)])
    
    # M3 = A00(B10-B11), contributes to C01
    add_product([('A00', 1)], [('B10', 1), ('B11', -1)], [('C01', 1)])
    
    # M4 = A11(B01-B10), contributes to C00 and C11
    add_product([('A11', 1)], [('B01', 1), ('B10', -1)], [('C00', 1), ('C11', 1)])
    
    # M5 = (A00+A01)B11, contributes to -C00 and C01
    add_product([('A00', 1), ('A01', 1)], [('B11', 1)], [('C00', -1), ('C01', 1)])
    
    # M6 = (A10-A00)(B00+B01), not used in C
    add_product([('A10', 1), ('A00', -1)], [('B00', 1), ('B01', 1)], [])
    
    # M7 = (A01-A11)(B10+B11), contributes to -C00 and C11
    add_product([('A01', 1), ('A11', -1)], [('B10', 1), ('B11', 1)], [('C00', -1), ('C11', 1)])
    
    # Part 2: A1 (4x4) * B1b (4x2) schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                terms.append((u_row, v_row, w_row, 1.0))
    
    # Part 3: A2 (4x1) * B2 (1x6) schoolbook
    for i in range(4):
        for k in range(6):
            u_row = i * m + 4
            v_row = 4 * p + k
            w_row = k * n + i
            terms.append((u_row, v_row, w_row, 1.0))
    
    # Combine terms with same (u_row, v_row, w_row)
    from collections import defaultdict
    combined = defaultdict(float)
    for (u_row, v_row, w_row, coeff) in terms:
        key = (u_row, v_row, w_row)
        combined[key] += coeff
    
    # Filter out zero coefficients
    final_terms = [(u, v, w, c) for (u, v, w), c in combined.items() if abs(c) > 1e-10]
    
    rank = len(final_terms)
    print(f"Rank: {rank}")
    
    # Build U, V, W matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col, (u_row, v_row, w_row, coeff) in enumerate(final_terms):
        U[u_row, col] = coeff
        V[v_row, col] = coeff
        W[w_row, col] = 1.0
    
    # Verify the decomposition
    def verify(U, V, W):
        # C = sum_t U_t * V_t * W_t (outer product)
        # C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for all j
        # Actually, C[i,k] = sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # But W doesn't depend on j, so:
        # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * V[j*p+k, t])
        
        C = np.zeros((n, p))
        for t in range(rank):
            # For each t, compute the contribution
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += U[i * m + j, t] * V[j * p + k, t]
                    C[i, k] += W[k * n + i, t] * s
        
        # Compare with expected: C[i,k] should be sum_j A[i,j]*B[j,k]
        # But we don't have A and B, so we just check that the decomposition is consistent
        # Actually, we need to verify that the tensor is correct
        # The tensor T[i,j,k] should be 1 for all i,j,k
        # In the decomposition: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
        
        # Check a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    if verify(U, V, W):
        print("Verification passed!")
    else:
        print("Verification failed!")
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
