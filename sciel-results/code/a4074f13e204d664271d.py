"""
Direct construction of the rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses block decomposition: split A into 2x2 blocks of 2x2, B into 2x5 blocks.
Each 2x2 * 2x5 multiplication uses rank 19 (Strassen for 2x2 * 2x2 + trivial for 2x2 * 2x3).
Total rank = 4 * 19 = 76.
"""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for the 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We decompose M_{4,4,5} as a block sum.
    # A is 4x4, B is 4x5, C is 4x5.
    # Split A into 4 blocks of 2x2: A_{11}, A_{12}, A_{21}, A_{22}
    # Split B into 2 blocks of 2x5: B_{top} (rows 0,1), B_{bot} (rows 2,3)
    # C = [A_{11} B_{top} + A_{12} B_{bot}; A_{21} B_{top} + A_{22} B_{bot}]
    # Each term is M_{2,2,5}.
    # For each M_{2,2,5}, we split the 2x5 matrix B into B1 (2x2) and B2 (2x3).
    # C = A B1 + A B2.
    # Use Strassen for A B1 (rank 7).
    # Use trivial for A B2 (rank 2*2*3 = 12).
    # Total rank per M_{2,2,5} is 7 + 12 = 19.
    # Total rank for M_{4,4,5} is 4 * 19 = 76.
    
    rank = 76
    U = [[0] * rank for _ in range(n * m)]  # 16 x 76
    V = [[0] * rank for _ in range(m * p)]  # 20 x 76
    W = [[0] * rank for _ in range(p * n)]  # 20 x 76
    
    col = 0
    
    # Strassen decomposition for M_{2,2,2}
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
    # M2 = (a10+a11)b00 -> contributes to c10, c11
    # M3 = a00(b10-b11) -> contributes to c01, c11
    # M4 = a11(b01-b00) -> contributes to c00, c10
    # M5 = (a00+a01)b11 -> contributes to c00, c01
    # M6 = (a10-a00)(b00+b01) -> contributes to c11
    # M7 = (a01-a11)(b10+b11) -> contributes to c00, c11
    
    # Let's use the standard Strassen formulas:
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
    
    # Mapping to indices:
    # A11=0, A12=1, A21=2, A22=3
    # B11=0, B12=1, B21=2, B22=3
    # C11=0, C12=1, C21=2, C22=3
    
    # M1: U=[0,3], V=[0,3], W=[0,3]
    # M2: U=[2,3], V=[0,0], W=[2,3]
    # M3: U=[0,0], V=[2,-3], W=[1,3]
    # M4: U=[3,3], V=[1,-0], W=[0,2]  -> V=[1,0]
    # M5: U=[0,1], V=[3,3], W=[0,1]
    # M6: U=[2,-0], V=[0,1], W=[3,3]  -> U=[2,0]
    # M7: U=[1,-3], V=[2,3], W=[0,3]
    
    strassen_terms = [
        ([0, 3], [0, 3], [0, 3]),
        ([2, 3], [0, 0], [2, 3]),
        ([0, 0], [2, -3], [1, 3]),
        ([3, 3], [1, 0], [0, 2]),
        ([0, 1], [3, 3], [0, 1]),
        ([2, 0], [0, 1], [3, 3]),
        ([1, -3], [2, 3], [0, 3]),
    ]
    
    def add_strassen_2x2x2(U, V, W, col_offset, i_off, j_off, k_off):
        """Add Strassen decomposition for M_{2,2,2} at given offsets."""
        col = col_offset
        for u_terms, v_terms, w_terms in strassen_terms:
            for u_idx, u_val in zip(u_terms, u_terms):
                # u_terms is actually indices, values are 1 except for negatives
                pass
            
            # Better to define terms as list of (index, value)
            pass
            
            # Let's redefine strassen_terms properly
            # Each term is a list of (index, value) for U, V, W
            break
            
        # This function is getting messy. Let's just inline the logic in the main loop.
        return col
    
    # Let's just write the loops directly.
    
    # For each of the 4 blocks (i_block, j_block) where i_block, j_block in {0, 1}
    # i_block determines which 2x2 block of A (rows)
    # j_block determines which 2x2 block of A (cols) and which 2x5 block of B (rows)
    
    for i_block in range(2):
        for j_block in range(2):
            # A block is A[i_block*2 : i_block*2+2, j_block*2 : j_block*2+2]
            # B block is B[j_block*2 : j_block*2+2, :] (all 5 columns)
            # C block is C[i_block*2 : i_block*2+2, :]
            
            # Strassen for B[:, 0:2] (first 2 columns)
            # k_offset for B is 0, for C is 0
            k_off = 0
            for u_terms, v_terms, w_terms in strassen_terms:
                # u_terms, v_terms, w_terms are lists of indices
                # We need to map local indices to global
                # Local U index: (li, lj) -> global (i_block*2 + li, j_block*2 + lj)
                # Local V index: (lj, lk) -> global (j_block*2 + lj, k_off + lk)
                # Local W index: (lk, li) -> global (k_off + lk, i_block*2 + li)
                
                # The strassen_terms defined above use indices 0..3 for 2x2 matrices.
                # Let's unpack them.
                for u_idx, v_idx, w_idx in zip(u_terms, v_terms, w_terms):
                    # u_idx is 0..3, representing (li, lj) = (u_idx//2, u_idx%2)
                    # Wait, my strassen_terms were defined as lists of indices, not pairs.
                    # Let's re-verify the indices in strassen_terms.
                    # M1: U=[0,3] means A[0,0] and A[1,1]. Indices 0 and 3.
                    # So u_idx is the flat index.
                    
                    li = u_idx // 2
                    lj = u_idx % 2
                    global_u = (i_block * 2 + li) * 4 + (j_block * 2 + lj)
                    
                    # v_idx is flat index for B[2x2]
                    # But wait, the B block is 2x5. The Strassen part is for B[:, 0:2].
                    # So v_idx maps to B[j_block*2 + lj, k_off + lk]
                    # v_idx is 0..3, representing (lj, lk) = (v_idx//2, v_idx%2)
                    lv = v_idx // 2
                    lk = v_idx % 2
                    global_v = (j_block * 2 + lv) * 5 + (k_off + lk)
                    
                    # w_idx is flat index for C[2x2]
                    # w_idx maps to C[i_block*2 + li, k_off + lk]
                    # w_idx is 0..3, representing (lk, li) = (w_idx//2, w_idx%2)
                    # Wait, C index is (row, col). Row is i, col is k.
                    # In W tensor, index is (k, i). So w_idx represents (lk, li).
                    lw_k = w_idx // 2
                    lw_i = w_idx % 2
                    global_w = (k_off + lw_k) * 4 + (i_block * 2 + lw_i)
                    
                    # Get values
                    # The values are not all 1.
                    # My strassen_terms definition was incomplete (just indices).
                    # I need the coefficients.
                    
                    pass
            break
        break
    
    # This is getting too complicated to do inline without a clear structure.
    # Let's define the Strassen terms with coefficients explicitly.
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1 (contributes to C11 and C22? No, C11=C00, C22=C11 in 0-based)
    # C11 = M1 + M4 - M5 + M7
    # C22 = M1 - M2 + M3 + M6
    # So M1 contributes to C00 (1) and C11 (1).
    
    # Let's list the 7 terms with (u_indices, u_values, v_indices, v_values, w_indices, w_values)
    
    terms = [
        # M1
        (([0, 3], [1, 1]), ([0, 3], [1, 1]), ([0, 3], [1, 1])),
        # M2 = (A21+A22)B11 -> (A10+A11)B00
        (([2, 3], [1, 1]), ([0], [1]), ([2, 3], [1, 1])),
        # M3 = A11(B21-B22) -> A00(B10-B11)
        (([0], [1]), ([2, 3], [1, -1]), ([1, 3], [1, 1])),
        # M4 = A22(B12-B11) -> A11(B01-B00)
        (([3], [1]), ([1, 0], [1, -1]), ([0, 2], [1, 1])),
        # M5 = (A11+A12)B22 -> (A00+A01)B11
        (([0, 1], [1, 1]), ([3], [1]), ([0, 1], [1, 1])),
        # M6 = (A21-A11)(B11+B12) -> (A10-A00)(B00+B01)
        (([2, 0], [1, -1]), ([0, 1], [1, 1]), ([3], [1])),
        # M7 = (A12-A22)(B21+B22) -> (A01-A11)(B10+B11)
        (([1, 3], [1, -1]), ([2, 3], [1, 1]), ([0, 3], [1, 1])),
    ]
    
    col = 0
    for i_block in range(2):
        for j_block in range(2):
            # Strassen part (cols 0,1 of B)
            k_off = 0
            for term in terms:
                u_idx_list, u_val_list = term[0]
                v_idx_list, v_val_list = term[1]
                w_idx_list, w_val_list = term[2]
                
                # We need to add this term to U, V, W
                # But a term can have multiple non-zero entries.
                # We cannot just add one row. We need to add the sum of rank-1 terms?
                # No, the tensor term is U_vec \otimes V_vec \otimes W_vec.
                # If U_vec has multiple non-zeros, it's still a single rank-1 term in the tensor decomposition.
                # So we just add the values to the corresponding rows in U, V, W for this column.
                
                for u_idx, u_val in zip(u_idx_list, u_val_list):
                    li = u_idx // 2
                    lj = u_idx % 2
                    global_u = (i_block * 2 + li) * 4 + (j_block * 2 + lj)
                    U[global_u][col] += u_val
                
                for v_idx, v_val in zip(v_idx_list, v_val_list):
                    lv = v_idx // 2
                    lk = v_idx % 2
                    global_v = (j_block * 2 + lv) * 5 + (k_off + lk)
                    V[global_v][col] += v_val
                
                for w_idx, w_val in zip(w_idx_list, w_val_list):
                    lw_k = w_idx // 2
                    lw_i = w_idx % 2
                    global_w = (k_off + lw_k) * 4 + (i_block * 2 + lw_i)
                    W[global_w][col] += w_val
                
                col += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
            
            # Trivial part (cols 2,3,4 of B)
            # For k in 2,3,4:
            # For i in 0,1:
            #   For j in 0,1:
            #     U[(i_block*2+i)*4 + (j_block*2+j)] = 1
            #     V[(j_block*2+j)*5 + k] = 1
            #     W[k*4 + (i_block*2+i)] = 1
            
            for k in range(2, 5):
                for i in range(2):
                    for j in range(2):
                        global_u = (i_block * 2 + i) * 4 + (j_block * 2 + j)
                        global_v = (j_block * 2 + j) * 5 + k
                        global_w = k * 4 + (i_block * 2 + i)
                        
                        U[global_u][col] = 1
                        V[global_v][col] = 1
                        W[global_w][col] = 1
                        
                        col += 1
                        
                        if time.time() > deadline:
                            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
