"""Matrix multiplication 4,4,4 — rank-49 recursive Strassen decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # 2x2 Strassen terms: each is (u_vec, v_vec, w_vec) of length 4
    # u_vec corresponds to [x00, x01, x10, x11] of the left 2x2 matrix
    # v_vec corresponds to [y00, y01, y10, y11] of the right 2x2 matrix
    # w_vec corresponds to [z00, z01, z10, z11] of the result 2x2 matrix
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # t1: (x00+x11)(y00+y11)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # t2: (x10+x11)y00
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # t3: x00(y01-y11)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # t4: x11(y10-y00)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # t5: (x00+x01)y11
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # t6: (x10-x00)(y00+y01)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # t7: (x01-x11)(y10+y11)
    ]
    
    # Define the 7 block-level Strassen terms.
    # Each block term b: X_b * Y_b contributes to C with certain signs.
    # X_b is a 2x2 matrix whose entries are linear combinations of A elements.
    # Y_b is a 2x2 matrix whose entries are linear combinations of B elements.
    # The result Z_b = X_b * Y_b is a 2x2 matrix of C-blocks.
    
    # We represent each 2x2 matrix as a 2x2 list of lists of (element_idx, coeff).
    # For A: element_idx = i*4 + j for A[i,j]
    # For B: element_idx = j*4 + k for B[j,k]
    
    def block_X_Y(block_idx):
        """Return (X, Y) where X and Y are 2x2 lists of (element_idx, coeff) lists."""
        if block_idx == 0:  # M1: X=A11+A22, Y=B11+B22
            X = [[[(0,1),(10,1)], [(1,1),(11,1)]],
                 [[(4,1),(14,1)], [(5,1),(15,1)]]]
            Y = [[[(0,1),(10,1)], [(1,1),(11,1)]],
                 [[(4,1),(14,1)], [(5,1),(15,1)]]]
        elif block_idx == 1:  # M2: X=A21+A22, Y=B11
            X = [[[(8,1),(10,1)], [(9,1),(11,1)]],
                 [[(12,1),(14,1)], [(13,1),(15,1)]]]
            Y = [[[(0,1)], [(1,1)]],
                 [[(4,1)], [(5,1)]]]
        elif block_idx == 2:  # M3: X=A11, Y=B12-B22
            X = [[[(0,1)], [(1,1)]],
                 [[(4,1)], [(5,1)]]]
            Y = [[[(2,1),(10,-1)], [(3,1),(11,-1)]],
                 [[(6,1),(14,-1)], [(7,1),(15,-1)]]]
        elif block_idx == 3:  # M4: X=A22, Y=B21-B11
            X = [[[(10,1)], [(11,1)]],
                 [[(14,1)], [(15,1)]]]
            Y = [[[(8,1),(0,-1)], [(9,1),(1,-1)]],
                 [[(12,1),(4,-1)], [(13,1),(5,-1)]]]
        elif block_idx == 4:  # M5: X=A11+A12, Y=B22
            X = [[[(0,1),(2,1)], [(1,1),(3,1)]],
                 [[(4,1),(6,1)], [(5,1),(7,1)]]]
            Y = [[[(10,1)], [(11,1)]],
                 [[(14,1)], [(15,1)]]]
        elif block_idx == 5:  # M6: X=A21-A11, Y=B11+B12
            X = [[[(8,1),(0,-1)], [(9,1),(1,-1)]],
                 [[(12,1),(4,-1)], [(13,1),(5,-1)]]]
            Y = [[[(0,1),(2,1)], [(1,1),(3,1)]],
                 [[(4,1),(6,1)], [(5,1),(7,1)]]]
        else:  # block_idx == 6: M7: X=A12-A22, Y=B21+B22
            X = [[[(2,1),(10,-1)], [(3,1),(11,-1)]],
                 [[(6,1),(14,-1)], [(7,1),(15,-1)]]]
            Y = [[[(8,1),(10,1)], [(9,1),(11,1)]],
                 [[(12,1),(14,1)], [(13,1),(15,1)]]]
        return X, Y
    
    # W mapping: for each block term, which C-blocks get which coefficient.
    # C-block indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1) in block coords.
    # Each C-block is a 2x2 submatrix of C.
    # C[i,k] where i,k in 0..3. Block (bi,bk) covers C[2*bi:2*bi+2, 2*bk:2*bk+2].
    # W row for C[i,k] is k*4+i.
    
    def w_block_coeffs(block_idx):
        """Return dict: (block_row, block_col) -> coeff for the 2x2 block result."""
        if block_idx == 0:  # M1: C11 += 1, C22 += 1
            return {(0,0): 1, (1,1): 1}
        elif block_idx == 1:  # M2: C21 += 1, C22 += -1
            return {(1,0): 1, (1,1): -1}
        elif block_idx == 2:  # M3: C12 += 1, C22 += 1
            return {(0,1): 1, (1,1): 1}
        elif block_idx == 3:  # M4: C11 += 1, C21 += 1
            return {(0,0): 1, (1,0): 1}
        elif block_idx == 4:  # M5: C11 += -1, C12 += 1
            return {(0,0): -1, (0,1): 1}
        elif block_idx == 5:  # M6: C22 += 1
            return {(1,1): 1}
        else:  # block_idx == 6: M7: C11 += 1
            return {(0,0): 1}
    
    # For each block term, apply 2x2 Strassen.
    for block_idx in range(7):
        X, Y = block_X_Y(block_idx)
        w_coeffs = w_block_coeffs(block_idx)
        base_col = block_idx * 7
        
        for s in range(7):
            col = base_col + s
            u_vec, v_vec, w_vec = strassen_2x2[s]
            
            # u_vec[j] gives coefficient for X element j (j: 0=x00, 1=x01, 2=x10, 3=x11)
            # v_vec[j] gives coefficient for Y element j
            # w_vec[j] gives coefficient for Z element j (j: 0=z00, 1=z01, 2=z10, 3=z11)
            
            # Build U column: for each X element position, multiply by u_vec coeff
            # and expand into original A elements.
            # X is 2x2: X[0][0]=x00, X[0][1]=x01, X[1][0]=x10, X[1][1]=x11
            x_positions = [[X[0][0], X[0][1]], [X[1][0], X[1][1]]]
            y_positions = [[Y[0][0], Y[0][1]], [Y[1][0], Y[1][1]]]
            
            # For U: accumulate contributions from all x positions with nonzero u_vec
            for pos in range(4):
                u_coeff = u_vec[pos]
                if u_coeff == 0:
                    continue
                # pos 0 -> x00, pos 1 -> x01, pos 2 -> x10, pos 3 -> x11
                if pos == 0:
                    x_elem = x_positions[0][0]
                elif pos == 1:
                    x_elem = x_positions[0][1]
                elif pos == 2:
                    x_elem = x_positions[1][0]
                else:
                    x_elem = x_positions[1][1]
                for (elem_idx, coeff) in x_elem:
                    U[elem_idx, col] += u_coeff * coeff
            
            # For V: accumulate contributions from all y positions with nonzero v_vec
            for pos in range(4):
                v_coeff = v_vec[pos]
                if v_coeff == 0:
                    continue
                if pos == 0:
                    y_elem = y_positions[0][0]
                elif pos == 1:
                    y_elem = y_positions[0][1]
                elif pos == 2:
                    y_elem = y_positions[1][0]
                else:
                    y_elem = y_positions[1][1]
                for (elem_idx, coeff) in y_elem:
                    V[elem_idx, col] += v_coeff * coeff
            
            # For W: the result Z_b = X_b * Y_b is a 2x2 block matrix.
            # Z_b[br][bc] contributes to C-block (br, bc) with coefficient w_vec[j] * w_coeffs[(br,bc)]
            # where j maps to (br, bc): j=0->(0,0), j=1->(0,1), j=2->(1,0), j=3->(1,1)
            for j in range(4):
                w_coeff_inner = w_vec[j]
                if w_coeff_inner == 0:
                    continue
                br = j // 2
                bc = j % 2
                if (br, bc) not in w_coeffs:
                    continue
                w_coeff_outer = w_coeffs[(br, bc)]
                total_w_coeff = w_coeff_inner * w_coeff_outer
                if total_w_coeff == 0:
                    continue
                # C-block (br, bc) covers C[2*br:2*br+2, 2*bc:2*bc+2]
                # W row for C[i,k] is k*4+i
                for di in range(2):
                    for dk in range(2):
                        i = 2*br + di
                        k = 2*bc + dk
                        w_row = k*4 + i
                        W[w_row, col] += total_w_coeff
    
    # Convert to lists of integers (or half-integers, but all entries here are integers)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure all values are exactly representable as int or half-int
    for mat in (U_list, V_list, W_list):
        for row in mat:
            for val in row:
                if abs(val - round(val)) > 1e-10:
                    # Should not happen with this construction
                    pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
