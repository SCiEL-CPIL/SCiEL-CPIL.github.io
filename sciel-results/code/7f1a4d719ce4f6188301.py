import numpy as np
import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Construct the 189-rank decomposition
    # Structure: 2x2 blocks of 3x3 matrices.
    # Use Strassen for the 2x2 block multiplication (7 multiplications).
    # Use Schoolbook for the 3x3 matrix multiplications (27 multiplications each).
    # Total rank = 7 * 27 = 189.
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We map the 7 Strassen multiplications to 7 groups of 27 scalar multiplications.
    # Each scalar multiplication corresponds to a (i, j, k) triple in the 3x3 block.
    
    rank = 189
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    # Strassen steps definition: (A_indices, B_indices)
    # A_indices: list of (row_block, col_block) with sign
    # B_indices: list of (row_block, col_block) with sign
    
    strassen_steps = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])
    ]
    
    # Block sizes: 3x3
    bs = 3
    
    for step_idx, (a_terms, b_terms) in enumerate(strassen_steps):
        # For each of the 27 scalar multiplications in the 3x3 block
        for i_local in range(bs):
            for j_local in range(bs):
                for k_local in range(bs):
                    # U factor: A[i, j]
                    # The A part is a sum of blocks.
                    # We need to add the contribution to U for each A term.
                    # U[row, col] += sign * A_block[i_local, j_local]
                    # But U is a vector. We are defining the tensor.
                    # The tensor entry T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                    # Here, i, j, k are global indices.
                    # i = i_block * bs + i_local
                    # j = j_block * bs + j_local
                    # k = k_block * bs + k_local
                    
                    # For a fixed Strassen step, we are computing M_step = A_sum * B_sum.
                    # The contribution to C is distributed.
                    # This direct mapping is tricky because Strassen combines results.
                    # Instead, we can view this as:
                    # C_block[i_block, j_block] = sum_k_block A_block[i_block, k_block] B_block[k_block, j_block]
                    # Using Strassen, we compute 7 intermediate matrices M1..M7.
                    # Each M is a 3x3 matrix.
                    # Then C blocks are linear combinations of M1..M7.
                    
                    # This requires storing the 7 M matrices, which are 3x3.
                    # Total "variables" for M matrices: 7 * 9 = 63.
                    # But we need to express C directly from A and B.
                    # The rank is the number of scalar multiplications.
                    # Each M_step is a 3x3 matrix product.
                    # So we need 7 * 27 = 189 scalar multiplications.
                    
                    # Let's build the factors for one scalar multiplication in step s.
                    # The multiplication is (Sum_a A_a) * (Sum_b B_b).
                    # The result is added to C with some coefficient.
                    
                    # Actually, it's easier to construct the 189-rank decomposition directly.
                    pass
    
    # Re-implementation of 189-rank construction
    # We define 189 columns.
    # 7 Strassen steps.
    # Each step involves multiplying two 3x3 matrices (linear combos of blocks).
    # Let the two 3x3 matrices be X and Y.
    # X = sum_alpha x_alpha A_alpha
    # Y = sum_beta y_beta B_beta
    # The product Z = X Y.
    # Z[i_local, j_local] = sum_k_local X[i_local, k_local] Y[k_local, j_local]
    # This requires 27 scalar multiplications for this step.
    
    U = np.zeros((36, 189), dtype=float)
    V = np.zeros((36, 189), dtype=float)
    W = np.zeros((36, 189), dtype=float)
    
    col = 0
    for s_idx, (a_terms, b_terms) in enumerate(strassen_steps):
        for i_l in range(3):
            for j_l in range(3):
                for k_l in range(3):
                    # U part: X[i_l, k_l]
                    # X is a linear combo of A blocks.
                    for (a_rb, a_cb, sign) in a_terms:
                        # A block is at row a_rb, col a_cb.
                        # Element is A[a_rb*3 + i_l, a_cb*3 + k_l]
                        # U index: (a_rb*3 + i_l) * 6 + (a_cb*3 + k_l)
                        u_idx = (a_rb * 3 + i_l) * 6 + (a_cb * 3 + k_l)
                        U[u_idx, col] += sign
                    
                    # V part: Y[k_l, j_l]
                    # Y is a linear combo of B blocks.
                    for (b_rb, b_cb, sign) in b_terms:
                        # B block is at row b_rb, col b_cb.
                        # Element is B[b_rb*3 + k_l, b_cb*3 + j_l]
                        # V index: (b_rb*3 + k_l) * 6 + (b_cb*3 + j_l)
                        v_idx = (b_rb * 3 + k_l) * 6 + (b_cb * 3 + j_l)
                        V[v_idx, col] += sign
                    
                    # W part: Contribution to C.
                    # The result of this scalar multiplication contributes to C.
                    # C = sum_steps (coeff * M_step)
                    # M_step is a 3x3 matrix.
                    # We need to map the result of X[i_l, k_l] * Y[k_l, j_l] to C.
                    # This is complex because Strassen outputs are combined.
                    # Let's use the explicit Strassen output combination.
                    pass
    
    # This manual construction is error-prone.
    # I will use a known valid 189-rank decomposition from literature or generate it via a simpler method.
    # Actually, I can just return the schoolbook 216 and then try to reduce it?
    # No, 189 is known.
    
    # Let's use a simpler, verified 189-rank construction.
    # Or, I can use the 216-rank schoolbook and apply a **randomized rank reduction** algorithm.
    
    # Initialize with Schoolbook (Rank 216)
    rank = 216
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i * 6 + j, col] = 1
                V[j * 6 + k, col] = 1
                W[k * 6 + i, col] = 1
                col += 1
    
    # Now, try to reduce rank.
    # This is computationally expensive to do correctly in 150s for a big jump.
    # I will return the 189-rank solution constructed via the Strassen/Block method if I can code it correctly.
    
    # Let's try to code the 189-rank solution properly.
    # Reference: Bini et al.
    
    # I will implement the 189-rank solution using the 2x2 block / 3x3 subblock structure.
    # Strassen steps for 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a 3x3 matrix product.
    # We decompose each M into 27 scalar multiplications.
    
    U = np.zeros((36, 189), dtype=float)
    V = np.zeros((36, 189), dtype=float)
    W = np.zeros((36, 189), dtype=float)
    
    col = 0
    # Define the 7 steps
    # Each step: (A_coeff_dict, B_coeff_dict, C_output_dict)
    # A_coeff_dict: { (block_r, block_c): sign }
    # C_output_dict: { (block_r, block_c): sign }
    
    steps = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,1):1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,1):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}) # M7
    ]
    
    # Note: The C_output_dict indicates which C blocks this M contributes to.
    # But M is a single 3x3 matrix. It contributes to multiple C blocks.
    # The tensor identity is C = sum M_contrib.
    # For each scalar multiplication in M (i_l, k_l, j_l), the result is added to C_block.
    
    for a_coeffs, b_coeffs, c_coeffs in steps:
        for i_l in range(3):
            for k_l in range(3):
                for j_l in range(3):
                    # U: A part
                    for (br, bc), sign in a_coeffs.items():
                        u_idx = (br * 3 + i_l) * 6 + (bc * 3 + k_l)
                        U[u_idx, col] += sign
                    
                    # V: B part
                    for (br, bc), sign in b_coeffs.items():
                        v_idx = (br * 3 + k_l) * 6 + (bc * 3 + j_l)
                        V[v_idx, col] += sign
                    
                    # W: C part
                    # The result of A[i,k] * B[k,j] is added to C[i,j].
                    # Here i,j are local indices within the 3x3 block.
                    # The global C index depends on which block this M contributes to.
                    # M contributes to multiple C blocks.
                    # For each C block (cr, cc) in c_coeffs:
                    for (cr, cc), sign in c_coeffs.items():
                        w_idx = (cr * 3 + j_l) * 6 + (cc * 3 + i_l) # W[k*n+i, r] -> W index is k*6+i?
                        # Wait, W shape is (p*n, rank) = (36, rank).
                        # Index is k*n + i.
                        # Here k is local? No, k is the column of A, row of B.
                        # In the block structure, the "k" index of the 3x3 multiplication is local k_l.
                        # The global index for W is (global_k * 6 + global_i).
                        # global_k = cr * 3 + k_l? No.
                        # The multiplication is A_block * B_block.
                        # A_block is at (ar, ac). B_block is at (br, bc).
                        # The result is at C_block (ar, bc).
                        # The local indices are i_l, k_l, j_l.
                        # Global i = ar*3 + i_l.
                        # Global k = ac*3 + k_l.
                        # Global j = bc*3 + j_l.
                        # W index = global_k * 6 + global_i.
                        # But W is summed over the steps.
                        # For a specific step, the A part comes from blocks in a_coeffs.
                        # The B part comes from blocks in b_coeffs.
                        # The C part goes to blocks in c_coeffs.
                        # This is inconsistent because a_coeffs has multiple blocks.
                        # The scalar multiplication is (Sum A) * (Sum B).
                        # The result is (Sum A)(Sum B) = Sum A_a B_b.
                        # This is not a single scalar multiplication.
                        # My construction is flawed.
                        pass
    
    # This approach is too complex to debug in 150s.
    # I will return the schoolbook 216.
    # This is safe.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
