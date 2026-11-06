"""Matrix multiplication 6x6x6 tensor decomposition.

Strategy:
  - Outer level: 2x2 Strassen on 3x3 blocks (7 products).
  - Inner level: rank-23 Laderman for 3x3 matrix multiplication.
  Total rank = 7 * 23 = 161.

  Verified at runtime with random tests; falls back to rank-189
  (2x2 Strassen on 2x2 blocks with schoolbook 3x3) if the Laderman
  coefficients are found invalid, and to rank-216 schoolbook as a last resort.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_rank23_laderman():
    """Return the 23-term Laderman decomposition for 3x3 matrix multiplication.
    Each term is (x_coeffs, y_coeffs, z_coeffs) where each is a list of
    ((row, col), coeff) tuples.
    """
    laderman = [
        # 1: (x11+x22+x33)(y11+y22+y33) -> z11+z22+z33
        ([((0,0),1),((1,1),1),((2,2),1)],
         [((0,0),1),((1,1),1),((2,2),1)],
         [((0,0),1),((1,1),1),((2,2),1)]),
        # 2: (x11+x12+x13)(y21+y22+y23) -> z12+z13
        ([((0,0),1),((0,1),1),((0,2),1)],
         [((1,0),1),((1,1),1),((1,2),1)],
         [((0,1),1),((0,2),1)]),
        # 3: (x11+x21+x31)(y12+y13+y23) -> z21+z31
        ([((0,0),1),((1,0),1),((2,0),1)],
         [((0,1),1),((0,2),1),((1,2),1)],
         [((1,0),1),((2,0),1)]),
        # 4: (x11+x22+x33)(y12+y23+y31) -> z12+z23+z31
        ([((0,0),1),((1,1),1),((2,2),1)],
         [((0,1),1),((1,2),1),((2,0),1)],
         [((0,1),1),((1,2),1),((2,0),1)]),
        # 5: (x11+x22+x33)(y13+y21+y32) -> z13+z21+z32
        ([((0,0),1),((1,1),1),((2,2),1)],
         [((0,2),1),((1,0),1),((2,1),1)],
         [((0,2),1),((1,0),1),((2,1),1)]),
        # 6: (x11+x12+x13)(y11+y21+y31) -> z11+z12
        ([((0,0),1),((0,1),1),((0,2),1)],
         [((0,0),1),((1,0),1),((2,0),1)],
         [((0,0),1),((0,1),1)]),
        # 7: (x11+x21+x31)(y11+y12+y22) -> z21+z22
        ([((0,0),1),((1,0),1),((2,0),1)],
         [((0,0),1),((0,1),1),((1,1),1)],
         [((1,0),1),((1,1),1)]),
        # 8: (x11+x22+x33)(y11+y12+y13) -> z11+z12+z13
        ([((0,0),1),((1,1),1),((2,2),1)],
         [((0,0),1),((0,1),1),((0,2),1)],
         [((0,0),1),((0,1),1),((0,2),1)]),
        # 9: (x11+x21+x31)(y11+y21+y22) -> z21+z22+z23
        ([((0,0),1),((1,0),1),((2,0),1)],
         [((0,0),1),((1,0),1),((1,1),1)],
         [((1,0),1),((1,1),1),((1,2),1)]),
        # 10: (x11+x22+x33)(y11+y21+y31) -> z11+z21+z31
        ([((0,0),1),((1,1),1),((2,2),1)],
         [((0,0),1),((1,0),1),((2,0),1)],
         [((0,0),1),((1,0),1),((2,0),1)]),
        # 11: (x11+x12+x13)(y11+y12+y13) -> z11+z12+z13
        ([((0,0),1),((0,1),1),((0,2),1)],
         [((0,0),1),((0,1),1),((0,2),1)],
         [((0,0),1),((0,1),1),((0,2),1)]),
        # 12: (x21+x22+x23)(y11+y12+y13) -> z21+z22+z23
        ([((1,0),1),((1,1),1),((1,2),1)],
         [((0,0),1),((0,1),1),((0,2),1)],
         [((1,0),1),((1,1),1),((1,2),1)]),
        # 13: (x31+x32+x33)(y11+y12+y13) -> z31+z32+z33
        ([((2,0),1),((2,1),1),((2,2),1)],
         [((0,0),1),((0,1),1),((0,2),1)],
         [((2,0),1),((2,1),1),((2,2),1)]),
        # 14: (x11+x21+x31)(y21+y22+y23) -> z21+z22+z23
        ([((0,0),1),((1,0),1),((2,0),1)],
         [((1,0),1),((1,1),1),((1,2),1)],
         [((1,0),1),((1,1),1),((1,2),1)]),
        # 15: (x12+x22+x32)(y21+y22+y23) -> z22+z23+z32
        ([((0,1),1),((1,1),1),((2,1),1)],
         [((1,0),1),((1,1),1),((1,2),1)],
         [((1,1),1),((1,2),1),((2,1),1)]),
        # 16: (x13+x23+x33)(y21+y22+y23) -> z23+z33+z32
        ([((0,2),1),((1,2),1),((2,2),1)],
         [((1,0),1),((1,1),1),((1,2),1)],
         [((1,2),1),((2,2),1),((2,1),1)]),
        # 17: (x11+x21+x31)(y31+y32+y33) -> z31+z32+z33
        ([((0,0),1),((1,0),1),((2,0),1)],
         [((2,0),1),((2,1),1),((2,2),1)],
         [((2,0),1),((2,1),1),((2,2),1)]),
        # 18: (x12+x22+x32)(y31+y32+y33) -> z32+z33+z12
        ([((0,1),1),((1,1),1),((2,1),1)],
         [((2,0),1),((2,1),1),((2,2),1)],
         [((2,1),1),((2,2),1),((0,1),1)]),
        # 19: (x13+x23+x33)(y31+y32+y33) -> z33+z13+z23
        ([((0,2),1),((1,2),1),((2,2),1)],
         [((2,0),1),((2,1),1),((2,2),1)],
         [((2,2),1),((0,2),1),((1,2),1)]),
        # 20: (x11+x12+x13)(y31+y32+y33) -> z13+z23+z33
        ([((0,0),1),((0,1),1),((0,2),1)],
         [((2,0),1),((2,1),1),((2,2),1)],
         [((0,2),1),((1,2),1),((2,2),1)]),
        # 21: (x21+x22+x23)(y31+y32+y33) -> z23+z33+z13
        ([((1,0),1),((1,1),1),((1,2),1)],
         [((2,0),1),((2,1),1),((2,2),1)],
         [((1,2),1),((2,2),1),((0,2),1)]),
        # 22: (x31+x32+x33)(y31+y32+y33) -> z33+z13+z23
        ([((2,0),1),((2,1),1),((2,2),1)],
         [((2,0),1),((2,1),1),((2,2),1)],
         [((2,2),1),((0,2),1),((1,2),1)]),
        # 23: (x11+x22+x33)(y21+y32+y13) -> z12+z23+z31
        ([((0,0),1),((1,1),1),((2,2),1)],
         [((1,0),1),((2,1),1),((0,2),1)],
         [((0,1),1),((1,2),1),((2,0),1)]),
    ]
    return laderman


def _verify_3x3_decomp(decomp, n_tests=200, seed=42):
    """Verify that a 3x3 decomposition correctly computes matrix multiplication."""
    rng = np.random.RandomState(seed)
    for _ in range(n_tests):
        X = rng.randn(3, 3)
        Y = rng.randn(3, 3)
        Z_expected = X @ Y
        Z_computed = np.zeros((3, 3))
        for x_coeffs, y_coeffs, z_coeffs in decomp:
            x_val = sum(X[r, c] * coeff for (r, c), coeff in x_coeffs)
            y_val = sum(Y[r, c] * coeff for (r, c), coeff in y_coeffs)
            for (r, c), coeff in z_coeffs:
                Z_computed[r, c] += x_val * y_val * coeff
        if not np.allclose(Z_computed, Z_expected, atol=1e-10):
            return False
    return True


def _build_6x6_strassen_laderman(n, m, p):
    """Build rank-161 (7*23) decomposition using 2x2 Strassen outer + Laderman inner."""
    laderman = _build_rank23_laderman()
    rank = 7 * len(laderman)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 terms for 2x2 block multiplication
    # A offsets: 0=A11, 1=A12, 2=A21, 3=A22  (as (row,col) in block)
    # B offsets: 0=B11, 1=B12, 2=B21, 3=B22
    strassen_terms = [
        # (A_comb, B_comb) where each is list of ((br,bc), coeff)
        # M1 = (A11+A22)(B11+B22)
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        # M2 = (A21+A22)B11
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        # M3 = A11(B12-B22)
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        # M4 = A22(B21-B11)
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        # M5 = (A11+A12)B22
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        # M6 = (A21-A11)(B11+B12)
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        # M7 = (A12-A22)(B21+B22)
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    
    # C contributions: for each Strassen term, list of ((cr,cc), coeff)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    c_contributions = [
        [((0,0),1),((1,1),1)],    # M1
        [((1,0),1),((1,1),-1)],   # M2
        [((0,1),1),((1,1),1)],    # M3
        [((0,0),1),((1,0),1)],    # M4
        [((0,0),-1),((0,1),1)],   # M5
        [((1,1),1)],              # M6
        [((0,0),1)],              # M7
    ]
    
    col = 0
    for s in range(7):
        a_comb, b_comb = strassen_terms[s]
        c_comb = c_contributions[s]
        
        for l in laderman:
            x_coeffs, y_coeffs, z_coeffs = l
            
            # U: A element at global position (block_row*3 + lr, block_col*3 + lc)
            # U index = (global_row) * m + (global_col) = (br*3+lr)*6 + (bc*3+lc)
            for (a_br, a_bc), a_coeff in a_comb:
                for (lr, lc), l_coeff in x_coeffs:
                    gr = a_br * 3 + lr
                    gc = a_bc * 3 + lc
                    U[gr * m + gc][col] += a_coeff * l_coeff
            
            # V: B element at global position (block_row*3 + lr, block_col*3 + lc)
            # V index = (global_col_of_A) * p + (global_row_of_B) = (bc*3+lc)*6 + (br*3+lr)
            # Wait: V is (m*p, rank). V[j*p + k] where j is col of A, k is row of B.
            # For block (bi, bj) in A and (bk, bj) in B:
            # Actually in Strassen outer, we have A block at (bi, bk) and B block at (bk, bj)
            # where bi is block row of A, bk is block col of A = block row of B, bj is block col of B.
            # A element (br, bc) in block (bi, bk) -> global (bi*3+br, bk*3+bc)
            # B element (br, bc) in block (bk, bj) -> global (bk*3+br, bj*3+bc)
            # V index = (global_col_A) * p + (global_row_B) = (bk*3+bc) * 6 + (bk*3+br)
            # Hmm, that's not right. Let me reconsider.
            # V[j*p + k] where j is the column index in A (which equals row index in B for the product),
            # and k is the row index in B.
            # Actually for matrix multiplication C = A * B:
            # C[i,j] = sum_k A[i,k] * B[k,j]
            # U[i*m + k], V[k*p + j], W[j*n + i]
            # So U row = i*m + k where i is row of A, k is col of A = row of B
            # V row = k*p + j where k is row of B, j is col of B
            # W row = j*n + i where j is col of B, i is row of A
            
            # In block terms:
            # A block (bi, bk): A[i,k] where i = bi*3+li, k = bk*3+lk
            # B block (bk, bj): B[k,j] where k = bk*3+lk, j = bj*3+lj
            # C block (bi, bj): C[i,j] where i = bi*3+li, j = bj*3+lj
            
            # U row = i*m + k = (bi*3+li)*6 + (bk*3+lk)
            # V row = k*p + j = (bk*3+lk)*6 + (bj*3+lj)
            # W row = j*n + i = (bj*3+lj)*6 + (bi*3+li)
            
            # But in Strassen, the "A block" and "B block" share the middle index bk.
            # For each Strassen term, we have a combination of A elements from blocks
            # (bi, bk) and B elements from blocks (bk, bj).
            # But Strassen combines A elements from different blocks!
            # E.g., M1 = (A11+A22)(B11+B22) means:
            # A part: A block (bi, bk) at position (0,0) [which is A11] + A block (bi, bk) at (1,1) [A22]
            # Wait no - in 2x2 block Strassen, A11 is the top-left 3x3 block, A22 is bottom-right.
            # So A11 = A[0:3, 0:3], A22 = A[3:6, 3:6] when bi=0, bk=0 for the block structure.
            # Actually I need to be more careful.
            
            # Let me restart the indexing. The 6x6 matrix is divided into 2x2 blocks of size 3x3.
            # Block (0,0) = rows 0-2, cols 0-2
            # Block (0,1) = rows 0-2, cols 3-5
            # Block (1,0) = rows 3-5, cols 0-2
            # Block (1,1) = rows 3-5, cols 3-5
            
            # For C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
            # In block form: C_block(r,c) = sum_s A_block(r,s) * B_block(s,c)
            # where r,c,s are block indices (0 or 1)
            
            # Strassen on the 2x2 block structure:
            # M1 = (A_block(0,0)+A_block(1,1)) * (B_block(0,0)+B_block(1,1))
            # M2 = (A_block(1,0)+A_block(1,1)) * B_block(0,0)
            # M3 = A_block(0,0) * (B_block(0,1)-B_block(1,1))
            # M4 = A_block(1,1) * (B_block(1,0)-B_block(0,0))
            # M5 = (A_block(0,0)+A_block(0,1)) * B_block(1,1)
            # M6 = (A_block(1,0)-A_block(0,0)) * (B_block(0,0)+B_block(0,1))
            # M7 = (A_block(0,1)-A_block(1,1)) * (B_block(1,0)+B_block(1,1))
            
            # C_block(0,0) = M1 + M4 - M5 + M7
            # C_block(0,1) = M3 + M5
            # C_block(1,0) = M2 + M4
            # C_block(1,1) = M1 - M2 + M3 + M6
            
            # So for each Strassen term, we have:
            # A part: sum of A_block(r,s) with coefficients
            # B part: sum of B_block(s,c) with coefficients
            # C part: sum of C_block(r,c) with coefficients
            
            # For the A part, A_block(r,s) contains elements A[i,k] where i=r*3+li, k=s*3+lk
            # U row = i*6 + k = (r*3+li)*6 + (s*3+lk)
            
            # For the B part, B_block(s,c) contains elements B[k,j] where k=s*3+lk, j=c*3+lj
            # V row = k*6 + j = (s*3+lk)*6 + (c*3+lj)
            
            # For the C part, C_block(r,c) contains elements C[i,j] where i=r*3+li, j=c*3+lj
            # W row = j*6 + i = (c*3+lj)*6 + (r*3+li)
            
            # Now I need to restructure. Each Strassen term involves:
            # A: a set of (block_row, block_col, coeff) for A blocks
            # B: a set of (block_row, block_col, coeff) for B blocks
            # C: a set of (block_row, block_col, coeff) for C blocks
            
            # Let me redefine strassen_terms properly:
            pass  # We'll handle this below
            
            col += 1
    
    # The above approach is getting complicated. Let me use a cleaner formulation.
    # I'll restart the construction with a clearer structure.
    
    # Actually, let me just build it directly with explicit block indices.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms defined as:
    # (list of (a_br, a_bc, coeff), list of (b_br, b_bc, coeff), list of (c_br, c_bc, coeff))
    # where a_br, a_bc are A block row/col; b_br, b_bc are B block row/col; c_br, c_bc are C block row/col
    
    strassen_defs = [
        # M1 = (A(0,0)+A(1,1))*(B(0,0)+B(1,1)) -> C(0,0)+C(1,1)
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        # M2 = (A(1,0)+A(1,1))*B(0,0) -> C(1,0)-C(1,1)
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        # M3 = A(0,0)*(B(0,1)-B(1,1)) -> C(0,1)+C(1,1)
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        # M4 = A(1,1)*(B(1,0)-B(0,0)) -> C(0,0)+C(1,0)
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        # M5 = (A(0,0)+A(0,1))*B(1,1) -> -C(0,0)+C(0,1)
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
        # M6 = (A(1,0)-A(0,0))*(B(0,0)+B(0,1)) -> C(1,1)
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        # M7 = (A(0,1)-A(1,1))*(B(1,0)+B(1,1)) -> C(0,0)
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]
    
    col = 0
    for s in range(7):
        a_parts, b_parts, c_parts = strassen_defs[s]
        
        for l in laderman:
            x_coeffs, y_coeffs, z_coeffs = l
            
            # Fill U: for each A block part and each Laderman X coefficient
            for (a_br, a_bc), a_coeff in a_parts:
                for (lr, lc), l_coeff in x_coeffs:
                    gr = a_br * 3 + lr
                    gc = a_bc * 3 + lc
                    U[gr * m + gc][col] += a_coeff * l_coeff
            
            # Fill V: for each B block part and each Laderman Y coefficient
            for (b_br, b_bc), b_coeff in b_parts:
                for (lr, lc), l_coeff in y_coeffs:
                    gr = b_br * 3 + lr  # row of B
                    gc = b_bc * 3 + lc  # col of B
                    V[gc * p + gr][col] += b_coeff * l_coeff
            
            # Fill W: for each C block part and each Laderman Z coefficient
            for (c_br, c_bc), c_coeff in c_parts:
                for (lr, lc), l_coeff in z_coeffs:
                    gr = c_br * 3 + lr  # row of C (= row of A)
                    gc = c_bc * 3 + lc  # col of C (= col of B)
                    W[gc * n + gr][col] += c_coeff * l_coeff
            
            col += 1
    
    return U, V, W, rank


def _verify_6x6_decomp(U, V, W, n, m, p, n_tests=50, seed=7):
    """Verify the 6x6 decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    rng = np.random.RandomState(seed)
    for _ in range(n_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(rank):
            C_computed += np.outer(W_arr[:, r], np.outer(U_arr[:, r], V_arr[:, r]).T)
            # Actually: C[i,j] = sum_r U[i*m+k... wait
            # The decomposition is: C[i,j] = sum_r U[i*m + k_r] * V[k_r*p + j] * W[j*n + i]
            # But U, V, W are fixed vectors. The correct formula is:
            # C[i,j] = sum_{r} (sum_k U[i*m+k]*V[k*p+j]) * W[j*n+i]
            # No wait. The tensor decomposition says:
            # T[i,j,k] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
            # where T[i,j,k] = 1 if this is a valid matmul term, 0 otherwise.
            # For matrix multiplication: C[i,j] = sum_k A[i,k]*B[k,j]
            # So T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matmul)
            # The decomposition: T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
            # So C[i,j] = sum_k A[i,k]*B[k,j] = sum_k sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r]
            #         = sum_r W[j*n+i,r] * (sum_k U[i*m+k,r]*V[k*p+j,r])
            #         = sum_r W[j*n+i,r] * (U_vec_r dot V_vec_r for the specific i,j)
            
            break  # Let me just do it the straightforward way
        # Redo verification properly
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_col = U_arr[:, r]  # length n*m
            v_col = V_arr[:, r]  # length m*p
            w_col = W_arr[:, r]  # length p*n
            # u_col[i*m+k], v_col[k*p+j], w_col[j*n+i]
            for i in range(n):
                for j in range(p):
                    s = 0.0
                    for k in range(m):
                        s += u_col[i*m+k] * v_col[k*p+j]
                    C_computed[i,j] += s * w_col[j*n+i]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True


def _build_6x6_strassen_strassen(n, m, p):
    """Build rank-189 decomposition: 2x2 Strassen outer with schoolbook 3x3 inner."""
    rank = 27 * 7  # 3x3 schoolbook (27) * 7 Strassen terms
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook 3x3: 27 terms, each is (x, y, z) single element
    schoolbook_3x3 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                schoolbook_3x3.append((((i,k),1), ((k,j),1), ((i,j),1)))
    
    strassen_defs = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]
    
    col = 0
    for s in range(7):
        a_parts, b_parts, c_parts = strassen_defs[s]
        for sb in schoolbook_3x3:
            x_coeff, y_coeff, z_coeff = sb
            
            for (a_br, a_bc), a_c in a_parts:
                lr, lc = x_coeff[0]
                gr = a_br * 3 + lr
                gc = a_bc * 3 + lc
                U[gr * m + gc][col] += a_c * x_coeff[1]
            
            for (b_br, b_bc), b_c in b_parts:
                lr, lc = y_coeff[0]
                gr = b_br * 3 + lr
                gc = b_bc * 3 + lc
                V[gc * p + gr][col] += b_c * y_coeff[1]
            
            for (c_br, c_bc), c_c in c_parts:
                lr, lc = z_coeff[0]
                gr = c_br * 3 + lr
                gc = c_bc * 3 + lc
                W[gc * n + gr][col] += c_c * z_coeff[1]
            
            col += 1
    
    return U, V, W, rank


def _build_6x6_schoolbook(n, m, p):
    """Build rank-216 schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix by a 6x6 matrix.
    Returns {'U', 'V', 'W'} with integer entries.
    """
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    # Try rank-161 first (Strassen outer + Laderman inner)
    laderman = _build_rank23_laderman()
    
    # Verify Laderman
    if _verify_3x3_decomp(laderman, n_tests=100, seed=123):
        U, V, W, rank = _build_6x6_strassen_laderman(n, m, p)
        if time.time() < deadline:
            if _verify_6x6_decomp(U, V, W, n, m, p, n_tests=30, seed=7):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to rank-189
    if time.time() < deadline:
        U, V, W, rank = _build_6x6_strassen_strassen(n, m, p)
        if time.time() < deadline:
            if _verify_6x6_decomp(U, V, W, n, m, p, n_tests=30, seed=7):
                return {"U": U, "V": V, "W": W}
    
    # Last resort: schoolbook rank-216
    U, V, W, rank = _build_6x6_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
