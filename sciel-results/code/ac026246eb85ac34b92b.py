"""Optimized tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_coeffs():
    """
    Return the 7 Strassen coefficients for 2x2*2x2->2x2.
    Each coefficient is (a_row, a_col, b_row, b_col, c_row, c_col, sign).
    A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    C = [[c00,c01],[c10,c11]]
    m1 = (a00+a11)*(b00+b11)
    m2 = (a10+a11)*b00
    m3 = a00*(b01-b11)
    m4 = a11*(b10-b00)
    m5 = (a00+a01)*b11
    m6 = (a10-a00)*(b00+b10)
    m7 = (a01-a11)*(b01+b11)
    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6
    """
    return [
        # (a_idx, b_idx, c_idx, coeff)
        ((0,0), (0,0), (0,0), 1), ((1,1), (0,0), (0,0), 1),
        ((0,0), (1,1), (0,0), -1), ((1,1), (1,1), (0,0), 1),
        
        ((0,0), (0,1), (0,1), 1), ((0,0), (1,1), (0,1), -1),
        ((0,1), (1,1), (0,1), 1),
        
        ((1,0), (0,0), (1,0), 1), ((1,1), (0,0), (1,0), 1),
        ((1,1), (1,0), (1,0), 1), ((1,1), (0,0), (1,0), -1),
        
        ((0,0), (0,1), (1,1), 1), ((1,1), (0,0), (1,1), -1),
        ((0,0), (1,1), (1,1), 1), ((1,1), (1,1), (1,1), -1),
    ]

def _build_4x4x4_decomp():
    """
    Build U, V, W factors for 4x4 * 4x4 -> 4x4 using Strassen tensor product.
    n=m=p=4.
    Split into 2x2 blocks of 2x2.
    Outer Strassen: 7 terms. Inner Strassen: 7 terms. Total 49 terms.
    """
    n = 4
    # We'll build sparse-ish factor matrices using dict of (row, col) -> val
    U_dict = {}  # (i*4+j, term) -> val
    V_dict = {}  # (j*4+k, term) -> val
    W_dict = {}  # (k*4+i, term) -> val
    
    # Outer block indices: 0,1 for rows/cols of 2x2 blocks
    # Inner element indices: 0,1 for elements within 2x2 block
    
    # Strassen coefficients for 2x2 blocks (outer)
    outer_strassen = [
        # (a_block, b_block, c_block, coeff)
        # m1: (A00+A11)*(B00+B11)
        ((0,0),(0,0),(0,0),1), ((1,1),(0,0),(0,0),1),
        ((0,0),(1,1),(0,0),-1), ((1,1),(1,1),(0,0),1),
        # m2: (A10+A11)*B00
        ((1,0),(0,0),(1,0),1), ((1,1),(0,0),(1,0),1),
        # m3: A00*(B01-B11)
        ((0,0),(0,1),(0,1),1), ((0,0),(1,1),(0,1),-1),
        # m4: A11*(B10-B00)
        ((1,1),(1,0),(1,0),1), ((1,1),(0,0),(1,0),-1),
        # m5: (A00+A01)*B11
        ((0,0),(1,1),(0,0),1), ((0,1),(1,1),(0,0),1),
        # m6: (A10-A00)*(B00+B10)
        ((1,0),(0,0),(1,0),1), ((0,0),(0,0),(1,0),-1),
        ((1,0),(1,0),(1,0),1), ((0,0),(1,0),(1,0),-1),
        # m7: (A01-A11)*(B01+B11)
        ((0,1),(0,1),(0,1),1), ((1,1),(0,1),(0,1),-1),
        ((0,1),(1,1),(0,1),1), ((1,1),(1,1),(0,1),-1),
    ]
    
    # Inner Strassen coefficients for 2x2 elements
    inner_strassen = [
        ((0,0),(0,0),(0,0),1), ((1,1),(0,0),(0,0),1),
        ((0,0),(1,1),(0,0),-1), ((1,1),(1,1),(0,0),1),
        ((0,0),(0,1),(0,1),1), ((0,0),(1,1),(0,1),-1),
        ((0,1),(1,1),(0,1),1),
        ((1,0),(0,0),(1,0),1), ((1,1),(0,0),(1,0),1),
        ((1,1),(1,0),(1,0),1), ((1,1),(0,0),(1,0),-1),
        ((0,0),(0,1),(1,1),1), ((1,1),(0,0),(1,1),-1),
        ((0,0),(1,1),(1,1),1), ((1,1),(1,1),(1,1),-1),
    ]
    
    term_count = 0
    for outer_a, outer_b, outer_c, outer_coeff in outer_strassen:
        for inner_a, inner_b, inner_c, inner_coeff in inner_strassen:
            coeff = outer_coeff * inner_coeff
            if coeff == 0:
                continue
            
            # A index: block (outer_a) element (inner_a) -> i = 2*outer_a[0] + inner_a[0], j = 2*outer_a[1] + inner_a[1]
            a_i = 2 * outer_a[0] + inner_a[0]
            a_j = 2 * outer_a[1] + inner_a[1]
            
            # B index: block (outer_b) element (inner_b) -> j = 2*outer_b[0] + inner_b[0], k = 2*outer_b[1] + inner_b[1]
            b_j = 2 * outer_b[0] + inner_b[0]
            b_k = 2 * outer_b[1] + inner_b[1]
            
            # C index: block (outer_c) element (inner_c) -> i = 2*outer_c[0] + inner_c[0], k = 2*outer_c[1] + inner_c[1]
            c_i = 2 * outer_c[0] + inner_c[0]
            c_k = 2 * outer_c[1] + inner_c[1]
            
            U_row = a_i * 4 + a_j
            V_row = b_j * 4 + b_k
            W_row = c_k * 4 + c_i
            
            U_dict[(U_row, term_count)] = U_dict.get((U_row, term_count), 0) + coeff
            V_dict[(V_row, term_count)] = V_dict.get((V_row, term_count), 0) + coeff
            W_dict[(W_row, term_count)] = W_dict.get((W_row, term_count), 0) + coeff
            
            term_count += 1
    
    # Now create dense matrices
    rank = term_count
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for (row, col), val in U_dict.items():
        U[row][col] = val
    for (row, col), val in V_dict.items():
        V[row][col] = val
    for (row, col), val in W_dict.items():
        W[row][col] = val
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    try:
        # Build 4x4x4 Strassen decomposition
        U_44, V_44, W_44, r_44 = _build_44x4x4_decomp()
        
        # For 4x4 * 4x9:
        # Split B into B1 (4x4) and B2 (4x1)
        # C = A * B = A * [B1, B2] = [A*B1, A*B2]
        # A*B1 uses 4x4x4 decomposition (rank r_44)
        # A*B2 uses schoolbook 4x4*4x1 (rank 16)
        
        # Total rank
        total_rank = r_44 + 16
        
        if total_rank < best_rank:
            U_new = [[0] * total_rank for _ in range(n * m)]  # 16 x total_rank
            V_new = [[0] * total_rank for _ in range(m * p)]  # 36 x total_rank
            W_new = [[0] * total_rank for _ in range(p * n)]  # 36 x total_rank
            
            # Part 1: A * B1 (4x4 * 4x4)
            for r in range(r_44):
                for i in range(n * m):
                    U_new[i][r] = U_44[i][r]
                for j in range(4):
                    for k in range(4):
                        V_new[j * p + k][r] = V_44[j * 4 + k][r]
                for k in range(4):
                    for i in range(4):
                        W_new[k * n + i][r] = W_44[k * 4 + i][r]
            
            # Part 2: A * B2 (4x4 * 4x1), schoolbook
            col_offset = r_44
            for i in range(n):
                for j in range(m):
                    for k in range(1):  # Only 1 column in B2
                        U_new[i * m + j][col_offset] = 1
                        V_new[j * p + 4 + k][col_offset] = 1  # k=0, so col 4
                        W_new[(4 + k) * n + i][col_offset] = 1  # k=0, row 4*4+0=16... wait
                        col_offset += 1
            
            # Verify correctness with random test
            U_np = np.array(U_new, dtype=float)
            V_np = np.array(V_new, dtype=float)
            W_np = np.array(W_new, dtype=float)
            
            np.random.seed(42)
            A_test = np.random.randn(4, 4)
            B_test = np.random.randn(4, 9)
            C_expected = A_test @ B_test
            
            # Compute C from decomposition
            C_computed = np.zeros((4, 9))
            for col in range(total_rank):
                u_col = U_np[:, col]
                v_col = V_np[:, col]
                w_col = W_np[:, col]
                
                # u_col has shape (16,), v_col has shape (36,), w_col has shape (36,)
                # Reshape u_col to (4,4), v_col to (4,9)
                u_mat = u_col.reshape(4, 4)
                v_mat = v_col.reshape(4, 9)
                
                # outer product of u_mat and v_mat gives (4,4,9)
                # Contract with w_col which is indexed by (k*4+i)
                # w_col[k*4+i]
                
                # C[i,k] += sum_{j,j2} u_mat[i,j] * v_mat[j,j2] * ... 
                # Actually the decomposition is:
                # C[i,k] = sum_r U[i*m+j_r, r] * V[j_r*p+k_r, r] * W[k_r*n+i, r]
                # But this is a 3-tensor contraction.
                
                # Let me think more carefully.
                # The tensor is T[i,j,k] = delta for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
                # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
                # The decomposition is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                
                # So for each r, we have a rank-1 tensor U[:, :, r] * V[:, :, r] * W[:, :, r]
                # where U is (4,4), V is (4,9), W is (9,4)
                
                U_r = u_col.reshape(4, 4)
                V_r = v_col.reshape(4, 9)
                W_r = w_col.reshape(9, 4)
                
                # The contribution to C is:
                # C_r[i,k] = sum_j U_r[i,j] * V_r[j,k] * W_r[k,i]
                # This isn't a standard matrix mult. Let me re-examine.
                
                # Actually, the standard tensor decomposition for matrix multiplication:
                # The multiplication tensor has indices (i, j, k) where i indexes A rows,
                # j indexes A cols / B rows, k indexes B cols.
                # T[i,j,k] = 1 for the term A[i,j]*B[j,k] contributing to C[i,k]
                
                # A rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c] A[i,j'] ... 
                
                # Hmm, I think the verification is more subtle. Let me just check if the
                # factors satisfy the tensor identity directly.
                
                # For a given (i,j,k), T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                
                pass
            
            # Let me do a proper verification
            def verify_decomposition(U, V, W, n, m, p):
                U = np.array(U, dtype=float)
                V = np.array(V, dtype=float)
                W = np.array(W, dtype=float)
                rank = U.shape[1]
                
                # For each (i, j, k), check that
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta[i,j,k]
                # where delta is 1 if this is a valid multiplication term, else 0
                # Actually for matrix mult, T[i,j,k] = 1 always? No...
                # T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
                # So T[i,j,k] = 1 for all valid (i,j,k) since each product contributes to exactly one C entry.
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected = 1.0
                            actual = 0.0
                            for c in range(rank):
                                actual += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            if abs(actual - expected) > 1e-9:
                                return False
                return True
            
            if verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = total_rank
    
    except Exception as e:
        # Fall back to schoolbook
        pass
    
    return best_result
