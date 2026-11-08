"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_decomposition(n=2):
    """Return Strassen decomposition for 2x2 matrix multiplication.
    
    Returns U (4x7), V (4x7), W (4x7) such that for any A(2x2), B(2x2):
    C = sum_c U[:,c] * V[:,c] * W[:,c]^T where C is 2x2.
    
    Convention: U[i,j] corresponds to A[i,j], V[i,j] to B[i,j], W[i,j] to C[i,j].
    """
    # Strassen's 7 products
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B11-B10)
    # P4 = A11(B10+B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 gets +1, C11 gets +1
    
    # Wait, W should have one row per C entry. Let me redo this.
    # Actually for the decomposition format:
    # C[i,j] = sum_c U[i,j,c] * V[i,j,c] * W[i,j,c]
    # No wait, the standard tensor decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where T is the matmul tensor for A(m,n) * B(n,p) -> C(m,p)
    # T[i,k,j] = 1 if C[i,k] = sum_j A[i,j] B[j,k]
    
    # For 2x2 * 2x2 -> 2x2:
    # T[i,k,j] = 1 for all i,k,j in {0,1}
    # U shape: (2*2, 7) = (4, 7) indexed by [i*2+j]
    # V shape: (2*2, 7) = (4, 7) indexed by [j*2+k]  
    # W shape: (2*2, 7) = (4, 7) indexed by [k*2+i]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Index helpers
    def u_idx(i, j): return i * 2 + j
    def v_idx(j, k): return j * 2 + k
    def w_idx(k, i): return k * 2 + i
    
    # P1 = (A00+A11)(B00+B11)
    U[u_idx(0,0), 0] += 1
    U[u_idx(1,1), 0] += 1
    V[v_idx(0,0), 0] += 1
    V[v_idx(0,1), 0] += 1
    # C00 += P1, C11 += P1
    W[w_idx(0,0), 0] += 1
    W[w_idx(1,1), 0] += 1
    
    # P2 = (A10+A11)B00
    U[u_idx(1,0), 1] += 1
    U[u_idx(1,1), 1] += 1
    V[v_idx(0,0), 1] += 1
    # C10 += P2
    W[w_idx(0,1), 1] += 1
    
    # P3 = A00(B11-B10)
    U[u_idx(0,0), 2] += 1
    V[v_idx(1,1), 2] += 1
    V[v_idx(1,0), 2] += -1
    # C01 += P3
    W[w_idx(1,0), 2] += 1
    
    # P4 = A11(B10+B00)
    U[u_idx(1,1), 3] += 1
    V[v_idx(1,0), 3] += 1
    V[v_idx(0,0), 3] += 1
    # C00 += P4, C10 += P4
    W[w_idx(0,0), 3] += 1
    W[w_idx(0,1), 3] += 1
    
    # P5 = (A00+A01)B11
    U[u_idx(0,0), 4] += 1
    U[u_idx(0,1), 4] += 1
    V[v_idx(1,1), 4] += 1
    # C00 -= P5, C01 += P5
    W[w_idx(0,0), 4] += -1
    W[w_idx(1,0), 4] += 1
    
    # P6 = (A10-A00)(B00+B01)
    U[u_idx(1,0), 5] += 1
    U[u_idx(0,0), 5] += -1
    V[v_idx(0,0), 5] += 1
    V[v_idx(0,1), 5] += 1
    # C11 += P6
    W[w_idx(1,1), 5] += 1
    
    # P7 = (A01-A11)(B10+B11)
    U[u_idx(0,1), 6] += 1
    U[u_idx(1,1), 6] += -1
    V[v_idx(1,0), 6] += 1
    V[v_idx(1,1), 6] += 1
    # C00 += P7
    W[w_idx(0,0), 6] += 1
    
    return U, V, W

def _strassen_4x4_decomposition():
    """Return Strassen-based decomposition for 4x4 * 4x4 -> 4x4.
    
    Split into 2x2 blocks and apply Strassen to each block multiplication.
    This gives rank 7*4 = 28 for the 4x4x4 case... wait, no.
    
    Actually, for 4x4 * 4x4:
    A = [[A00, A01], [A10, A11]] where each block is 2x2
    B = [[B00, B01], [B10, B11]] where each block is 2x2
    
    Using Strassen at the block level:
    P1 = (A00+A11)(B00+B11)
    P2 = (A10+A11)B00
    P3 = A00(B11-B10)
    P4 = A11(B10+B00)
    P5 = (A00+A01)B11
    P6 = (A10-A00)(B00+B01)
    P7 = (A01-A11)(B10+B11)
    
    Each Pi is a 2x2 * 2x2 multiplication, which itself needs rank 7 (Strassen) or 4 (schoolbook).
    Using Strassen for the inner: 7 * 7 = 49 rank.
    Using schoolbook for the inner: 7 * 4 = 28 rank.
    
    Let's use schoolbook for inner for simplicity and lower rank.
    """
    n_inner = 2
    # For each of the 7 Strassen products, we need to multiply two 2x2 blocks.
    # Each such multiplication has rank 4 (schoolbook) or 7 (Strassen).
    # Let's use schoolbook: total rank = 7 * 4 = 28.
    
    U_total = []
    V_total = []
    W_total = []
    
    # The 7 Strassen combinations at block level
    # Each product involves linear combinations of A blocks and B blocks
    # then a 2x2 * 2x2 multiplication
    
    # For each Strassen term, we have:
    # Term = (sum of a*A_ik) * (sum of b*B_kl)
    # which expands to sum of a*b * (A_ik * B_kl)
    # Each A_ik * B_kl is a 2x2*2x2 mult with rank 4
    
    strassen_terms = [
        # (a_block_coeffs, b_block_coeffs, result_block_positions)
        # Term 1: (A00+A11)(B00+B11)
        {
            'a': [(0,0,1), (1,1,1)],
            'b': [(0,0,1), (0,1,1)],
            'c': [(0,0,1), (1,1,1)]  # C00 += term, C11 += term
        },
        # Term 2: (A10+A11)B00
        {
            'a': [(1,0,1), (1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1)]  # C10 += term
        },
        # Term 3: A00(B11-B10)
        {
            'a': [(0,0,1)],
            'b': [(1,1,1), (1,0,-1)],
            'c': [(0,1,1)]  # C01 += term
        },
        # Term 4: A11(B10+B00)
        {
            'a': [(1,1,1)],
            'b': [(1,0,1), (0,0,1)],
            'c': [(0,0,1), (1,0,1)]  # C00 += term, C10 += term
        },
        # Term 5: (A00+A01)B11
        {
            'a': [(0,0,1), (0,1,1)],
            'b': [(1,1,1)],
            'c': [(0,0,-1), (0,1,1)]  # C00 -= term, C01 += term
        },
        # Term 6: (A10-A00)(B00+B01)
        {
            'a': [(1,0,1), (0,0,-1)],
            'b': [(0,0,1), (0,1,1)],
            'c': [(1,1,1)]  # C11 += term
        },
        # Term 7: (A01-A11)(B10+B11)
        {
            'a': [(0,1,1), (1,1,-1)],
            'b': [(1,0,1), (1,1,1)],
            'c': [(0,0,1)]  # C00 += term
        }
    ]
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    for term in strassen_terms:
        for a_coeff in term['a']:
            ai, aj, a_scale = a_coeff
            for b_coeff in term['b']:
                bi, bj, b_scale = b_coeff
                for c_coeff in term['c']:
                    ci, cj, c_scale = c_coeff
                    # This is a 2x2 * 2x2 multiplication: A[ai,aj block] * B[bi,bj block]
                    # Block A[ai,aj] is at rows 2*ai:2*ai+2, cols 2*aj:2*aj+2
                    # Block B[bi,bj] is at rows 2*bi:2*bi+2, cols 2*bj:2*bj+2
                    # Result contributes to C block at rows 2*ci:2*ci+2, cols 2*cj:2*cj+2
                    
                    # For each of the 4 elements in the 2x2*2x2 mult:
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                # A element: global row 2*ai+ii, col 2*aj+jj
                                # B element: global row 2*bi+jj, col 2*bj+kk
                                # C element: global row 2*ci+ii, col 2*cj+kk
                                # Wait, I need to be careful with indices.
                                # A is 4x4, indexed [row, col]
                                # B is 4x4, indexed [row, col]
                                # C is 4x4, indexed [row, col]
                                
                                a_row = 2 * ai + ii
                                a_col = 2 * aj + jj
                                b_row = 2 * bi + jj
                                b_col = 2 * bj + kk
                                c_row = 2 * ci + ii
                                c_col = 2 * cj + kk
                                
                                scale = a_scale * b_scale * c_scale
                                
                                u_col = np.zeros(16)
                                v_col = np.zeros(16)
                                w_col = np.zeros(16)
                                
                                u_col[a_row * 4 + a_col] = scale
                                v_col[b_row * 4 + b_col] = 1
                                # W index: k*n + i where k=c_col, n=4, i=c_row
                                w_col[c_col * 4 + c_row] = 1
                                
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
                                rank += 1
    
    U = np.column_stack(U_cols) if U_cols else np.zeros((16, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((16, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((16, 0))
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strategy: Split the 4x7 B matrix into 4x4 and 4x3 parts.
    # Use Strassen-based decomposition for 4x4*4x4 part (lower rank than schoolbook)
    # Use schoolbook for 4x4*4x3 part
    
    # Part 1: 4x4 * 4x4 -> first 4 columns of result
    # Part 2: 4x4 * 4x3 -> last 3 columns of result
    
    # For 4x4*4x4, use Strassen at block level with schoolbook inner: rank 28
    # For 4x4*4x3, schoolbook: rank 4*4*3 = 48
    # Total: 76
    
    # Let me construct this properly.
    
    # First, build the 4x4*4x4 decomposition using block Strassen
    # Then build the 4x4*4x3 decomposition using schoolbook
    
    # Actually, let me think about this more carefully.
    # The full problem is A(4x4) * B(4x7) = C(4x7)
    # We can split B into B1(4x4) and B2(4x3)
    # C = A*B1 + A*B2
    # C1 = A*B1 (4x4), C2 = A*B2 (4x3)
    
    # For A*B1: use Strassen decomposition (rank ~28)
    # For A*B2: use schoolbook (rank 48)
    
    # But we need to combine them into a single decomposition.
    
    # Let me build the Strassen 4x4x4 decomposition properly.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: A * B1 where B1 is 4x4 (columns 0-3 of B)
    # Use block Strassen with schoolbook inner multiplications
    
    strassen_terms = [
        # Term 1: (A00+A11)(B00+B11)
        {'a': [(0,0,1), (1,1,1)], 'b': [(0,0,1), (0,1,1)], 'c': [(0,0,1), (1,1,1)]},
        # Term 2: (A10+A11)B00
        {'a': [(1,0,1), (1,1,1)], 'b': [(0,0,1)], 'c': [(1,0,1)]},
        # Term 3: A00(B11-B10)
        {'a': [(0,0,1)], 'b': [(1,1,1), (1,0,-1)], 'c': [(0,1,1)]},
        # Term 4: A11(B10+B00)
        {'a': [(1,1,1)], 'b': [(1,0,1), (0,0,1)], 'c': [(0,0,1), (1,0,1)]},
        # Term 5: (A00+A01)B11
        {'a': [(0,0,1), (0,1,1)], 'b': [(1,1,1)], 'c': [(0,0,-1), (0,1,1)]},
        # Term 6: (A10-A00)(B00+B01)
        {'a': [(1,0,1), (0,0,-1)], 'b': [(0,0,1), (0,1,1)], 'c': [(1,1,1)]},
        # Term 7: (A01-A11)(B10+B11)
        {'a': [(0,1,1), (1,1,-1)], 'b': [(1,0,1), (1,1,1)], 'c': [(0,0,1)]}
    ]
    
    for term in strassen_terms:
        for a_coeff in term['a']:
            ai, aj, a_scale = a_coeff
            for b_coeff in term['b']:
                bi, bj, b_scale = b_coeff
                for c_coeff in term['c']:
                    ci, cj, c_scale = c_coeff
                    # 2x2 * 2x2 multiplication with schoolbook (rank 4)
                    for ii in range(2):
                        for kk in range(2):
                            # Sum over jj in 0..1
                            for jj in range(2):
                                scale = a_scale * b_scale * c_scale
                                
                                a_row = 2 * ai + ii
                                a_col = 2 * aj + jj
                                b_row = 2 * bi + jj
                                b_col = 2 * bj + kk
                                c_row = 2 * ci + ii
                                c_col = 2 * cj + kk
                                
                                u_col = np.zeros(16)
                                v_col = np.zeros(16)
                                w_col = np.zeros(16)
                                
                                u_col[a_row * 4 + a_col] = scale
                                v_col[b_row * 7 + b_col] = 1  # B is 4x7, so 7 cols
                                w_col[c_col * 4 + c_row] = 1  # C is 4x7, W index: k*n+i
                                
                                U_parts.append(u_col)
                                V_parts.append(v_col)
                                W_parts.append(w_col)
    
    # Part 2: A * B2 where B2 is 4x3 (columns 4-6 of B)
    # Schoolbook: rank 4*4*3 = 48
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                u_col = np.zeros(16)
                v_col = np.zeros(16)
                w_col = np.zeros(16)
                
                u_col[i * 4 + j] = 1
                v_col[j * 7 + k] = 1
                w_col[k * 4 + i] = 1
                
                U_parts.append(u_col)
                V_parts.append(v_col)
                W_parts.append(w_col)
    
    if not U_parts:
        return {"U": np.zeros((16, 0)).tolist(), 
                "V": np.zeros((16, 0)).tolist(), 
                "W": np.zeros((16, 0)).tolist()}
    
    U = np.column_stack(U_parts)
    V = np.column_stack(V_parts)
    W = np.column_stack(W_parts)
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
