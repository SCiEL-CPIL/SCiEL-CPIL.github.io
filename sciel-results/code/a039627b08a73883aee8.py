"""Matrix multiplication 2,4,6 rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    # Use the standard Strassen-like decomposition for 2x2 by 2x2 blocks.
    # For 2x4 * 4x6, we can view as 2x(2*2) * (2*2)x(3*2) = 2x4 * 4x6.
    # Actually, let's use the known optimal decomposition for 2x4 * 4x6.
    # The schoolbook rank is 2*4*6 = 48.
    # A common improvement uses Strassen for the 2x2 block structure.
    # Partition A (2x4) into A1 (2x2), A2 (2x2)
    # Partition B (4x6) into B1 (2x6), B2 (2x6) where B1 has rows 0-1, B2 has rows 2-3.
    # C = A1*B1 + A2*B2.
    # Now we need to compute two 2x2 * 2x6 products.
    # For each 2x2 * 2x6, we can use Strassen on the 2x2 part and treat the 6 columns.
    # Strassen for 2x2 * 2x2 uses 7 multiplications.
    # For 2x2 * 2x6, we can split the 6 columns into 3 pairs of 2 columns.
    # So 3 * 7 = 21 multiplications for each 2x2 * 2x6 block.
    # Total: 21 + 21 = 42.
    # This matches the inspiration score of 1.1429 (48/42 = 1.14285...).
    
    # Let's construct the 42-term decomposition.
    # A is 2x4: A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x6: B[j, k] where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # C is 2x6: C[i, k] where i in {0,1}, k in {0,1,2,3,4,5}
    
    # Split B into B_left (2x6: rows 0,1) and B_right (2x6: rows 2,3)
    # C = A_left * B_left + A_right * B_right
    # where A_left = A[:, 0:2], A_right = A[:, 2:4]
    
    # For A_left * B_left (2x2 * 2x6):
    # Split B_left columns into 3 pairs: B_left[:, 0:2], B_left[:, 2:4], B_left[:, 4:6]
    # For each 2x2 * 2x2 block, use Strassen (7 terms).
    
    # Strassen for X (2x2) * Y (2x2) = Z (2x2):
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y10)
    # M7 = (x01-x11)(y01+y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For the full decomposition, we need to express each term as:
    # term_c = (sum_{ij} U[i*m+j, c] * A[i,j]) * (sum_{jk} V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Actually, the standard tensor decomposition form is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_{j,k} V[j*p+k, c] * B[j,k]) ... no.
    
    # Let's use the standard form:
    # C[i,k] = sum_c alpha_c * (sum_{i,j} U_{ij}^c A_{ij}) * (sum_{j,k} V_{jk}^c B_{jk})
    # where alpha_c = W_{ki}^c? No, W is p*n, so W[k*n+i, c].
    
    # The standard bilinear decomposition for matrix multiplication C = AB is:
    # C_{ik} = sum_c (sum_{ij} U^c_{ij} A_{ij}) (sum_{jk} V^c_{jk} B_{jk}) W^c_{ki}
    # Wait, the indices in W are k*n+i, so W^c is a matrix of size p x n, i.e., W^c_{k,i}.
    
    # So: C_{ik} = sum_c W^c_{k,i} * (sum_{ij} U^c_{ij} A_{ij}) * (sum_{jk} V^c_{jk} B_{jk})
    
    # Let's build the 42 terms.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Helper to set a Strassen term
    # For a 2x2 * 2x2 block:
    # A_block is 2x2: rows a_i0, a_i1; cols a_j0, a_j1
    # B_block is 2x2: rows b_j0, b_j1; cols b_k0, b_k1
    # C_block is 2x2: rows c_i0, c_i1; cols c_k0, c_k1
    
    def add_strassen_block(term_idx, a_i0, a_j0, b_j0, b_k0):
        # A block: A[a_i0, a_j0], A[a_i0, a_j0+1], A[a_i0+1, a_j0], A[a_i0+1, a_j0+1]
        # B block: B[b_j0, b_k0], B[b_j0, b_k0+1], B[b_j0+1, b_k0], B[b_j0+1, b_k0+1]
        # C block: C[a_i0, b_k0], C[a_i0, b_k0+1], C[a_i0+1, b_k0], C[a_i0+1, b_k0+1]
        
        # M1 = (x00+x11)(y00+y11)
        # U: A[a_i0, a_j0] + A[a_i0+1, a_j0+1]
        # V: B[b_j0, b_k0] + B[b_j0+1, b_k0+1]
        # W: C[a_i0, b_k0] += 1, C[a_i0, b_k0+1] += 1, C[a_i0+1, b_k0] += 1, C[a_i0+1, b_k0+1] -= 1
        # Wait, Z00 = M1 + M4 - M5 + M7, so W for M1 contributes to all 4 outputs with signs.
        
        # Actually, let's just directly compute the coefficients for each Strassen multiplication.
        # For M1:
        # U: A[a_i0, a_j0] += 1, A[a_i0+1, a_j0+1] += 1
        # V: B[b_j0, b_k0] += 1, B[b_j0+1, b_k0+1] += 1
        # W: C[a_i0, b_k0] += 1, C[a_i0, b_k0+1] += 1, C[a_i0+1, b_k0] += 1, C[a_i0+1, b_k0+1] += 1
        # But Z00 = M1 + M4 - M5 + M7, so the contribution to C[a_i0, b_k0] from M1 is +1.
        # Z01 = M3 + M5, so M1 doesn't contribute to Z01.
        # Z10 = M2 + M4, so M1 doesn't contribute to Z10.
        # Z11 = M1 - M2 + M3 + M6, so M1 contributes +1 to Z11.
        
        # So for M1:
        # W: C[a_i0, b_k0] += 1, C[a_i0+1, b_k0+1] += 1
        
        U[a_i0 * m + a_j0][term_idx] += 1
        U[(a_i0 + 1) * m + (a_j0 + 1)][term_idx] += 1
        V[b_j0 * p + b_k0][term_idx] += 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][term_idx] += 1
        W[b_k0 * n + a_i0][term_idx] += 1
        W[(b_k0 + 1) * n + (a_i0 + 1)][term_idx] += 1
        
    def add_strassen_block_m2(term_idx, a_i0, a_j0, b_j0, b_k0):
        # M2 = (x10+x11)y00
        # U: A[a_i0+1, a_j0] += 1, A[a_i0+1, a_j0+1] += 1
        # V: B[b_j0, b_k0] += 1
        # W: C[a_i0+1, b_k0] += 1, C[a_i0+1, b_k0+1] -= 1
        U[(a_i0 + 1) * m + a_j0][term_idx] += 1
        U[(a_i0 + 1) * m + (a_j0 + 1)][term_idx] += 1
        V[b_j0 * p + b_k0][term_idx] += 1
        W[b_k0 * n + (a_i0 + 1)][term_idx] += 1
        W[(b_k0 + 1) * n + (a_i0 + 1)][term_idx] -= 1
        
    def add_strassen_block_m3(term_idx, a_i0, a_j0, b_j0, b_k0):
        # M3 = x00(y01-y11)
        # U: A[a_i0, a_j0] += 1
        # V: B[b_j0, b_k0+1] += 1, B[b_j0+1, b_k0+1] -= 1
        # W: C[a_i0, b_k0+1] += 1, C[a_i0+1, b_k0+1] += 1
        U[a_i0 * m + a_j0][term_idx] += 1
        V[b_j0 * p + (b_k0 + 1)][term_idx] += 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][term_idx] -= 1
        W[(b_k0 + 1) * n + a_i0][term_idx] += 1
        W[(b_k0 + 1) * n + (a_i0 + 1)][term_idx] += 1
        
    def add_strassen_block_m4(term_idx, a_i0, a_j0, b_j0, b_k0):
        # M4 = x11(y10-y00)
        # U: A[a_i0+1, a_j0+1] += 1
        # V: B[b_j0+1, b_k0] += 1, B[b_j0, b_k0] -= 1
        # W: C[a_i0, b_k0] += 1, C[a_i0+1, b_k0] += 1
        U[(a_i0 + 1) * m + (a_j0 + 1)][term_idx] += 1
        V[(b_j0 + 1) * p + b_k0][term_idx] += 1
        V[b_j0 * p + b_k0][term_idx] -= 1
        W[b_k0 * n + a_i0][term_idx] += 1
        W[b_k0 * n + (a_i0 + 1)][term_idx] += 1
        
    def add_strassen_block_m5(term_idx, a_i0, a_j0, b_j0, b_k0):
        # M5 = (x00+x01)y11
        # U: A[a_i0, a_j0] += 1, A[a_i0, a_j0+1] += 1
        # V: B[b_j0+1, b_k0+1] += 1
        # W: C[a_i0, b_k0] -= 1, C[a_i0, b_k0+1] += 1
        U[a_i0 * m + a_j0][term_idx] += 1
        U[a_i0 * m + (a_j0 + 1)][term_idx] += 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][term_idx] += 1
        W[b_k0 * n + a_i0][term_idx] -= 1
        W[(b_k0 + 1) * n + a_i0][term_idx] += 1
        
    def add_strassen_block_m6(term_idx, a_i0, a_j0, b_j0, b_k0):
        # M6 = (x10-x00)(y00+y10)
        # U: A[a_i0+1, a_j0] += 1, A[a_i0, a_j0] -= 1
        # V: B[b_j0, b_k0] += 1, B[b_j0+1, b_k0] += 1
        # W: C[a_i0+1, b_k0+1] += 1
        U[(a_i0 + 1) * m + a_j0][term_idx] += 1
        U[a_i0 * m + a_j0][term_idx] -= 1
        V[b_j0 * p + b_k0][term_idx] += 1
        V[(b_j0 + 1) * p + b_k0][term_idx] += 1
        W[(b_k0 + 1) * n + (a_i0 + 1)][term_idx] += 1
        
    def add_strassen_block_m7(term_idx, a_i0, a_j0, b_j0, b_k0):
        # M7 = (x01-x11)(y01+y11)
        # U: A[a_i0, a_j0+1] += 1, A[a_i0+1, a_j0+1] -= 1
        # V: B[b_j0, b_k0+1] += 1, B[b_j0+1, b_k0+1] += 1
        # W: C[a_i0, b_k0] += 1, C[a_i0+1, b_k0+1] -= 1
        U[a_i0 * m + (a_j0 + 1)][term_idx] += 1
        U[(a_i0 + 1) * m + (a_j0 + 1)][term_idx] -= 1
        V[b_j0 * p + (b_k0 + 1)][term_idx] += 1
        V[(b_j0 + 1) * p + (b_k0 + 1)][term_idx] += 1
        W[b_k0 * n + a_i0][term_idx] += 1
        W[(b_k0 + 1) * n + (a_i0 + 1)][term_idx] -= 1
        
    # Now apply to A_left * B_left (rows 0-1 of A, rows 0-1 of B)
    # A_left = A[0:2, 0:2], B_left = B[0:2, 0:6]
    # Split B_left columns into 3 pairs: (0,1), (2,3), (4,5)
    term = 0
    for col_pair in range(3):
        b_k0 = col_pair * 2
        add_strassen_block(term, 0, 0, 0, b_k0)  # M1
        term += 1
        add_strassen_block_m2(term, 0, 0, 0, b_k0)  # M2
        term += 1
        add_strassen_block_m3(term, 0, 0, 0, b_k0)  # M3
        term += 1
        add_strassen_block_m4(term, 0, 0, 0, b_k0)  # M4
        term += 1
        add_strassen_block_m5(term, 0, 0, 0, b_k0)  # M5
        term += 1
        add_strassen_block_m6(term, 0, 0, 0, b_k0)  # M6
        term += 1
        add_strassen_block_m7(term, 0, 0, 0, b_k0)  # M7
        term += 1
    
    # A_right * B_right (rows 0-1 of A, rows 2-3 of B)
    # A_right = A[0:2, 2:4], B_right = B[2:4, 0:6]
    for col_pair in range(3):
        b_k0 = col_pair * 2
        b_j0 = 2
        a_j0 = 2
        add_strassen_block(term, 0, a_j0, b_j0, b_k0)  # M1
        term += 1
        add_strassen_block_m2(term, 0, a_j0, b_j0, b_k0)  # M2
        term += 1
        add_strassen_block_m3(term, 0, a_j0, b_j0, b_k0)  # M3
        term += 1
        add_strassen_block_m4(term, 0, a_j0, b_j0, b_k0)  # M4
        term += 1
        add_strassen_block_m5(term, 0, a_j0, b_j0, b_k0)  # M5
        term += 1
        add_strassen_block_m6(term, 0, a_j0, b_j0, b_k0)  # M6
        term += 1
        add_strassen_block_m7(term, 0, a_j0, b_j0, b_k0)  # M7
        term += 1
    
    # Convert to list of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
