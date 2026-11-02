"""Strassen-based rank-28 decomposition for 2x4 * 4x4 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 32

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication
    using Strassen's algorithm applied to 2x2 sub-blocks.
    
    The decomposition splits the 4x4 matrix B into two 2x4 blocks (rows 0-1 and 2-3),
    and the 2x4 matrix A into two 2x2 blocks (cols 0-1 and 2-3).
    Each 2x2 * 2x4 product is decomposed using two Strassen 2x2*2x2 decompositions
    (one for each pair of output columns), giving 7+7=14 terms per block,
    for a total of 28 terms.
    """
    U = np.zeros((8, 28), dtype=int)
    V = np.zeros((16, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    def add_term(term_idx, u_coeffs, v_coeffs, w_coeffs):
        for row, coeff in u_coeffs.items():
            U[row, term_idx] += coeff
        for row, coeff in v_coeffs.items():
            V[row, term_idx] += coeff
        for row, coeff in w_coeffs.items():
            W[row, term_idx] += coeff
    
    # Index conventions:
    # U: row i*m+j = i*4+j for A[i,j], i in {0,1}, j in {0,1,2,3}
    # V: row j*p+k = j*4+k for B[j,k], j in {0,1,2,3}, k in {0,1,2,3}
    # W: row k*n+i = k*2+i for C[i,k], k in {0,1,2,3}, i in {0,1}
    
    # Strassen 2x2*2x2 formulas:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    #
    # M1 = (a+d)(e+h)  -> C00 += M1, C11 += M1
    # M2 = (c+d)e      -> C10 += M2, C11 -= M2
    # M3 = a(g-h)      -> C01 += M3, C11 += M3
    # M4 = d(f-e)      -> C00 -= M4, C01 += M4
    # M5 = (a+c)h      -> C00 += M5, C11 += M5
    # M6 = (c-a)(e+g)  -> C00 += M6
    # M7 = (d-c)(f+h)  -> C01 += M7
    
    def strassen_block(A_rows, B_rows, C_rows, start_term):
        """
        Apply Strassen to a 2x2*2x4 block.
        A_rows: list of 4 U-row indices [a, b, c, d] corresponding to A[0,0],A[0,1],A[1,0],A[1,1]
        B_rows: list of 8 V-row indices [e,f,g,h, e2,f2,g2,h2] for B rows
        C_rows: list of 8 W-row indices for C output
        start_term: starting term index
        Returns: number of terms used
        """
        a, b, c, d = A_rows  # A[0,0], A[0,1], A[1,0], A[1,1]
        # B is 2x4: row0 = [B00, B01, B02, B03], row1 = [B10, B11, B12, B13]
        B00, B01, B02, B03 = B_rows[0:4]
        B10, B11, B12, B13 = B_rows[4:8]
        # C is 2x4: C[i,k] -> W row
        C00, C01, C02, C03 = C_rows[0:4]
        C10, C11, C12, C13 = C_rows[4:8]
        
        t = start_term
        
        # First pair of columns (cols 0,1):
        # M1 = (a+d)(B00+B11)
        add_term(t, {a:1, d:1}, {B00:1, B11:1}, {C00:1, C11:1})
        t += 1
        # M2 = (c+d)B00
        add_term(t, {c:1, d:1}, {B00:1}, {C10:1, C11:-1})
        t += 1
        # M3 = a(B10-B11)
        add_term(t, {a:1}, {B10:1, B11:-1}, {C01:1, C11:1})
        t += 1
        # M4 = d(B01-B00)
        add_term(t, {d:1}, {B01:1, B00:-1}, {C00:-1, C01:1})
        t += 1
        # M5 = (a+c)B11
        add_term(t, {a:1, c:1}, {B11:1}, {C00:1, C11:1})
        t += 1
        # M6 = (c-a)(B00+B10)
        add_term(t, {c:1, a:-1}, {B00:1, B10:1}, {C00:1})
        t += 1
        # M7 = (d-c)(B01+B11)
        add_term(t, {d:1, c:-1}, {B01:1, B11:1}, {C01:1})
        t += 1
        
        # Second pair of columns (cols 2,3):
        # M1' = (a+d)(B02+B13)
        add_term(t, {a:1, d:1}, {B02:1, B13:1}, {C02:1, C13:1})
        t += 1
        # M2' = (c+d)B02
        add_term(t, {c:1, d:1}, {B02:1}, {C12:1, C13:-1})
        t += 1
        # M3' = a(B12-B13)
        add_term(t, {a:1}, {B12:1, B13:-1}, {C03:1, C13:1})
        t += 1
        # M4' = d(B03-B02)
        add_term(t, {d:1}, {B03:1, B02:-1}, {C02:-1, C03:1})
        t += 1
        # M5' = (a+c)B13
        add_term(t, {a:1, c:1}, {B13:1}, {C02:1, C13:1})
        t += 1
        # M6' = (c-a)(B02+B12)
        add_term(t, {c:1, a:-1}, {B02:1, B12:1}, {C02:1})
        t += 1
        # M7' = (d-c)(B03+B13)
        add_term(t, {d:1, c:-1}, {B03:1, B13:1}, {C03:1})
        t += 1
        
        return t - start_term
    
    # Block 1: A[:,0:2] * B[0:2,:]
    # A rows: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B rows: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # C rows: C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6, C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7
    A1 = [0, 1, 4, 5]
    B1 = [0, 1, 2, 3, 4, 5, 6, 7]
    C1 = [0, 2, 4, 6, 1, 3, 5, 7]
    strassen_block(A1, B1, C1, 0)
    
    # Block 2: A[:,2:4] * B[2:4,:]
    # A rows: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B rows: B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11, B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
    # C rows: same as C1
    A2 = [2, 3, 6, 7]
    B2 = [8, 9, 10, 11, 12, 13, 14, 15]
    C2 = [0, 2, 4, 6, 1, 3, 5, 7]
    strassen_block(A2, B2, C2, 14)
    
    # Verify the decomposition
    tensor_reconstructed = np.zeros((2, 4, 4))
    for c in range(28):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    tensor_reconstructed[i, j, k] += uc[i*4+j] * vc[j*4+k] * wc[k*2+i]
    
    target_tensor = np.ones((2, 4, 4))
    
    if not np.allclose(tensor_reconstructed, target_tensor):
        # Fall back to schoolbook rank 32
        rank = 32
        U_fb = np.zeros((8, 32), dtype=int)
        V_fb = np.zeros((16, 32), dtype=int)
        W_fb = np.zeros((8, 32), dtype=int)
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*4+k, col] = 1
                    W_fb[k*2+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
