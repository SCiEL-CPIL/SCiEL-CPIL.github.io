"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Strategy: Use Strassen for the 4x4x4 part (first 4 columns of B)
    # and schoolbook for the 5th column.
    #
    # The 4x4x4 part: decompose into 7 Strassen block multiplications.
    # Each block multiplication is a 2x2x2 tensor.
    # We can either use schoolbook (rank 8) or Strassen (rank 7) for each 2x2x2.
    #
    # Using schoolbook for each 2x2x2: 7 * 8 = 56 terms
    # Using Strassen for each 2x2x2: 7 * 7 = 49 terms
    #
    # Let's use Strassen (rank 7) for each 2x2x2 to get 49 terms.
    # Plus 16 schoolbook terms for the 5th column.
    # Total rank = 49 + 16 = 65.

    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 decomposition (rank 7)
    # For multiplying 2x2 matrices X * Y = Z:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    #
    # Each M is a scalar multiplication, so we need 7 terms.
    # The U, V, W vectors for each term encode the linear combinations.

    def add_strassen_2x2(t_base, a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add 7 Strassen terms for a 2x2x2 multiplication."""
        # X = A block, Y = B block, Z = C block
        # X[i,j] corresponds to A[a_r0+i, a_c0+j]
        # Y[i,j] corresponds to B[b_r0+i, b_c0+j]
        # Z[i,j] corresponds to C[c_r0+i, c_c0+j]
        
        # Term 1: M1 = (X00+X11)(Y00+Y11)
        # U: X00 and X11 have coeff 1
        U[a_r0*2 + a_c0, t_base] = 1.0
        U[(a_r0+1)*2 + (a_c0+1), t_base] = 1.0
        V[b_r0*2 + b_c0, t_base] = 1.0
        V[(b_r0+1)*2 + (b_c0+1), t_base] = 1.0
        # W: contributes to Z00 (+1) and Z11 (+1)
        W[c_r0*2 + c_c0, t_base] = 1.0
        W[(c_r0+1)*2 + (c_c0+1), t_base] = 1.0
        
        # Term 2: M2 = (X10+X11)Y00
        U[(a_r0+1)*2 + a_c0, t_base+1] = 1.0
        U[(a_r0+1)*2 + (a_c0+1), t_base+1] = 1.0
        V[b_r0*2 + b_c0, t_base+1] = 1.0
        # W: contributes to Z10 (+1) and Z11 (-1)
        W[(c_r0+1)*2 + c_c0, t_base+1] = 1.0
        W[(c_r0+1)*2 + (c_c0+1), t_base+1] = -1.0
        
        # Term 3: M3 = X00(Y01-Y11)
        U[a_r0*2 + a_c0, t_base+2] = 1.0
        V[b_r0*2 + (b_c0+1), t_base+2] = 1.0
        V[(b_r0+1)*2 + (b_c0+1), t_base+2] = -1.0
        # W: contributes to Z01 (+1) and Z11 (+1)
        W[c_r0*2 + (c_c0+1), t_base+2] = 1.0
        W[(c_r0+1)*2 + (c_c0+1), t_base+2] = 1.0
        
        # Term 4: M4 = X11(Y10-Y00)
        U[(a_r0+1)*2 + (a_c0+1), t_base+3] = 1.0
        V[(b_r0+1)*2 + b_c0, t_base+3] = 1.0
        V[b_r0*2 + b_c0, t_base+3] = -1.0
        # W: contributes to Z00 (+1) and Z10 (+1)
        W[c_r0*2 + c_c0, t_base+3] = 1.0
        W[(c_r0+1)*2 + c_c0, t_base+3] = 1.0
        
        # Term 5: M5 = (X00+X01)Y11
        U[a_r0*2 + a_c0, t_base+4] = 1.0
        U[a_r0*2 + (a_c0+1), t_base+4] = 1.0
        V[(b_r0+1)*2 + (b_c0+1), t_base+4] = 1.0
        # W: contributes to Z00 (-1) and Z01 (+1)
        W[c_r0*2 + c_c0, t_base+4] = -1.0
        W[c_r0*2 + (c_c0+1), t_base+4] = 1.0
        
        # Term 6: M6 = (X10-X00)(Y00+Y01)
        U[(a_r0+1)*2 + a_c0, t_base+5] = 1.0
        U[a_r0*2 + a_c0, t_base+5] = -1.0
        V[b_r0*2 + b_c0, t_base+5] = 1.0
        V[b_r0*2 + (b_c0+1), t_base+5] = 1.0
        # W: contributes to Z11 (+1)
        W[(c_r0+1)*2 + (c_c0+1), t_base+5] = 1.0
        
        # Term 7: M7 = (X01-X11)(Y10+Y11)
        U[a_r0*2 + (a_c0+1), t_base+6] = 1.0
        U[(a_r0+1)*2 + (a_c0+1), t_base+6] = -1.0
        V[(b_r0+1)*2 + b_c0, t_base+6] = 1.0
        V[(b_r0+1)*2 + (b_c0+1), t_base+6] = 1.0
        # W: contributes to Z00 (+1)
        W[c_r0*2 + c_c0, t_base+6] = 1.0

    # Part 1: 4x4x4 using 7 Strassen 2x2x2 decompositions (49 terms)
    # Block layout:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # C = [[C00, C01], [C10, C11]] where each block is 2x2
    
    # Strassen formulas for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product, so we use Strassen 2x2x2 (rank 7) for each.
    # Total: 7 * 7 = 49 terms.
    
    # We need to handle the linear combinations in the Strassen formulas.
    # The M blocks are products of linear combinations of A and B blocks.
    # When we apply Strassen 2x2x2 to compute M = X*Y where X and Y are themselves
    # linear combinations, we need to propagate the coefficients.
    
    # This is complex. Let's use a different approach:
    # Use schoolbook (rank 8) for each 2x2x2, giving 7*8 = 56 terms.
    # Then total rank = 56 + 16 = 72.
    
    # Let's implement the 56-term version first.
    
    def add_schoolbook_2x2(t_base, a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add 8 schoolbook terms for a 2x2x2 multiplication."""
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    t = t_base + i*4 + j*2 + k
                    U[(a_r0+i)*2 + (a_c0+k), t] = 1.0
                    V[(b_r0+k)*2 + (b_c0+j), t] = 1.0
                    W[(c_r0+j)*2 + (c_r0+i), t] = 1.0  # Bug: should be (c_r0+j)*n + (c_r0+i) but n=4, so (c_r0+j)*4 + (c_r0+i)
                    # Actually W index is k*n + i, but here k is the column of B / row of C, and i is row of A / col of C
                    # In the 2x2 block context, local indices are 0,1
                    # Global: W[(c_r0+j)*n + (c_r0+i)] where n=4
    
    # Let's fix the indexing. W index = k_global * n + i_global
    # where k_global = b_r0 + k_local, i_global = a_r0 + i_local
    # But in the C block, the row is c_r0 + j_local and col is c_r0 + i_local
    # So W[(c_r0+j)*n + (c_r0+i)] is correct if n=4.
    
    # Actually, let me reconsider. The tensor T[i,j,k] = 1 if C[i,k] = A[i,j]*B[j,k]
    # So for a term, we set:
    # U[i*m+j, t] = 1
    # V[j*p+k, t] = 1  
    # W[k*n+i, t] = 1
    
    # For the 2x2 block at positions (a_r0, a_c0) for A, (b_r0, b_c0) for B, (c_r0, c_c0) for C:
    # i = a_r0 + i_local
    # j = a_c0 + k_local  (column of A = row of B)
    # k = b_c0 + j_local  (column of B)
    # So W index = k*n + i = (b_c0+j_local)*n + (a_r0+i_local)
    # But C row = c_r0 + j_local, C col = c_r0 + i_local
    # So W[(c_r0+j_local)*n + (c_r0+i_local)] should equal W[(b_c0+j_local)*n + (a_r0+i_local)]
    # This requires c_r0 = a_r0 and c_c0 = b_c0, which is true for the diagonal blocks.
    # But for off-diagonal blocks, this doesn't hold!
    
    # I think I'm overcomplicating this. Let me just use the global indices directly.
    
    # Let's restart with a cleaner approach.
    # Use Strassen for the 4x4x4 part with schoolbook 2x2x2 base (rank 56).
    # The key is to correctly map the Strassen block combinations to global indices.
    
    # Actually, let's just use the schoolbook for the entire 4x4x5 (rank 80) 
    # and verify it's correct, then try to improve.
    
    U, V, W, rank = schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

# EVOLVE-BLOCK-END
