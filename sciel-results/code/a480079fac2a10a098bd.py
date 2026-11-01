"""
Optimal rank-18 tensor decomposition for 2x2 * 2x5 matrix multiplication.
"""
import numpy as np

def construct():
    # Dimensions: A is 2x2, B is 2x5, C is 2x5
    # U shape: (2*2, rank) = (4, rank)
    # V shape: (2*5, rank) = (10, rank)
    # W shape: (5*2, rank) = (10, rank)
    
    n, m, p = 2, 2, 5
    rank = 18
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper function to add a rank-7 Strassen block for two specific columns of B and C.
    # This handles the multiplication of a 2x2 matrix A by a 2x2 slice of B.
    def add_strassen_slice(offset, b_col_start):
        # Strassen variables M1..M7
        # A indices: 0:A00, 1:A01, 2:A10, 3:A11
        # B slice indices: 0:B00, 1:B01, 2:B10, 3:B11 (local to the slice, mapped to global V)
        # C slice indices: 0:C00, 1:C01, 2:C10, 3:C11 (local to the slice, mapped to global W)
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0, offset+0] += 1
        U[3, offset+0] += 1
        V[b_col_start, offset+0] += 1      # B00
        V[b_col_start + p, offset+0] += 1   # B11
        
        # M2 = (A10 + A11) * B00
        U[2, offset+1] += 1
        U[3, offset+1] += 1
        V[b_col_start, offset+1] += 1      # B00
        
        # M3 = A00 * (B01 - B11)
        U[0, offset+2] += 1
        V[b_col_start + 1, offset+2] += 1      # B01
        V[b_col_start + p + 1, offset+2] += -1 # B11
        
        # M4 = A11 * (B10 - B00)
        U[3, offset+3] += 1
        V[b_col_start + p, offset+3] += 1      # B10
        V[b_col_start, offset+3] += -1         # B00
        
        # M5 = (A00 + A01) * B11
        U[0, offset+4] += 1
        U[1, offset+4] += 1
        V[b_col_start + p, offset+4] += 1      # B11
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[2, offset+5] += 1
        U[0, offset+5] += -1
        V[b_col_start, offset+5] += 1          # B00
        V[b_col_start + 1, offset+5] += 1      # B01
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1, offset+6] += 1
        U[3, offset+6] += -1
        V[b_col_start + p, offset+6] += 1      # B10
        V[b_col_start + p + 1, offset+6] += 1  # B11
        
        # C reconstruction for this slice
        # C00 = M1 + M4 - M5 + M7
        W[b_col_start*n + 0, offset+0] += 1
        W[b_col_start*n + 0, offset+3] += 1
        W[b_col_start*n + 0, offset+4] += -1
        W[b_col_start*n + 0, offset+6] += 1
        
        # C01 = M3 + M5
        W[b_col_start*n + 1, offset+2] += 1
        W[b_col_start*n + 1, offset+4] += 1
        
        # C10 = M2 + M4
        W[b_col_start*n + 1, offset+1] += 1 # Wait, C10 corresponds to row 1, col 0 of C slice?
        # Let's check indices carefully.
        # C is p x n = 5 x 2.
        # W index is k*n + i.
        # For slice columns k0, k1:
        # C[k0, 0] is W[k0*n + 0]
        # C[k0, 1] is W[k0*n + 1]
        # C[k1, 0] is W[k1*n + 0]
        # C[k1, 1] is W[k1*n + 1]
        
        # My previous comments in the draft were slightly mixed. Let's re-verify Strassen C formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Mapping to W indices for slice starting at column `b_col_start`:
        # Let k0 = b_col_start, k1 = b_col_start + 1
        # C[k0, 0] -> W[k0*n + 0]
        # C[k0, 1] -> W[k0*n + 1]
        # C[k1, 0] -> W[k1*n + 0]
        # C[k1, 1] -> W[k1*n + 1]
        
        # Correction: The previous code block in the prompt had:
        # M2: W[k0*n + 1] and W[k1*n + 1] with -1?
        # Let's re-derive standard Strassen C reconstruction to be safe.
        
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for slice columns k0, k1:
        # W[k0*n + 0] (C00) gets: M1, M4, -M5, M7
        # W[k0*n + 1] (C01) gets: M3, M5
        # W[k1*n + 0] (C10) gets: M2, M4  <-- WAIT. C10 is row 1, col 0 of the RESULT.
        # In matrix multiplication C = A*B, C[i][j] = sum_k A[i][k]B[k][j].
        # Strassen computes 2x2 result.
        # Result[0][0] = C00. Result[0][1] = C01.
        # Result[1][0] = C10. Result[1][1] = C11.
        
        # So:
        # W[k0*n + 0] corresponds to C[k0][0]? No.
        # The slice of B is 2 rows (from A's perspective, A is 2x2, so it multiplies 2 rows of B).
        # Wait, A is 2x2. B is 2x5.
        # C = A * B. C is 2x5.
        # C[i][j] = A[0][j-2?] No.
        # C[i][j] = sum_{k=0}^{1} A[i][k] B[k][j].
        # The "Strassen block" operates on a 2x2 submatrix of B.
        # Let the submatrix of B be B_sub with rows 0,1 and columns c, c+1.
        # Let the submatrix of A be the full A.
        # The result submatrix of C will have rows 0,1 and columns c, c+1.
        
        # So:
        # C[0][c] = C00 of the Strassen result.
        # C[0][c+1] = C01 of the Strassen result.
        # C[1][c] = C10 of the Strassen result.
        # C[1][c+1] = C11 of the Strassen result.
        
        # Mapping to W indices (W[k*n + i]):
        # C[0][c] -> W[c*n + 0]
        # C[0][c+1] -> W[(c+1)*n + 0]
        # C[1][c] -> W[c*n + 1]
        # C[1][c+1] -> W[(c+1)*n + 1]
        
        # Let's re-map the Strassen outputs to these W indices.
        # C00 (Strassen) = C[0][c] -> W[c*n + 0]
        # C01 (Strassen) = C[0][c+1] -> W[(c+1)*n + 0]
        # C10 (Strassen) = C[1][c] -> W[c*n + 1]
        # C11 (Strassen) = C[1][c+1] -> W[(c+1)*n + 1]
        
        # Now apply the coefficients:
        
        # M1:
        # C00 += M1 -> W[c*n + 0] += 1
        # C11 += M1 -> W[(c+1)*n + 1] += 1
        
        # M2:
        # C10 += M2 -> W[c*n + 1] += 1
        # C11 -= M2 -> W[(c+1)*n + 1] += -1
        
        # M3:
        # C01 += M3 -> W[(c+1)*n + 0] += 1
        # C11 += M3 -> W[(c+1)*n + 1] += 1
        
        # M4:
        # C00 += M4 -> W[c*n + 0] += 1
        # C10 += M4 -> W[c*n + 1] += 1
        
        # M5:
        # C00 -= M5 -> W[c*n + 0] += -1
        # C01 += M5 -> W[(c+1)*n + 0] += 1
        
        # M6:
        # C11 += M6 -> W[(c+1)*n + 1] += 1
        
        # M7:
        # C00 += M7 -> W[c*n + 0] += 1
        
        # Let's rewrite the helper function correctly with this mapping.
        # I will delete the previous incorrect assignments in the helper and replace them.
        
    # Clear the matrices to start fresh with the corrected logic
    U[:] = 0
    V[:] = 0
    W[:] = 0

    def add_strassen_correct(offset, c_start):
        # c_start is the column index in B and C
        c0 = c_start
        c1 = c_start + 1
        
        # U (A) terms are same as before
        U[0, offset+0] += 1; U[3, offset+0] += 1 # M1
        U[2, offset+1] += 1; U[3, offset+1] += 1 # M2
        U[0, offset+2] += 1                       # M3
        U[3, offset+3] += 1                       # M4
        U[0, offset+4] += 1; U[1, offset+4] += 1 # M5
        U[2, offset+5] += 1; U[0, offset+5] += -1# M6
        U[1, offset+6] += 1; U[3, offset+6] += -1# M7
        
        # V (B) terms:
        # M1: B00+B11
        V[c0, offset+0] += 1
        V[c1, offset+0] += 1
        # M2: B00
        V[c0, offset+1] += 1
        # M3: B01-B11
        V[c1, offset+2] += 1
        V[c0+p, offset+2] += -1 # Wait, B is 2x5. Rows are 0,1. Cols are 0..4.
        # V index is j*p + k.
        # B[0][c0] -> V[0*p + c0]
        # B[1][c1] -> V[1*p + c1]
        # M1: B[0][c0] + B[1][c1]
        # V[0*p + c0] += 1
        # V[1*p + c1] += 1
        
        # Let's redo V carefully.
        # B is 2x5. Index in V is row*5 + col.
        # Row 0: 0,1,2,3,4
        # Row 1: 5,6,7,8,9
        
        # M1: B[0][c0] + B[1][c1]
        V[0*p + c0, offset+0] += 1
        V[1*p + c1, offset+0] += 1
        
        # M2: B[0][c0]
        V[0*p + c0, offset+1] += 1
        
        # M3: B[0][c1] - B[1][c1]
        V[0*p + c1, offset+2] += 1
        V[1*p + c1, offset+2] += -1
        
        # M4: B[1][c0] - B[0][c0]
        V[1*p + c0, offset+3] += 1
        V[0*p + c0, offset+3] += -1
        
        # M5: B[1][c1]
        V[1*p + c1, offset+4] += 1
        
        # M6: B[0][c0] + B[0][c1]
        V[0*p + c0, offset+5] += 1
        V[0*p + c1, offset+5] += 1
        
        # M7: B[1][c0] + B[1][c1]
        V[1*p + c0, offset+6] += 1
        V[1*p + c1, offset+6] += 1
        
        # W (C) terms:
        # C is 2x5. Index in W is col*2 + row.
        # Col c0: indices 2*c0, 2*c0+1
        # Col c1: indices 2*c1, 2*c1+1
        
        # M1:
        # C[0][c0] += M1 -> W[2*c0 + 0] += 1
        # C[1][c1] += M1 -> W[2*c1 + 1] += 1
        W[c0*n + 0, offset+0] += 1
        W[c1*n + 1, offset+0] += 1
        
        # M2:
        # C[1][c0] += M2 -> W[2*c0 + 1] += 1
        # C[1][c1] -= M2 -> W[2*c1 + 1] += -1
        W[c0*n + 1, offset+1] += 1
        W[c1*n + 1, offset+1] += -1
        
        # M3:
        # C[0][c1] += M3 -> W[2*c1 + 0] += 1
        # C[1][c1] += M3 -> W[2*c1 + 1] += 1
        W[c1*n + 0, offset+2] += 1
        W[c1*n + 1, offset+2] += 1
        
        # M4:
        # C[0][c0] += M4 -> W[2*c0 + 0] += 1
        # C[1][c0] += M4 -> W[2*c0 + 1] += 1
        W[c0*n + 0, offset+3] += 1
        W[c0*n + 1, offset+3] += 1
        
        # M5:
        # C[0][c0] -= M5 -> W[2*c0 + 0] += -1
        # C[0][c1] += M5 -> W[2*c1 + 0] += 1
        W[c0*n + 0, offset+4] += -1
        W[c1*n + 0, offset+4] += 1
        
        # M6:
        # C[1][c1] += M6 -> W[2*c1 + 1] += 1
        W[c1*n + 1, offset+5] += 1
        
        # M7:
        # C[0][c0] += M7 -> W[2*c0 + 0] += 1
        W[c0*n + 0, offset+6] += 1

    # Add Strassen block for columns 0, 1
    add_strassen_correct(0, 0)
    
    # Add Strassen block for columns 2, 3
    add_strassen_correct(7, 2)
    
    # Add Schoolbook for column 4
    # C[0][4] = A[0][0]*B[0][4] + A[0][1]*B[1][4]
    # C[1][4] = A[1][0]*B[0][4] + A[1][1]*B[1][4]
    
    # Term 1: A[0][0]*B[0][4] -> C[0][4]
    U[0, 14] += 1
    V[0*p + 4, 14] += 1
    W[4*n + 0, 14] += 1
    
    # Term 2: A[0][1]*B[1][4] -> C[0][4]
    U[1, 15] += 1
    V[1*p + 4, 15] += 1
    W[4*n + 0, 15] += 1
    
    # Term 3: A[1][0]*B[0][4] -> C[1][4]
    U[2, 16] += 1
    V[0*p + 4, 16] += 1
    W[4*n + 1, 16] += 1
    
    # Term 4: A[1][1]*B[1][4] -> C[1][4]
    U[3, 17] += 1
    V[1*p + 4, 17] += 1
    W[4*n + 1, 17] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
