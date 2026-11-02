import time
import sys

DIMENSIONS = [2, 2, 7]

def construct():
    """
    Returns a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    The decomposition splits the 7 columns of the B matrix into three 2-column blocks
    (handled by Strassen rank-7) and one 1-column block (handled by schoolbook rank-4).
    
    Convention:
      U: (n*m, rank) = (4, 25)
      V: (m*p, rank) = (14, 25)
      W: (p*n, rank) = (14, 25) with W[k*n + i, r]
      
    The tensor equation is:
      C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    which in index notation for the full tensor T[i,j,k] = delta (trivially, the multiplication tensor has T[i,j,k]=1 if we're computing C[i,k]=sum A[i,j]B[j,k])
    
    Wait, the standard matrix multiplication tensor M_{n,m,p} has entries T[i,j,k] where:
      C[i,k] = sum_j A[i,j] B[j,k]
    The tensor is T[i,j,k] = 1 for all i,j,k? No.
    The tensor T has components T[i,j,k] such that the multiplication is:
      C = A * B means C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    So T[i,j,k] = 1 for all valid i,j,k.
    
    The decomposition T = sum_r U_r V_r W_r where:
      U_r[i*m+j] = coefficient for A[i,j]
      V_r[j*p+k] = coefficient for B[j,k]
      W_r[k*n+i] = coefficient for C[i,k]
      
    For Strassen on 2x2 blocks:
      Let A_block = A[i0:i0+2, j0:j0+2], B_block = B[j0:j0+2, k0:k0+2], C_block = C[i0:i0+2, k0:k0+2]
      
      Strassen uses 7 products M1..M7.
      For each product Mr, we have:
        Mr = (linear combo of A entries) * (linear combo of B entries)
        Then C entries are linear combos of M1..M7.
        
      So for rank 7, we have 7 columns r=0..6.
      U[r] has nonzero entries only at positions (i*m+j) for i in {i0,i0+1}, j in {j0,j0+1}
      V[r] has nonzero entries only at positions (j*p+k) for j in {j0,j0+1}, k in {k0,k0+1}
      W[r] has nonzero entries only at positions (k*n+i) for k in {k0,k0+1}, i in {i0,i0+1}
    """
    n, m, p = DIMENSIONS
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 25)
    V = [[0] * rank for _ in range(m * p)]  # (14, 25)
    W = [[0] * rank for _ in range(p * n)]  # (14, 25)
    
    # Strassen coefficients for 2x2x2:
    # M1 = A00 * (B00 + B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A01 + A11) * (B00 + B01)
    # M4 = A11 * (B10 + B11)
    # M5 = (A00 + A01) * B00
    # M6 = (A10 - A00) * (B01 + B11)
    # M7 = (A01 - A10) * (B00 + B10)
    #
    # C00 = M1 + M4 - M2 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    
    def strassen_block(i0, j0, k0, col_offset):
        """Add Strassen decomposition for a 2x2x2 block starting at (i0, j0, k0)."""
        # 7 columns for this block
        # Column 0: M1
        # Column 1: M2
        # Column 2: M3
        # Column 3: M4
        # Column 4: M5
        # Column 5: M6
        # Column 6: M7
        
        # M1 = A00 * (B00 + B11)
        # A00 -> i=i0, j=j0 -> U index: i0*m + j0
        # B00 -> j=j0, k=k0 -> V index: j0*p + k0
        # B11 -> j=j0+1, k=k0+1 -> V index: (j0+1)*p + (k0+1)
        U[i0*m + j0][col_offset + 0] = 1
        V[j0*p + k0][col_offset + 0] = 1
        V[(j0+1)*p + (k0+1)][col_offset + 0] = 1
        # W for M1: contributes to C00 (+1), C11 (+1)
        # C00 -> i=i0, k=k0 -> W index: k0*n + i0
        # C11 -> i=i0+1, k=k0+1 -> W index: (k0+1)*n + (i0+1)
        W[k0*n + i0][col_offset + 0] = 1
        W[(k0+1)*n + (i0+1)][col_offset + 0] = 1
        
        # M2 = (A00 + A10) * B11
        # A00 -> i0*m + j0
        # A10 -> i0+1, j0 -> (i0+1)*m + j0
        # B11 -> (j0+1)*p + (k0+1)
        U[i0*m + j0][col_offset + 1] = 1
        U[(i0+1)*m + j0][col_offset + 1] = 1
        V[(j0+1)*p + (k0+1)][col_offset + 1] = 1
        # W for M2: contributes to C00 (-1), C10 (+1)
        W[k0*n + i0][col_offset + 1] = -1
        W[k0*n + (i0+1)][col_offset + 1] = 1  # C10: i=i0+1, k=k0
        
        # M3 = (A01 + A11) * (B00 + B01)
        # A01 -> i0, j0+1 -> i0*m + (j0+1)
        # A11 -> i0+1, j0+1 -> (i0+1)*m + (j0+1)
        # B00 -> j0*p + k0
        # B01 -> j0*p + (k0+1)
        U[i0*m + (j0+1)][col_offset + 2] = 1
        U[(i0+1)*m + (j0+1)][col_offset + 2] = 1
        V[j0*p + k0][col_offset + 2] = 1
        V[j0*p + (k0+1)][col_offset + 2] = 1
        # W for M3: contributes to C01 (+1), C11 (-1)
        W[(k0+1)*n + i0][col_offset + 2] = 1  # C01: i=i0, k=k0+1
        W[(k0+1)*n + (i0+1)][col_offset + 2] = -1  # C11: i=i0+1, k=k0+1
        
        # M4 = A11 * (B10 + B11)
        # A11 -> (i0+1)*m + (j0+1)
        # B10 -> (j0+1)*p + k0
        # B11 -> (j0+1)*p + (k0+1)
        U[(i0+1)*m + (j0+1)][col_offset + 3] = 1
        V[(j0+1)*p + k0][col_offset + 3] = 1
        V[(j0+1)*p + (k0+1)][col_offset + 3] = 1
        # W for M4: contributes to C00 (+1), C10 (+1)
        W[k0*n + i0][col_offset + 3] = 1
        W[k0*n + (i0+1)][col_offset + 3] = 1
        
        # M5 = (A00 + A01) * B00
        # A00 -> i0*m + j0
        # A01 -> i0*m + (j0+1)
        # B00 -> j0*p + k0
        U[i0*m + j0][col_offset + 4] = 1
        U[i0*m + (j0+1)][col_offset + 4] = 1
        V[j0*p + k0][col_offset + 4] = 1
        # W for M5: contributes to C01 (+1), C11 (+1)
        W[(k0+1)*n + i0][col_offset + 4] = 1
        W[(k0+1)*n + (i0+1)][col_offset + 4] = 1
        
        # M6 = (A10 - A00) * (B01 + B11)
        # A10 -> (i0+1)*m + j0
        # A00 -> i0*m + j0
        # B01 -> j0*p + (k0+1)
        # B11 -> (j0+1)*p + (k0+1)
        U[(i0+1)*m + j0][col_offset + 5] = 1
        U[i0*m + j0][col_offset + 5] = -1
        V[j0*p + (k0+1)][col_offset + 5] = 1
        V[(j0+1)*p + (k0+1)][col_offset + 5] = 1
        # W for M6: contributes to C00 (+1)
        W[k0*n + i0][col_offset + 5] = 1
        
        # M7 = (A01 - A10) * (B00 + B10)
        # A01 -> i0*m + (j0+1)
        # A10 -> (i0+1)*m + j0
        # B00 -> j0*p + k0
        # B10 -> (j0+1)*p + k0
        U[i0*m + (j0+1)][col_offset + 6] = 1
        U[(i0+1)*m + j0][col_offset + 6] = -1
        V[j0*p + k0][col_offset + 6] = 1
        V[(j0+1)*p + k0][col_offset + 6] = 1
        # W for M7: contributes to C11 (+1)
        W[(k0+1)*n + (i0+1)][col_offset + 6] = 1
    
    # Schoolbook for 2x2x1: rank 4
    def schoolbook_2x2x1(i0, j0, k0, col_offset):
        """Add schoolbook decomposition for a 2x2 (A) by 1-column (B) block."""
        # C[i, k0] = sum_j A[i,j] B[j,k0] for i in {i0, i0+1}, j in {j0, j0+1}
        # 4 terms:
        # Term 0: A[i0,j0] * B[j0,k0] -> C[i0,k0]
        # Term 1: A[i0,j0+1] * B[j0+1,k0] -> C[i0,k0]
        # Term 2: A[i0+1,j0] * B[j0,k0] -> C[i0+1,k0]
        # Term 3: A[i0+1,j0+1] * B[j0+1,k0] -> C[i0+1,k0]
        
        # Term 0
        U[i0*m + j0][col_offset + 0] = 1
        V[j0*p + k0][col_offset + 0] = 1
        W[k0*n + i0][col_offset + 0] = 1
        
        # Term 1
        U[i0*m + (j0+1)][col_offset + 1] = 1
        V[(j0+1)*p + k0][col_offset + 1] = 1
        W[k0*n + i0][col_offset + 1] = 1
        
        # Term 2
        U[(i0+1)*m + j0][col_offset + 2] = 1
        V[j0*p + k0][col_offset + 2] = 1
        W[k0*n + (i0+1)][col_offset + 2] = 1
        
        # Term 3
        U[(i0+1)*m + (j0+1)][col_offset + 3] = 1
        V[(j0+1)*p + k0][col_offset + 3] = 1
        W[k0*n + (i0+1)][col_offset + 3] = 1
    
    # Block 1: columns 0-1 of B, i.e., k0=0, k0+1=1
    # i0=0, j0=0, k0=0
    strassen_block(0, 0, 0, 0)
    
    # Block 2: columns 2-3 of B, i.e., k0=2, k0+1=3
    # i0=0, j0=0, k0=2
    strassen_block(0, 0, 2, 7)
    
    # Block 3: columns 4-5 of B, i.e., k0=4, k0+1=5
    # i0=0, j0=0, k0=4
    strassen_block(0, 0, 4, 14)
    
    # Block 4: column 6 of B, i.e., k0=6
    # i0=0, j0=0, k0=6
    schoolbook_2x2x1(0, 0, 6, 21)
    
    return {"U": U, "V": V, "W": W}
