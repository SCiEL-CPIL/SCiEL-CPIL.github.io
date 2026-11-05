"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build a rank-minimal decomposition using Strassen-style blocks
    # Strategy: Use 2x2 Strassen blocks where possible
    # For 2x4 * 4x7:
    # Split A into [A11, A12] where each is 2x2
    # Split B into [B1; B2] where B1 is 4x3, B2 is 4x4 (or better: B1 4x4, B2 4x3)
    # Actually, let's be more systematic.
    
    # A is 2x4: A = [A1 | A2] where A1 is 2x2, A2 is 2x2
    # B is 4x7: B = [B1; B2] where B1 is 2x7, B2 is 2x7
    # C = A1*B1 + A2*B2, where C is 2x7
    
    # For A1*B1: 2x2 * 2x7, use Strassen-like
    # For A2*B2: 2x2 * 2x7, use Strassen-like
    
    # Actually, let's think about this differently.
    # We can view this as computing C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's use a block decomposition:
    # A = [[A00, A01], [A10, A11]] where each is 1x2
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13], [B20, B21, B22, B23], [B30, B31, B32, B33]]
    
    # This is getting complex. Let me use the standard approach:
    # Split into 2x2 blocks where possible.
    
    # A (2x4) = [A_left (2x2) | A_right (2x2)]
    # B (4x7) = [B_top (2x7); B_bottom (2x7)]
    # C (2x7) = A_left @ B_top + A_right @ B_bottom
    
    # Now for each 2x2 * 2x7 multiplication, we can use a Strassen-like approach.
    # 2x2 * 2x7: 
    #   M1 = (A00+A11)*(B00+B10)  -- but B has 7 columns...
    
    # Let me think more carefully. For 2x2 * 2xk:
    # C = [A00 A01; A10 A11] @ [B0; B1] where B0, B1 are 1xk
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # We can use:
    # M1 = (A00+A11)*(B0+B1)
    # M2 = (A10+A11)*B0
    # M3 = A00*(B1-B0)
    # M4 = (A01-A10)*(B0+B1)  -- wait, this doesn't quite work
    
    # Actually for 2x2 * 2xk, we can do it in 7 multiplications:
    # M1 = (A00+A11)*(B0+B1)
    # M2 = (A10+A11)*B0
    # M3 = A00*(B1-B0)
    # M4 = (A01-A10)*(B0+B1)  -- no, this mixes things up
    
    # Let me just use the schoolbook for the small blocks and see if we can optimize.
    
    # Actually, the cleanest approach:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a tensor of size 2x4x7, total 56 entries.
    # Schoolbook: 56 rank-1 terms (one for each i,j,k triple)
    # But we can do better with block structure.
    
    # Let's use the following block decomposition:
    # Split j into two groups: j in {0,1} and j in {2,3}
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    
    # First sum: 2x2 * 2x7 (A[:,0:2] @ B[0:2,:])
    # Second sum: 2x2 * 2x7 (A[:,2:4] @ B[2:4,:])
    
    # For each 2x2 * 2x7, we can use Strassen:
    # For matrices X (2x2) and Y (2x7):
    #   Z = X @ Y
    #   Z[0,k] = X[0,0]*Y[0,k] + X[0,1]*Y[1,k]
    #   Z[1,k] = X[1,0]*Y[0,k] + X[1,1]*Y[1,k]
    
    # Strassen for 2x2 * 2xk:
    #   M1 = (X00+X11)*(Y0+Y1)
    #   M2 = (X10+X11)*Y0
    #   M3 = X00*(Y1-Y0)
    #   M4 = (X01-X10)*(Y0+Y1)  -- this isn't right either
    
    # Let me just implement the direct approach with 2x2 blocks and count:
    # For 2x2 * 2x7: 4*7 = 28 schoolbook terms, or 7 Strassen terms
    # Two such blocks: 14 terms total with Strassen
    # Plus we need to handle the column split for the 7 columns.
    
    # Wait, I think I'm overcomplicating this. Let me just use the current 
    # rank-50 solution which is verified to work, and try to improve it.
    
    # The current solution uses:
    # 6 Strassen blocks (2x2*2x2 each) = 42 terms
    # 8 direct terms for k=6
    # Total: 50
    
    # Can we do better? Let's think about the structure.
    # 2x4 * 4x7:
    # We can split B into [B1 (4x4) | B2 (4x3)]
    # C = A @ B1 + A @ B2
    # A @ B1: 2x4 * 4x4
    # A @ B2: 2x4 * 4x3
    
    # For 2x4 * 4x4:
    # Split A = [A1 (2x2) | A2 (2x2)], B1 = [B1_top (2x4); B1_bot (2x4)]
    # A1 @ B1_top: 2x2 * 2x4 = 7 Strassen terms
    # A2 @ B1_bot: 2x2 * 2x4 = 7 Strassen terms
    # Total for A@B1: 14 terms
    
    # For 2x4 * 4x3:
    # Similarly: 14 terms
    # Total: 28 terms
    
    # That's much better! Let me implement this.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 28 x 28
    W = [[0] * rank for _ in range(p * n)]  # 14 x 28
    
    col = 0
    
    def add_strassen_2x2_2xk(i0, j0, k0, kk):
        """Add Strassen terms for A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+kk]"""
        nonlocal col
        # U indices: i*m + j
        # V indices: j*p + k  
        # W indices: k*n + i
        
        # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0])
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        col += 1
        
        # M3: A[i0,j0] * (B[j0+1,k0]-B[j0,k0])
        U[i0*m + j0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        col += 1
        
        # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0]
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
        
        # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0+1,k0])
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0,k0])
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
    
    # We need to handle the fact that B has 7 columns, not a power of 2.
    # Let's split the 7 columns into groups: 4 + 3
    # For each group, use Strassen on 2x2 * 2xk blocks.
    
    # Actually, let me reconsider the approach. The Strassen formula above
    # is for 2x2 * 2x1. For 2x2 * 2xk, we need to apply it k times or
    # find a better formula.
    
    # Let me just go with the verified rank-50 solution and return it.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Standard 2x2 * 2x2 Strassen
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
    
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
