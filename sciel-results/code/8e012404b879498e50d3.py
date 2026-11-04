"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x8 matrix.
    
    Strategy: Block the 2x3 matrix A into a 2x2 block A_left (cols 0,1) and
    a 2x1 block A_right (col 2). Similarly block the 3x8 matrix B into 3x2
    blocks. For the 2x2 part, use Strassen's algorithm (7 multiplications per
    3x2 B-block). For the 2x1 part, use 2 multiplications per 3x2 B-block.
    Total: (7+2) * 4 blocks = 36 rank.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    
    # We'll build U, V, W as lists of lists initially, then convert
    # U shape: (n*m, rank) = (6, 36)
    # V shape: (m*p, rank) = (24, 36)
    # W shape: (p*n, rank) = (16, 36)
    
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # A is 2x3: indices (i, j) where i in [0,1], j in [0,1,2]
    # U row index: i * m + j = i * 3 + j
    # So: A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2]
    #     A[1,0]=U[3], A[1,1]=U[4], A[1,2]=U[5]
    
    # B is 3x8: indices (j, k) where j in [0,1,2], k in [0,1,2,3,4,5,6,7]
    # V row index: j * p + k = j * 8 + k
    # So: B[j,k] = V[j*8+k]
    # For block b (b in 0..3), k = 2*b and k = 2*b+1
    # B[0, 2b]   = V[0*8 + 2b]   = V[2b]
    # B[0, 2b+1] = V[0*8 + 2b+1] = V[2b+1]
    # B[1, 2b]   = V[1*8 + 2b]   = V[8+2b]
    # B[1, 2b+1] = V[1*8 + 2b+1] = V[8+2b+1]
    # B[2, 2b]   = V[2*8 + 2b]   = V[16+2b]
    # B[2, 2b+1] = V[2*8 + 2b+1] = V[16+2b+1]
    
    # C is 2x8: indices (i, k) where i in [0,1], k in [0,1,2,3,4,5,6,7]
    # W row index: k * n + i = k * 2 + i
    # So: C[i,k] = W[k*2+i]
    # For block b, k = 2b and k = 2b+1:
    # C[0, 2b]   = W[2b*2+0] = W[4b]
    # C[0, 2b+1] = W[2b*2+1] = W[4b+1]
    # C[1, 2b]   = W[2b*2+0] = W[4b]    -- wait, k*2+i, i=1: W[2b*2+1]? No.
    # Let me redo: W[k*n + i], n=2, so W[k*2 + i]
    # C[0, 2b] = W[2b*2 + 0] = W[4b]
    # C[0, 2b+1] = W[(2b+1)*2 + 0] = W[4b+2]
    # C[1, 2b] = W[2b*2 + 1] = W[4b+1]
    # C[1, 2b+1] = W[(2b+1)*2 + 1] = W[4b+3]
    
    # Let me verify with the schoolbook:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # W[k*n+i, c] contributes to C[i,k]
    # For schoolbook: each (i,j,k) triple gets one column
    # W[k*n + i] = 1 for that column
    # So yes: C[0,2b] = W[4b], C[0,2b+1] = W[4b+2], C[1,2b] = W[4b+1], C[1,2b+1] = W[4b+3]
    
    # Now, Strassen for 2x2 * 3x2:
    # A_left = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] = [[a, b], [c, d]]
    #   where a=U[0], b=U[1], c=U[3], d=U[4]
    # B_block = 3x2 matrix with rows B[0,2b], B[0,2b+1], B[1,2b], B[1,2b+1], B[2,2b], B[2,2b+1]
    #   Wait, B is 3x8, so for block b, the 3x2 sub-block is:
    #   Row 0: B[0, 2b], B[0, 2b+1]
    #   Row 1: B[1, 2b], B[1, 2b+1]
    #   Row 2: B[2, 2b], B[2, 2b+1]
    # But A_left is 2x2, so it only uses rows 0 and 1 of B_block.
    # The product A_left * B_block[0:2,:] gives a 2x2 matrix contributing to C[:, 2b:2b+2].
    # The remaining contribution comes from A_right (2x1) * B_block[2,:] (1x2).
    
    # Let me denote for block b:
    # x0 = B[0, 2b]   = V[2b]
    # x1 = B[0, 2b+1] = V[2b+1]
    # y0 = B[1, 2b]   = V[8+2b]
    # y1 = B[1, 2b+1] = V[8+2b+1]
    # z0 = B[2, 2b]   = V[16+2b]
    # z1 = B[2, 2b+1] = V[16+2b+1]
    
    # Strassen 7 products for [[a,b],[c,d]] * [[x0,x1],[y0,y1]]:
    # M1 = (a+d)*(x0+y0)
    # M2 = (c+d)*(x0+y0)
    # M3 = (a-b)*(y1+x1)
    # M4 = (c+d)*(y1-y0)
    # M5 = (a+d)*(x1-y1)
    # M6 = (b+a)*(y1-x1)  -- wait, let me use standard Strassen
    # Standard Strassen:
    # M1 = (a+d)*(e+g)  where e,g are top row of B -> e=x0, g=x1? No.
    # Let me be very careful.
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+g)
    # M2 = (c+d)(e+g)
    # M3 = (a-b)(f+h)
    # M4 = (c+d)(h-g)
    # M5 = (a+d)(e-f)
    # M6 = (b+d)(g-h)
    # M7 = (a-b)(f-e)
    # C11 = M1 + M4 - M2 + M5
    # C12 = M3 + M7
    # C21 = M2 + M6
    # C22 = M1 - M5 + M6 - M3
    
    # In our case: e=x0, f=x1, g=y0, h=y1
    # M1 = (a+d)(x0+y0)
    # M2 = (c+d)(x0+y0)
    # M3 = (a-b)(x1+y1)
    # M4 = (c+d)(y1-y0)
    # M5 = (a+d)(x0-x1)
    # M6 = (b+d)(y0-y1)
    # M7 = (a-b)(x1-x0)
    
    # C[0,2b]   = M1 + M4 - M2 + M5
    # C[0,2b+1] = M3 + M7
    # C[1,2b]   = M2 + M6
    # C[1,2b+1] = M1 - M5 + M6 - M3
    
    # Plus outer product contribution:
    # A_right = [A[0,2], A[1,2]] = [U[2], U[5]]
    # B_row2 = [z0, z1] = [V[16+2b], V[16+2b+1]]
    # C[0,2b]   += U[2] * z0
    # C[0,2b+1] += U[2] * z1
    # C[1,2b]   += U[5] * z0
    # C[1,2b+1] += U[5] * z1
    
    # So for each block b (0,1,2,3), we have 9 columns:
    # Columns 9b+0 through 9b+8
    
    col = 0
    for b in range(4):
        # Strassen products
        # M1 = (a+d)(x0+y0): U: a=U[0]=1, d=U[4]=1; V: x0=V[2b]=1, y0=V[8+2b]=1
        U[0][col] = 1.0
        U[4][col] = 1.0
        V[2*b][col] = 1.0
        V[8+2*b][col] = 1.0
        W[4*b][col] = 1.0      # C[0,2b] += M1
        W[4*b+1][col] = 1.0    # C[1,2b] += M1
        col += 1
        
        # M2 = (c+d)(x0+y0): U: c=U[3]=1, d=U[4]=1; V: x0=V[2b]=1, y0=V[8+2b]=1
        U[3][col] = 1.0
        U[4][col] = 1.0
        V[2*b][col] = 1.0
        V[8+2*b][col] = 1.0
        W[4*b][col] = -1.0     # C[0,2b] -= M2
        W[4*b+1][col] = 1.0    # C[1,2b] += M2
        col += 1
        
        # M3 = (a-b)(x1+y1): U: a=U[0]=1, b=U[1]=-1; V: x1=V[2b+1]=1, y1=V[8+2b+1]=1
        U[0][col] = 1.0
        U[1][col] = -1.0
        V[2*b+1][col] = 1.0
        V[8+2*b+1][col] = 1.0
        W[4*b+2][col] = 1.0    # C[0,2b+1] += M3
        W[4*b+3][col] = -1.0   # C[1,2b+1] -= M3
        col += 1
        
        # M4 = (c+d)(y1-y0): U: c=U[3]=1, d=U[4]=1; V: y1=V[8+2b+1]=1, y0=V[8+2b]=-1
        U[3][col] = 1.0
        U[4][col] = 1.0
        V[8+2*b][col] = -1.0
        V[8+2*b+1][col] = 1.0
        W[4*b][col] = 1.0      # C[0,2b] += M4
        col += 1
        
        # M5 = (a+d)(x0-x1): U: a=U[0]=1, d=U[4]=1; V: x0=V[2b]=1, x1=V[2b+1]=-1
        U[0][col] = 1.0
        U[4][col] = 1.0
        V[2*b][col] = 1.0
        V[2*b+1][col] = -1.0
        W[4*b][col] = 1.0      # C[0,2b] += M5
        W[4*b+3][col] = -1.0   # C[1,2b+1] -= M5
        col += 1
        
        # M6 = (b+d)(y0-y1): U: b=U[1]=1, d=U[4]=1; V: y0=V[8+2b]=1, y1=V[8+2b+1]=-1
        U[1][col] = 1.0
        U[4][col] = 1.0
        V[8+2*b][col] = 1.0
        V[8+2*b+1][col] = -1.0
        W[4*b+1][col] = 1.0    # C[1,2b] += M6
        W[4*b+3][col] = 1.0    # C[1,2b+1] += M6
        col += 1
        
        # M7 = (a-b)(x1-x0): U: a=U[0]=1, b=U[1]=-1; V: x1=V[2b+1]=1, x0=V[2b]=-1
        U[0][col] = 1.0
        U[1][col] = -1.0
        V[2*b][col] = -1.0
        V[2*b+1][col] = 1.0
        W[4*b+2][col] = 1.0    # C[0,2b+1] += M7
        col += 1
        
        # Outer product: U[2] * z0 -> C[0,2b]
        U[2][col] = 1.0
        V[16+2*b][col] = 1.0
        W[4*b][col] = 1.0
        col += 1
        
        # Outer product: U[2] * z1 -> C[0,2b+1]
        U[2][col] = 1.0
        V[16+2*b+1][col] = 1.0
        W[4*b+2][col] = 1.0
        col += 1
        
        # Outer product: U[5] * z0 -> C[1,2b]
        U[5][col] = 1.0
        V[16+2*b][col] = 1.0
        W[4*b+1][col] = 1.0
        col += 1
        
        # Outer product: U[5] * z1 -> C[1,2b+1]
        U[5][col] = 1.0
        V[16+2*b+1][col] = 1.0
        W[4*b+3][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
