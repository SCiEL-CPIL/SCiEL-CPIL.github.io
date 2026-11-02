import numpy as np
import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 decomposition of the (2,2,8) matrix multiplication tensor.
    
    The key insight: multiplying a 2x2 matrix A by a 2x8 matrix B is equivalent
    to multiplying A by four independent 2x2 blocks of B. Each 2x2x2 matrix 
    multiplication can be done with rank 7 (Strassen's algorithm), so the total 
    rank is 4 * 7 = 28, which is better than the schoolbook rank of 32.
    
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    rank = 28  # 4 blocks * 7 Strassen multiplications per block
    
    # U shape: (n*m, rank) = (4, 28)
    # V shape: (m*p, rank) = (16, 28)  
    # W shape: (p*n, rank) = (16, 28)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # Given A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # The 7 products are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Result:
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # For each of the 4 blocks (k_block from 0 to 3), we apply Strassen
    for k_block in range(4):
        # Column offset for this block
        col_offset = k_block * 7
        
        # U factors: extract the relevant entries of A for each Strassen product
        # a = A[0,0] -> row 0 of U (i*m+j = 0*2+0 = 0)
        # b = A[0,1] -> row 1 of U (i*m+j = 0*2+1 = 1)
        # c = A[1,0] -> row 2 of U (i*m+j = 1*2+0 = 2)
        # d = A[1,1] -> row 3 of U (i*m+j = 1*2+1 = 3)
        
        # M1 = (a+d)(e+h): U rows 0,1 get coefficients 1,1
        U[0, col_offset + 0] = 1
        U[3, col_offset + 0] = 1
        
        # M2 = (c+d)e: U rows 2,3 get coefficients 1,1
        U[2, col_offset + 1] = 1
        U[3, col_offset + 1] = 1
        
        # M3 = a(f-h): U row 0 gets coefficient 1
        U[0, col_offset + 2] = 1
        
        # M4 = d(g-e): U row 3 gets coefficient 1
        U[3, col_offset + 3] = 1
        
        # M5 = (a+b)h: U rows 0,1 get coefficients 1,1
        U[0, col_offset + 4] = 1
        U[1, col_offset + 4] = 1
        
        # M6 = (c-a)(e+f): U rows 0,2 get coefficients -1,1
        U[0, col_offset + 5] = -1
        U[2, col_offset + 5] = 1
        
        # M7 = (b-d)(g+h): U rows 1,3 get coefficients 1,-1
        U[1, col_offset + 6] = 1
        U[3, col_offset + 6] = -1
        
        # V factors: extract the relevant entries of B for each Strassen product
        # For block k_block, the B entries are:
        # e = B[0, 2*k_block] -> row 0*8 + 2*k_block of V (j*p+k = 0*8+2*k_block)
        # f = B[0, 2*k_block+1] -> row 0*8 + 2*k_block+1 of V
        # g = B[1, 2*k_block] -> row 1*8 + 2*k_block of V
        # h = B[1, 2*k_block+1] -> row 1*8 + 2*k_block+1 of V
        
        # M1 = (a+d)(e+h): V rows for e,h get coefficients 1,1
        V[0 * m + 2 * k_block, col_offset + 0] = 1
        V[1 * m + 2 * k_block + 1, col_offset + 0] = 1
        
        # M2 = (c+d)e: V row for e gets coefficient 1
        V[0 * m + 2 * k_block, col_offset + 1] = 1
        
        # M3 = a(f-h): V rows for f,h get coefficients 1,-1
        V[0 * m + 2 * k_block + 1, col_offset + 2] = 1
        V[1 * m + 2 * k_block + 1, col_offset + 2] = -1
        
        # M4 = d(g-e): V rows for g,e get coefficients 1,-1
        V[1 * m + 2 * k_block, col_offset + 3] = 1
        V[0 * m + 2 * k_block, col_offset + 3] = -1
        
        # M5 = (a+b)h: V row for h gets coefficient 1
        V[1 * m + 2 * k_block + 1, col_offset + 4] = 1
        
        # M6 = (c-a)(e+f): V rows for e,f get coefficients 1,1
        V[0 * m + 2 * k_block, col_offset + 5] = 1
        V[0 * m + 2 * k_block + 1, col_offset + 5] = 1
        
        # M7 = (b-d)(g+h): V rows for g,h get coefficients 1,1
        V[1 * m + 2 * k_block, col_offset + 6] = 1
        V[1 * m + 2 * k_block + 1, col_offset + 6] = 1
        
        # W factors: reconstruct the output C entries from the Strassen products
        # For block k_block, the output C entries are at positions:
        # C[0, 2*k_block] = z00
        # C[0, 2*k_block+1] = z01
        # C[1, 2*k_block] = z10
        # C[1, 2*k_block+1] = z11
        
        # Using W[k*n+i, r] convention:
        # z00 = C[0, 2*k_block] -> W[(2*k_block)*n + 0, r]
        # z01 = C[0, 2*k_block+1] -> W[(2*k_block+1)*n + 0, r]
        # z10 = C[1, 2*k_block] -> W[(2*k_block)*n + 1, r]
        # z11 = C[1, 2*k_block+1] -> W[(2*k_block+1)*n + 1, r]
        
        # z00 = M1 + M4 - M5 + M7
        W[(2 * k_block) * n + 0, col_offset + 0] = 1   # M1
        W[(2 * k_block) * n + 0, col_offset + 3] = 1   # M4
        W[(2 * k_block) * n + 0, col_offset + 4] = -1  # -M5
        W[(2 * k_block) * n + 0, col_offset + 6] = 1   # M7
        
        # z01 = M3 + M5
        W[(2 * k_block + 1) * n + 0, col_offset + 2] = 1  # M3
        W[(2 * k_block + 1) * n + 0, col_offset + 4] = 1  # M5
        
        # z10 = M2 + M4
        W[(2 * k_block) * n + 1, col_offset + 1] = 1  # M2
        W[(2 * k_block) * n + 1, col_offset + 3] = 1  # M4
        
        # z11 = M1 - M2 + M3 + M6
        W[(2 * k_block + 1) * n + 1, col_offset + 0] = 1   # M1
        W[(2 * k_block + 1) * n + 1, col_offset + 1] = -1  # -M2
        W[(2 * k_block + 1) * n + 1, col_offset + 2] = 1   # M3
        W[(2 * k_block + 1) * n + 1, col_offset + 5] = 1   # M6
    
    # Convert to list of lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
