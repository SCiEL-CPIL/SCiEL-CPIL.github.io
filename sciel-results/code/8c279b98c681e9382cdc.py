"""
Rank-minimal tensor decomposition for 2x2 * 2x6 matrix multiplication.

Strategy:
1. Start with a schoolbook decomposition of rank 24.
2. Use Strassen's algorithm on the 2x2 blocks to get a rank-21 decomposition.
   The 2x2 * 2x6 product can be viewed as three independent 2x2 * 2x2 multiplications
   (since the second matrix is 2x6 = 2x2 * 3 columns). Each 2x2 * 2x2 can be done
   with Strassen's 7 multiplications, giving 3 * 7 = 21 total rank.
3. Try to verify and potentially improve via search if time permits.

The output factor W follows the trace convention W[k*n+i, r] where the output
C[i,k] = sum_j A[i,j] * B[j,k].
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 100  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=6
    
    # Strategy: decompose the 2x2 * 2x6 multiplication into three 2x2 * 2x2 blocks.
    # B is 2x6, so we can view it as three 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    # Each block multiplication is 2x2 * 2x2 -> 2x2
    # Use Strassen's algorithm for each 2x2 * 2x2: rank 7 per block
    # Total rank: 3 * 7 = 21
    
    # Strassen's algorithm for 2x2 * 2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B where:
    # C[0,0] = p1 - p5 + p6 + p7
    # C[0,1] = p4 + p5
    # C[1,0] = p3 + p6
    # C[1,1] = p1 - p3 + p4 + p7
    # where:
    # p1 = (a+d)*(e+h)
    # p2 = (c+d)*e
    # p3 = a*(f-h)
    # p4 = d*(g-e)
    # p5 = (a+b)*h
    # p6 = (c-b)*(e+f)
    # p7 = (d-b)*(g+h)
    
    # For each block t (t=0,1,2), the block of B is B[:, 2*t : 2*t+2]
    # The output C[i, 2*t + k] comes from the t-th block
    
    # We need to express this as a tensor decomposition.
    # The multiplication tensor T has entries T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For Strassen on block t:
    # Let A be the full 2x2 matrix (same for all blocks)
    # Let B_t be the t-th 2x2 block of B
    
    # The 7 products for block t are:
    # p1_t = (A[0,0]+A[1,1]) * (B_t[0,0]+B_t[1,1])
    # p2_t = (A[1,0]+A[1,1]) * B_t[0,0]
    # p3_t = A[0,0] * (B_t[0,1]-B_t[1,1])
    # p4_t = A[1,1] * (B_t[1,0]-B_t[0,0])
    # p5_t = (A[0,0]+A[0,1]) * B_t[1,1]
    # p6_t = (A[1,0]-A[0,1]) * (B_t[0,0]+B_t[0,1])
    # p7_t = (A[1,1]-A[0,1]) * (B_t[1,0]+B_t[1,1])
    
    # And the outputs for block t:
    # C[0, 2*t+0] = p1_t - p5_t + p6_t + p7_t
    # C[0, 2*t+1] = p4_t + p5_t
    # C[1, 2*t+0] = p3_t + p6_t
    # C[1, 2*t+1] = p1_t - p3_t + p4_t + p7_t
    
    # Now build U, V, W matrices.
    # U has shape (n*m, rank) = (4, 21)
    # V has shape (m*p, rank) = (12, 21)
    # W has shape (p*n, rank) = (12, 21)
    
    # Indexing:
    # U[i*m + j, c] corresponds to A[i,j]
    # V[j*p + k, c] corresponds to B[j,k]
    # W[k*n + i, c] corresponds to C[i,k] (trace convention: W[k*n+i])
    
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # For each block t and each Strassen product s (0..6):
    # column index c = t*7 + s
    
    # Define the Strassen coefficients:
    # For product s, we have linear combinations of A and B.
    
    # Strassen product definitions:
    # s=0 (p1): A part: A[0,0] + A[1,1] => U entries: U[0,0]=1, U[3,0]=1 (A[0,0] is at index 0*2+0=0, A[1,1] is at 1*2+1=3)
    #           B part: B_t[0,0] + B_t[1,1] => V entries: V[0*6 + 2t + 0]=1, V[1*6 + 2t + 1]=1
    # s=1 (p2): A part: A[1,0] + A[1,1] => U[2,1]=1, U[3,1]=1
    #           B part: B_t[0,0] => V[0*6 + 2t + 0]=1
    # s=2 (p3): A part: A[0,0] => U[0,2]=1
    #           B part: B_t[0,1] - B_t[1,1] => V[0*6 + 2t + 1]=1, V[1*6 + 2t + 1]=-1
    # s=3 (p4): A part: A[1,1] => U[3,3]=1
    #           B part: B_t[1,0] - B_t[0,0] => V[1*6 + 2t + 0]=1, V[0*6 + 2t + 0]=-1
    # s=4 (p5): A part: A[0,0] + A[0,1] => U[0,4]=1, U[1,4]=1
    #           B part: B_t[1,1] => V[1*6 + 2t + 1]=1
    # s=5 (p6): A part: A[1,0] - A[0,1] => U[2,5]=1, U[1,5]=-1
    #           B part: B_t[0,0] + B_t[0,1] => V[0*6 + 2t + 0]=1, V[0*6 + 2t + 1]=1
    # s=6 (p7): A part: A[1,1] - A[0,1] => U[3,6]=1, U[1,6]=-1
    #           B part: B_t[1,0] + B_t[1,1] => V[1*6 + 2t + 0]=1, V[1*6 + 2t + 1]=1
    
    # Output coefficients for each block t:
    # C[0, 2t+0] = p1 - p5 + p6 + p7 => W[(2t+0)*2 + 0, c] += coeff for each s
    # C[0, 2t+1] = p4 + p5 => W[(2t+1)*2 + 0, c] += coeff
    # C[1, 2t+0] = p3 + p6 => W[(2t+0)*2 + 1, c] += coeff
    # C[1, 2t+1] = p1 - p3 + p4 + p7 => W[(2t+1)*2 + 1, c] += coeff
    
    for t in range(3):
        for s in range(7):
            c = t * 7 + s
            # B column offset in the 2x6 matrix
            bcol = 2 * t  # starting column of this block in B
            
            if s == 0:  # p1
                U[0, c] = 1   # A[0,0]
                U[3, c] = 1   # A[1,1]
                V[0*6 + bcol + 0, c] = 1   # B[0, bcol]
                V[1*6 + bcol + 1, c] = 1   # B[1, bcol+1]
                # Outputs: C[0, bcol] += 1, C[0, bcol+1] -= 1, C[1, bcol] += 0, C[1, bcol+1] += 1
                W[(bcol + 0)*2 + 0, c] = 1   # C[0, bcol]
                W[(bcol + 1)*2 + 0, c] = -1  # C[0, bcol+1]
                W[(bcol + 0)*2 + 1, c] = 0   # C[1, bcol]
                W[(bcol + 1)*2 + 1, c] = 1   # C[1, bcol+1]
            elif s == 1:  # p2
                U[2, c] = 1   # A[1,0]
                U[3, c] = 1   # A[1,1]
                V[0*6 + bcol + 0, c] = 1   # B[0, bcol]
                # Outputs: C[0, bcol] -= 1, C[0, bcol+1] += 0, C[1, bcol] += 0, C[1, bcol+1] -= 1
                W[(bcol + 0)*2 + 0, c] = -1  # C[0, bcol]
                W[(bcol + 1)*2 + 0, c] = 0   # C[0, bcol+1]
                W[(bcol + 0)*2 + 1, c] = 0   # C[1, bcol]
                W[(bcol + 1)*2 + 1, c] = -1  # C[1, bcol+1]
            elif s == 2:  # p3
                U[0, c] = 1   # A[0,0]
                V[0*6 + bcol + 1, c] = 1   # B[0, bcol+1]
                V[1*6 + bcol + 1, c] = -1  # B[1, bcol+1]
                # Outputs: C[0, bcol] += 0, C[0, bcol+1] += 0, C[1, bcol] += 1, C[1, bcol+1] -= 1
                W[(bcol + 0)*2 + 0, c] = 0   # C[0, bcol]
                W[(bcol + 1)*2 + 0, c] = 0   # C[0, bcol+1]
                W[(bcol + 0)*2 + 1, c] = 1   # C[1, bcol]
                W[(bcol + 1)*2 + 1, c] = -1  # C[1, bcol+1]
            elif s == 3:  # p4
                U[3, c] = 1   # A[1,1]
                V[1*6 + bcol + 0, c] = 1   # B[1, bcol]
                V[0*6 + bcol + 0, c] = -1  # B[0, bcol]
                # Outputs: C[0, bcol] += 0, C[0, bcol+1] += 1, C[1, bcol] += 0, C[1, bcol+1] += 1
                W[(bcol + 0)*2 + 0, c] = 0   # C[0, bcol]
                W[(bcol + 1)*2 + 0, c] = 1   # C[0, bcol+1]
                W[(bcol + 0)*2 + 1, c] = 0   # C[1, bcol]
                W[(bcol + 1)*2 + 1, c] = 1   # C[1, bcol+1]
            elif s == 4:  # p5
                U[0, c] = 1   # A[0,0]
                U[1, c] = 1   # A[0,1]
                V[1*6 + bcol + 1, c] = 1   # B[1, bcol+1]
                # Outputs: C[0, bcol] -= 1, C[0, bcol+1] += 1, C[1, bcol] += 0, C[1, bcol+1] += 0
                W[(bcol + 0)*2 + 0, c] = -1  # C[0, bcol]
                W[(bcol + 1)*2 + 0, c] = 1   # C[0, bcol+1]
                W[(bcol + 0)*2 + 1, c] = 0   # C[1, bcol]
                W[(bcol + 1)*2 + 1, c] = 0   # C[1, bcol+1]
            elif s == 5:  # p6
                U[2, c] = 1   # A[1,0]
                U[1, c] = -1  # A[0,1]
                V[0*6 + bcol + 0, c] = 1   # B[0, bcol]
                V[0*6 + bcol + 1, c] = 1   # B[0, bcol+1]
                # Outputs: C[0, bcol] += 1, C[0, bcol+1] += 0, C[1, bcol] += 1, C[1, bcol+1] += 0
                W[(bcol + 0)*2 + 0, c] = 1   # C[0, bcol]
                W[(bcol + 1)*2 + 0, c] = 0   # C[0, bcol+1]
                W[(bcol + 0)*2 + 1, c] = 1   # C[1, bcol]
                W[(bcol + 1)*2 + 1, c] = 0   # C[1, bcol+1]
            elif s == 6:  # p7
                U[3, c] = 1   # A[1,1]
                U[1, c] = -1  # A[0,1]
                V[1*6 + bcol + 0, c] = 1   # B[1, bcol]
                V[1*6 + bcol + 1, c] = 1   # B[1, bcol+1]
                # Outputs: C[0, bcol] += 1, C[0, bcol+1] += 0, C[1, bcol] += 0, C[1, bcol+1] += 1
                W[(bcol + 0)*2 + 0, c] = 1   # C[0, bcol]
                W[(bcol + 1)*2 + 0, c] = 0   # C[0, bcol+1]
                W[(bcol + 0)*2 + 1, c] = 0   # C[1, bcol]
                W[(bcol + 1)*2 + 1, c] = 1   # C[1, bcol+1]
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Check if the decomposition is correct."""
        # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where T[i,j,k] = 1 (the matrix multiplication tensor)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for c in range(len(U_mat[0])):
                        actual += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if not verify(U_list, V_list, W_list, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
