"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=4
    # Schoolbook rank = n*m*p = 16
    
    # We know that for matrix multiplication of (n x m) * (m x p),
    # the rank can sometimes be less than n*m*p.
    # For 2x2 * 2x4, let's try to find a lower rank decomposition.
    
    # The tensor T has entries T[i,j,k] = 1 if the output element (i,k) depends on input (j)
    # Actually, for matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,k,j] = 1 (since each product contributes)
    
    # In the trace convention W[k*n+i, r], the output index is (k, i) where i is row of A, k is col of B
    # U has shape (n*m, rank) - indices (i, j)
    # V has shape (m*p, rank) - indices (j, k)  
    # W has shape (p*n, rank) - indices (k, i)
    
    # We need: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for invalid combinations (if any), it should be 0. But in matrix multiplication,
    # every (i,j,k) triple contributes, so all should equal 1.
    
    # Wait, let me reconsider. The matrix multiplication tensor:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k]
    # So the tensor T[i,k,j] = 1 for all valid i, k, j.
    # There are n*p*m = 2*4*2 = 16 entries, all equal to 1.
    
    # A rank-1 decomposition term is u * v * w where:
    # u is a vector in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    # The product u[i*m+j] * v[j*p+k] * w[k*n+i] should sum over r to give T[i,k,j]
    
    # For the schoolbook: each (i,j,k) gets its own rank-1 term.
    # Can we do better? 
    
    # For 2x2 matrix multiplication, the rank is 7 (Strassen).
    # For 2x2 * 2x4, this is like two 2x2 multiplications (since p=4, we can split B into two 2x2 blocks).
    # So it's equivalent to two independent 2x2 multiplications, each of rank 7, total rank 14.
    # But maybe we can share some terms?
    
    # Let me think about this more carefully.
    # A is 2x2, B is 2x4. C = A*B is 2x4.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    
    # Using Strassen for 2x2:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M7
    
    # For our case, B has 4 columns. Let's split B into B_left (cols 0,1) and B_right (cols 2,3).
    # C_left = A * B_left (2x2), C_right = A * B_right (2x2)
    
    # Strassen for C_left uses B_left[0,0], B_left[0,1], B_left[1,0], B_left[1,1] = B[0,0], B[0,1], B[1,0], B[1,1]
    # Strassen for C_right uses B_right[0,0], B_right[0,1], B_right[1,0], B_right[1,1] = B[0,2], B[0,3], B[1,2], B[1,3]
    
    # If we use independent Strassen decompositions, rank = 14.
    # Can we do better by sharing? The A factors are the same, but the B factors are different.
    # In Strassen, the A factors are linear combinations of A entries, and B factors are linear combinations of B entries.
    # Since B_left and B_right involve different B entries, the B factors are independent.
    # The A factors could potentially be shared, but in Strassen they're specific combinations.
    
    # Let me just try rank 14 with independent Strassen decompositions.
    
    # Actually, let me try to construct a rank-14 decomposition.
    
    # For the left half (columns 0,1 of B, i.e., k=0,1):
    # Using Strassen with B entries B[0,0], B[0,1], B[1,0], B[1,1]
    
    # For the right half (columns 2,3 of B, i.e., k=2,3):
    # Using Strassen with B entries B[0,2], B[0,3], B[1,2], B[1,3]
    
    # Let me define the Strassen factors.
    # For a 2x2 * 2x2 multiplication:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Now, in our tensor decomposition:
    # U has indices (i,j) where i in {0,1}, j in {0,1}
    # V has indices (j,k) where j in {0,1}, k in {0,1,2,3}
    # W has indices (k,i) where k in {0,1,2,3}, i in {0,1}
    
    # For the left Strassen (k=0,1):
    # M1: U factor = a00+a11, V factor = b00+b11, W factor contributes to C00 and C11
    # etc.
    
    # This is getting complex. Let me just build it explicitly.
    
    # Let me use a different approach: try to find a low-rank decomposition using numerical methods.
    
    # Actually, let me just return the schoolbook decomposition of rank 16 first,
    # and then try to improve.
    
    # For now, let's return rank 16 (schoolbook).
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
