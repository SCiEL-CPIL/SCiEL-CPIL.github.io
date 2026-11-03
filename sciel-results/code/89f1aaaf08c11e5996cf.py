"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Strategy: Use a known low-rank decomposition for 3x3x3 matrix multiplication
    # and extend it to handle the 7 columns.
    # 
    # For 3x3x3, we use a rank-23 decomposition (a known construction).
    # For the remaining 4 columns, we add rank-12 (3*4) schoolbook terms.
    # Total: 23 + 12 = 35, which is much better than 63.
    #
    # Actually, let's use a more systematic approach:
    # Split B (3x7) into B1 (3x3) and B2 (3x4)
    # Use rank r1 for A*B1 (3x3x3) and rank r2 for A*B2 (3x3x4)
    # Total rank = r1 + r2
    
    # For 3x3x3: use a known rank-19 or rank-23 decomposition
    # For 3x3x4: use schoolbook rank 12, or try to do better
    
    # Let me use a constructive approach with the Strassen algorithm for 3x3
    # and schoolbook for the extra columns.
    
    # Known: 3x3 matrix multiplication can be done with rank 19 (Strassen 1969, 
    # generalized). Let me use a specific rank-23 construction that's easier to implement.
    
    # Actually, let me use a simpler but effective approach:
    # Use the fact that 3x3x7 can be decomposed by splitting the 7 columns
    # into 3+4, using a 3x3x3 low-rank decomposition and schoolbook for 3x3x4.
    
    # For 3x3x3, I'll use a rank-19 decomposition based on known results.
    # For 3x3x4, I'll use schoolbook (rank 12).
    # Total: 19 + 12 = 31
    
    # But implementing a specific rank-19 3x3x3 decomposition is complex.
    # Let me instead use a rank-27 decomposition for 3x3x3 (using 3x3 Strassen-like)
    # and schoolbook for the rest.
    
    # Simpler approach: Use a block decomposition.
    # Split A into 2x2 and 1x1 blocks, apply Strassen to the 2x2 parts.
    
    # Let me try a different strategy: direct algebraic decomposition.
    # For 3x3x7, the tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta
    # where delta[i,j,k] = 1 if the output index corresponds to (i,j,k).
    
    # The output C[i,l] = sum_j A[i,j] * B[j,l] for i in 0..2, l in 0..6
    # So W index is (l*n+i) = (l*3+i) for the official trace convention W[k*n+i,r]
    # Wait, the convention is W[k*n+i,r] where k is the column index of B (0..p-1=0..6)
    # and i is the row index of A (0..n-1=0..2). So W has 21 rows.
    
    # U[i*m+j, c] = U[i*3+j, c] for i in 0..2, j in 0..2 (9 rows)
    # V[j*p+k, c] = V[j*7+k, c] for j in 0..2, k in 0..6 (21 rows)  
    # W[k*n+i, c] = W[k*3+i, c] for k in 0..6, i in 0..2 (21 rows)
    
    # The decomposition: sum_c U[i*3+j, c] * V[j*7+k, c] * W[k*3+i, c] = delta_{i,j,k}
    # where delta_{i,j,k} = 1 if this is a valid product term, i.e., always 1 for the 
    # matrix multiplication tensor.
    
    # Actually, the matrix multiplication tensor for (n,m,p) = (3,3,7) is:
    # T[i,j,k] = 1 for all i in 0..2, j in 0..2, k in 0..6
    # The decomposition is: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c[i*3+j] = U[i*3+j, c], v_c[j*7+k] = V[j*7+k, c], w_c[k*3+i] = W[k*3+i, c]
    # and we need: sum_c U[i*3+j,c] * V[j*7+k,c] * W[k*3+i,c] = 1 for all i,j,k
    
    # This is the standard matrix multiplication tensor.
    
    # Let me use a known rank-23 decomposition for 3x3x3 and extend.
    # For the 3x3x3 part (columns 0,1,2 of B), use rank 23.
    # For the 3x3x4 part (columns 3,4,5,6 of B), use schoolbook rank 12.
    # Total: 23 + 12 = 35.
    
    # I need a specific rank-23 decomposition for 3x3x3.
    # Let me use the following known construction (simplified):
    
    # Actually, let me use a more practical approach: I'll construct a decomposition
    # by combining Strassen's algorithm for 2x2 blocks with handling of the remaining terms.
    
    # For 3x3 matrix multiplication, split A, B, C into 2x2, 2x1, 1x2, 1x1 blocks.
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # Similarly for B and C.
    
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x1)
    # C21 = A21*B11 + A22*B21 (1x2)
    # C22 = A21*B12 + A22*B22 (1x1)
    
    # Use Strassen for the 2x2 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M5
    
    # Each 2x2 product has rank 4 (schoolbook) or we can use Strassen recursively, but 
    # for 2x2, Strassen gives rank 7 instead of 4... wait, no. Strassen reduces the number 
    # of multiplications from 8 to 7 for 2x2. But in tensor decomposition terms, each "multiplication" 
    # is a rank-1 term.
    
    # Actually, in the tensor decomposition, each product of two submatrices corresponds to 
    # multiple rank-1 terms. Let me count:
    # - A11*B11: 2x2x2, rank 8 (schoolbook) or 7 (Strassen)
    # - A12*B21: 2x1x2, rank 4 (schoolbook)
    # - etc.
    
    # This is getting complex. Let me just use a direct approach with a known decomposition.
    
    # I'll use a rank-31 decomposition:
    # For the first 3 columns (3x3x3 part), use a rank-19 decomposition.
    # For the last 4 columns (3x3x4 part), use schoolbook rank 12.
    # Total: 19 + 12 = 31.
    
    # I need a specific rank-19 decomposition for 3x3x3. Here's one based on 
    # the work of Strassen (1969) and later improvements.
    
    # Let me use a simpler rank-23 decomposition that's easier to implement correctly.
    
    # For 3x3x3, here's a rank-23 decomposition (from known constructions):
    # I'll use the following approach:
    # - Use 7 Strassen products for the 2x2 core (rank 7*4 = 28? No, that's not right)
    
    # Let me just implement a practical decomposition and verify it.
    
    # Approach: Use the schoolbook for 3x3x3 (rank 27) and schoolbook for 3x3x4 (rank 12).
    # Total: 27 + 12 = 39. Better than 63.
    
    # Then I can try to reduce further.
    
    # For now, let me implement rank 39 and verify it works.
    
    rank = 39
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(21)]
    W = [[0]*rank for _ in range(21)]
    
    col = 0
    # First 3 columns: 3x3x3 schoolbook (rank 27)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*7+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Last 4 columns: 3x3x4 schoolbook (rank 12)
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U[i*3+j][col] = 1
                V[j*7+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Verify the decomposition
    if time.time() < deadline:
        # Check: sum_c U[i*3+j,c] * V[j*7+k,c] * W[k*3+i,c] should be 1 for all i,j,k
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    s = 0
                    for c in range(rank):
                        s += U[i*3+j][c] * V[j*7+k][c] * W[k*3+i][c]
                    if s != 1:
                        # Fallback to schoolbook
                        rank = 63
                        U = [[0]*rank for _ in range(9)]
                        V = [[0]*rank for _ in range(21)]
                        W = [[0]*rank for _ in range(21)]
                        col = 0
                        for i2 in range(3):
                            for j2 in range(3):
                                for k2 in range(7):
                                    U[i2*3+j2][col] = 1
                                    V[j2*7+k2][col] = 1
                                    W[k2*3+i2][col] = 1
                                    col += 1
                        break
                else:
                    continue
                break
            else:
                continue
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
