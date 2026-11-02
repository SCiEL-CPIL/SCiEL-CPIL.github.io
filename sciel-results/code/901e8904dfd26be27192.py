"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # A is n x m = 2 x 2
    # B is m x p = 2 x 10
    # C is n x p = 2 x 10
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Tensor: T[(i,k), (i',j'), (j'',k'')] = 1 iff i==i', j'==j'', k==k''
    # Mode sizes: C: n*p=20, A: n*m=4, B: m*p=20
    #
    # We seek rank-r decomposition: T = sum_{c=1}^{r} u_c * v_c * w_c
    # where u_c in R^{4}, v_c in R^{20}, w_c in R^{20}
    #
    # Strategy: Use the structure that C[i,k] depends on A[i,:] and B[:,k].
    # For each column k, we compute A * B[:,k]. 
    # 
    # Key insight: We can use Strassen's algorithm for 2x2 * 2x1 by treating
    # each column of B as a 2x1 matrix. But we need to handle 10 columns.
    #
    # Alternative: Since A is fixed (2x2), the map B -> C is linear in B for fixed A,
    # and linear in A for fixed B. The tensor is the "matrix multiplication tensor".
    #
    # Let's try to find a decomposition by exploiting that we only have 4 A-variables.
    # 
    # For each output C[i,k], we have:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # This is a sum of 2 terms per (i,k), giving 40 terms total (schoolbook).
    #
    # Can we share? If we define new B-variables:
    # B_new[l,k] = sum_j beta[l,j] * B[j,k] for l = 1..r
    # Then C[i,k] = sum_l alpha[i,l] * B_new[l,k]
    # where alpha[i,l] is linear in A's entries.
    #
    # This means we need: for all i,k:
    # sum_l alpha[i,l] * beta[l,j] = delta[i,j] (as a function of k, but actually independent of k)
    # Wait, that's not quite right. Let me re-derive.
    #
    # We want: C[i,k] = sum_j A[i,j] B[j,k] = sum_l (sum_j A[i,j] beta[l,j]) * B_new[l,k]
    # where B_new[l,k] = sum_j beta[l,j] B[j,k]... no, that's circular.
    #
    # Actually, the tensor decomposition is:
    # T = sum_c u_c * v_c * w_c
    # where (u_c)_{(i',j')} = coefficient of A[i',j'] in term c
    #       (v_c)_{(j'',k'')} = coefficient of B[j'',k''] in term c
    #       (w_c)_{(i,k)} = coefficient of C[i,k] in term c
    #
    # For the decomposition to be valid:
    # sum_c (u_c)_{(i',j')} (v_c)_{(j'',k'')} (w_c)_{(i,k)} = delta_{i,i'} delta_{j',j''} delta_{k,k''}
    # (with the constraint that j'=j'' for the term to be nonzero in the original tensor)
    #
    # More precisely, the original tensor is:
    # T[(i,k),(i',j'),(j'',k'')] = 1 if i==i', j'==j'', k==k'', else 0
    #
    # So we need: sum_c u_c[i',j'] * v_c[j'',k''] * w_c[i,k] = delta_{i,i'} delta_{j',j''} delta_{k,k''}
    #
    # Let's try a construction. Suppose we use r terms. For each term c, 
    # u_c is a 4-vector, v_c is a 20-vector, w_c is a 20-vector.
    #
    # Let me try rank = 30. I'll construct it as follows:
    # Group the 10 columns into 5 pairs. For each pair (2k, 2k+1), use 6 terms.
    # Total: 5 * 6 = 30.
    #
    # For a pair of columns, B has entries B[0,2k], B[1,2k], B[0,2k+1], B[1,2k+1].
    # C has entries C[0,2k], C[1,2k], C[0,2k+1], C[1,2k+1].
    #
    # C[0,2k] = a*B[0,2k] + b*B[1,2k]
    # C[1,2k] = c*B[0,2k] + d*B[1,2k]
    # C[0,2k+1] = a*B[0,2k+1] + b*B[1,2k+1]
    # C[1,2k+1] = c*B[0,2k+1] + d*B[1,2k+1]
    #
    # This is a 2x2 * 2x2 problem for the pair, which has rank <= 7 (Strassen).
    # But we want 6, which may not be possible. Let me use 7 per pair: 5*7=35.
    #
    # Actually, let me just implement the correct Strassen for 2x2*2x2 and apply it
    # to each pair of columns.
    #
    # Correct Strassen for 2x2 * 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let me verify with A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
    # C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    # M1 = (1+4)*(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 35
    # M3 = 1*(6-8) = -2
    # M4 = 4*(7-5) = 8
    # M5 = (1+2)*8 = 24
    # M6 = (3-1)*(5+6) = 2*11 = 22
    # M7 = (2-4)*(7+8) = -2*15 = -30
    # C00 = 65 - 8 + 24 + (-30) = 51... should be 19. Wrong!
    #
    # I keep getting Strassen wrong. Let me just use schoolbook for now with rank 40.
    # I'll return the schoolbook decomposition.
    
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
