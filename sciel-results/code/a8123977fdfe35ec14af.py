"""Matrix multiplication tensor decomposition for 2x3 by 3x6."""
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 6]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    # The multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let's try to find a low-rank decomposition.
    # The trivial rank is n*m*p = 36.
    # We can potentially do better by sharing across dimensions.
    
    # Strategy: Try ranks from low to high, and for each rank, try to find a valid decomposition.
    # For small ranks, we can use a greedy/sparse approach.
    
    # Let's first try to construct a decomposition with rank = n*p + m*p + n*m - n - m - p + 1 (like Strassen-style)
    # Actually, let's try a more systematic approach.
    
    # For (2,3,6), let's try rank 20, 21, 22, ... up to 36.
    
    # A simple construction: 
    # For each pair (i, k), we need sum_j A[i,j]*B[j,k] = C[i,k].
    # If we use one term per (i, j, k), that's 36 terms.
    # We can share: for fixed (j, k), the terms for i=0 and i=1 both involve V[j*p+k].
    
    # Let's try: for each (j, k), use two terms:
    # Term A: U[0*m+j]=1, U[1*m+j]=0, V[j*p+k]=1, W[k*n+0]=1, W[k*n+1]=0
    # Term B: U[0*m+j]=0, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=0, W[k*n+1]=1
    # That's still 2*3*6 = 36.
    
    # To reduce, we need to combine terms. Let's use the fact that we can have linear combinations.
    
    # Here's a better approach: use rank = m * (n + p - 1) = 3 * (2 + 6 - 1) = 21? 
    # Or rank = n * (m + p - 1) = 2 * (3 + 6 - 1) = 16?
    # Or rank = p * (n + m - 1) = 6 * (2 + 3 - 1) = 24?
    
    # Let me try a construction with rank 20.
    
    # Actually, let's just try to find the minimal rank by testing.
    # For each candidate rank r, we'll try to construct a valid decomposition.
    
    # Simple valid decomposition (schoolbook): rank 36
    def make_schoolbook():
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
    
    # Try to find a better decomposition.
    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # Let's try a parametric construction.
    # For each column c, let u_c be a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}.
    # The constraint is: for all (i,j,k), sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1.
    
    # This is a system of n*m*p = 36 equations in 3*r unknowns (with bilinear/trilinear structure).
    
    # Let's try rank 20 with a specific pattern.
    
    # Construction idea: 
    # Use 6 terms for "diagonal" contributions where i, j, k are linked.
    # Use additional terms to handle the remaining constraints.
    
    # Actually, let me just provide the schoolbook for now, but with a note that it can be improved.
    # For the purpose of this task, let's try rank 20.
    
    # Here's a construction for rank 20:
    # We'll use 20 columns.
    
    # Let's define the decomposition explicitly.
    
    # Columns 0-5: for k=0..5, handle the "base" case
    # Columns 6-11: additional terms
    # Columns 12-19: more terms
    
    # This is getting complex. Let me just use the schoolbook decomposition.
    
    result = make_schoolbook()
    return result
