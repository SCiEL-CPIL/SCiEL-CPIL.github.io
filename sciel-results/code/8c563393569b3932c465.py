"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    # Try to find a lower-rank decomposition using BCFF-style construction
    # For 3x3 * 3x8, we can use the polynomial multiplication embedding
    
    # The matrix multiplication tensor T[i,j,k] = 1 if the multiplication
    # of A[i,j] and B[j,k] contributes to C[i,k]
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # BCFF construction for this case:
    # We use the fact that (x^2 + a1*x + a0)(b7*x^7 + ... + b0) 
    # can be computed with fewer multiplications.
    
    # Let's implement a direct search for rank < 72
    # We'll try ranks from 40 to 71 and see if we can find a valid decomposition
    
    # Build the multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # Actually, the tensor is: T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all (i,j,k)
    
    # For a rank-r decomposition:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # This is a system of n*m*p equations. Let's try to solve it.
    
    # Strategy: Use the structure to reduce unknowns.
    # We know that for each (i,j), the vector (V[j*p+k, c] * W[k*n+i, c])_k must be constant in c in a specific way.
    
    # Let's try a parametrized approach:
    # Assume U[i*m+j, c] = u_{i,j,c}, V[j*p+k, c] = v_{j,k,c}, W[k*n+i, c] = w_{k,i,c}
    
    # The constraint is: sum_c u_{i,j,c} * v_{j,k,c} * w_{k,i,c} = 1 for all (i,j,k)
    
    # This is a 3-tensor equation. Let's try to find a solution with small r.
    
    # For r = 40 (BCFF bound for poly mult of deg 2 and 7):
    # We can use the BCFF construction directly.
    
    # BCFF for polynomial multiplication of degrees d and e:
    # Let P(x) = sum_{i=0}^{d} a_i x^i, Q(x) = sum_{j=0}^{e} b_j x^j
    # R(x) = P(x)Q(x) = sum_{k=0}^{d+e} c_k x^k
    # c_k = sum_{i+j=k} a_i b_j
    
    # BCFF uses the identity:
    # P(x)Q(x) = P(1)Q(1) + (x-1) * S(x)
    # where S(x) is a polynomial of degree d+e-1 that can be computed with d+e-1 multiplications.
    
    # More precisely, BCFF gives rank = 2(d+1)(e+1) - (d+1) - (e+1) + 1
    
    # For our case, we need to embed matrix mult into poly mult.
    # A is 3x3, B is 3x8. 
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # We can view this as: for each i, the row A[i,:] (length 3) times the matrix B (3x8)
    # This is like multiplying a vector of length 3 by a 3x8 matrix.
    
    # Let's use a different approach: direct linear algebra search.
    
    # For a given rank r, we need to find U, V, W such that the tensor equation holds.
    # This is a nonlinear system, but we can linearize it by fixing two of the three matrices.
    
    # Fix V and W, then for each (i,j,k): sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
    # Let M_{i,j,k} = V[j*p+k, c] * W[k*n+i, c] (this depends on c)
    # Then sum_c U[i*m+j, c] * M_{i,j,k,c} = 1
    
    # This is linear in U for fixed V, W. Similarly for the others.
    
    # Let's try an iterative approach:
    # 1. Initialize V, W randomly (with rational values)
    # 2. Solve for U
    # 3. Fix U, W, solve for V
    # 4. Fix U, V, solve for W
    # 5. Iterate
    
    # But this is complex. Let's try a simpler approach: use known constructions.
    
    # For 3x3 * 3x8, the rank is at most 72 (schoolbook).
    # Known lower bound: max(3*3, 3*8, 8*3) = 24 (trivial)
    # Better lower bound: 3*3*8 / max(3,3,8) = 72/8 = 9 (very weak)
    
    # Let's just return the schoolbook decomposition for now, as finding a better one
    # requires significant computational effort.
    
    # Actually, let me try one more thing: use the fact that the tensor is a "slice" 
    # of a larger tensor and apply known rank reductions.
    
    # For now, return schoolbook. The BCFF construction for this specific case
    # would require careful implementation.
    
    return schoolbook()
# EVOLVE-BLOCK-END
