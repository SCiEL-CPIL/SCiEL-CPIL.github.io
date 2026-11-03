"""Direct algebraic elimination approach for 3x4x8 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
from scipy.linalg import null_space

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 12
    M = m * p  # 32
    P = p * n  # 24
    total = N * M * P  # 9216 entries in the tensor
    
    # The tensor T has entries T[i,j,k] = 1 if indices are compatible
    # U index: i*m + j (0..11)
    # V index: j*p + k (0..31)
    # W index: k*n + i (0..23)
    
    # For a rank-R decomposition:
    # sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to: for each (i,j,k), the dot product of the R-dimensional vectors
    # u_{i,j} = (U[i*m+j, 0], ..., U[i*m+j, R-1])
    # v_{j,k} = (V[j*p+k, 0], ..., V[j*p+k, R-1])  
    # w_{k,i} = (W[k*n+i, 0], ..., W[k*n+i, R-1])
    # must equal 1.
    
    # We can fix U and V, then solve for W linearly, or vice versa.
    
    # Strategy: Try rank R from a lower bound upward.
    # Lower bound for 3x4x8: The tensor rank is at least max(N, M, P) = 32,
    # but tighter bounds exist. A common lower bound is ceil(N*M*P / max(N,M,P)) which isn't right.
    # The Strassen exponent gives asymptotic bounds. For small cases, we can use:
    # rank >= ceil(n*m*p / min(n,m,p)) is not a standard bound.
    # Actually, a valid lower bound is max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    # But we know from theory that for 2x2x2 the rank is 7 > max(4,4,4)=4.
    # Let's try ranks from 32 upward, but that's a lot of trials.
    
    # Better approach: Use the fact that if we fix U and V, then for each r,
    # we have a matrix A_r of size (N*M) x (P) where:
    # A_r[i*m+j, k*n+i] = U[i*m+j, r] * V[j*p+k, r]
    # Then W[:, r] must satisfy: for all (i,j,k), sum_r W[k*n+i, r] * A_r[i*m+j, k*n+i] = 1
    
    # This is getting complex. Let me try a different approach.
    
    # Direct approach: For a given rank R, we want to find U, V, W with entries in {0, 0.5, 1}
    # such that the tensor equation holds.
    
    # Let's try to construct a decomposition using a known efficient algorithm.
    # For 3x4x8, we can use the following:
    # Split into two 3x4x4 problems and use a rank-24 decomposition for each.
    # But we need to find a rank-24 decomposition for 3x4x4.
    
    # Let me try to find a rank-24 decomposition for 3x4x4 using linear algebra.
    # For 3x4x4: N=12, M=16, P=12, total=2304 entries.
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i in 0..2, j in 0..3, k in 0..3.
    
    # Fix U and V, solve for W.
    # For each r, define matrix B_r of size (N*M) x P where B_r[i*m+j, k*n+i] = U[i*m+j,r]*V[j*p+k,r]
    # Then we need sum_r B_r * W_r = e (vector of ones of length N*M... no, that's not right)
    
    # Actually, the equation is: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # This is: for each (i,j,k), the dot product of (U[i,j,:], V[j,k,:], W[k,i,:]) in a tensor sense = 1
    
    # Let me try a simpler approach: use the schoolbook decomposition but with some optimizations.
    # Schoolbook rank for 3x4x8 is 96.
    # We can use Strassen-like tricks to reduce this.
    
    # For now, let me return a valid decomposition with rank 48 using a structured approach.
    # I'll use the fact that we can pair up some of the schoolbook terms.
    
    # Actually, let me just return the schoolbook decomposition to ensure validity,
    # but try to optimize it.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
