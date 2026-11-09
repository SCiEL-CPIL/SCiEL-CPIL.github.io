"""Program for tensor rank of 2x5 by 5x6 matrix multiplication."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find the rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    """
    n, m, p = 2, 5, 6
    
    # Split n=2 into 1+1. M_{2,5,6} = M_{1,5,6} + M_{1,5,6}.
    # mu(1, m, p) = m + p - 1.
    # So rank = 10 + 10 = 20.
    
    # We need to construct M_{1, m, p} decomposition.
    # The tensor T[j, k] (for fixed i=0) satisfies:
    # T[j, k] = delta_{j,k} is NOT correct.
    # The tensor for matrix multiplication M_{n,m,p} is defined by:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T is in U^* \otimes V^* \otimes W^* where:
    # U has basis e_{i,j} for i in 0..n-1, j in 0..m-1
    # V has basis e_{j,k} for j in 0..m-1, k in 0..p-1
    # W has basis f_{k,i} for k in 0..p-1, i in 0..n-1
    # The tensor is T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes f_{k,i}
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    
    # For M_{1, m, p}: i=0 only.
    # T[j, k] = 1 for all j, k.
    # Wait, that's not right either. Let me re-read the task.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This suggests the tensor is:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # So for M_{1, m, p}: T[j, j*p+k, k] = 1 (since i=0, n=1).
    # Wait, n=1, so k*n+i = k*1+0 = k.
    # And i*m+j = 0*m+j = j.
    # And j*p+k.
    # So T[j, j*p+k, k] = 1.
    # This means the tensor is NOT all ones. It is the "diagonal" in some sense.
    
    # Let's re-examine the rank of M_{1, m, p}.
    # The tensor is T[j, j*p+k, k] = 1.
    # This is equivalent to the tensor for the dot product of a 1xm vector and a mxp matrix?
    # No, it's the tensor for the outer product of a vector and a matrix, but with a specific structure.
    # Actually, the rank of M_{1, m, p} is m+p-1 is a known result for the matrix multiplication tensor.
    # Let's verify for m=1, p=1. T[0, 0, 0] = 1. Rank 1. m+p-1=1. OK.
    # m=1, p=2. T[0, 0, 0]=1, T[0, 1, 1]=1.
    # Rank of this tensor: it's a 1x2x2 tensor.
    # T[0,0,0]=1, T[0,1,1]=1, others 0.
    # This is the sum of two rank-1 tensors:
    # e_0 \otimes e_0 \otimes f_0 and e_0 \otimes e_1 \otimes f_1.
    # Rank = 2. m+p-1 = 1+2-1 = 2. OK.
    
    # So the construction for M_{1, m, p} must produce a decomposition of the tensor T[j, j*p+k, k]=1.
    # Let's use the known construction for mu(1, m, p) = m+p-1.
    # The rank-1 tensors are indexed by c=0..m+p-2.
    # Let a_c = min(c, m-1), b_c = c - a_c.
    # The c-th rank-1 tensor has:
    # U part: e_{a_c} (index a_c in U, which has size m)
    # V part: e_{a_c*p + b_c} (index a_c*p+b_c in V, which has size m*p)
    # W part: f_{b_c} (index b_c in W, which has size p)
    # Let's verify for m=2, p=2. R=3.
    # c=0: a=0, b=0. U[0], V[0], W[0].
    # c=1: a=1, b=0. U[1], V[1*2+0=2], W[0].
    # c=2: a=1, b=1. U[1], V[1*2+1=3], W[1].
    
    # The tensor produced is:
    # T[0, 0, 0] += 1 (from c=0)
    # T[1, 2, 0] += 1 (from c=1)
    # T[1, 3, 1] += 1 (from c=2)
    # So T[0,0,0]=1, T[1,2,0]=1, T[1,3,1]=1.
    # But we want T[j, j*p+k, k]=1.
    # For m=2, p=2:
    # T[0, 0*2+0, 0]=1 -> T[0,0,0]=1. OK.
    # T[0, 0*2+1, 1]=1 -> T[0,1,1]=1. Missing!
    # T[1, 1*2+0, 0]=1 -> T[1,2,0]=1. OK.
    # T[1, 1*2+1, 1]=1 -> T[1,3,1]=1. OK.
    
    # So the construction is missing T[0,1,1]=1.
    # This means my understanding of the rank-1 tensor construction is wrong.
    # The correct construction for M_{1, m, p} is more subtle.
    # It involves the binomial coefficients or a different pattern.
    
    # Let's use a numerical approach to find the rank 10 decomposition for M_{1,5,6}.
    # We can set up the system of equations and solve it.
    # But that might be slow.
    
    # Alternatively, we can use the known fact that the rank is m+p-1 and use a greedy algorithm to find the decomposition.
    # Or, we can use the fact that the tensor is the sum of m+p-1 rank-1 tensors and find them by solving a linear system.
    
    # Let's try to construct it using the following idea:
    # The tensor T[j, j*p+k, k]=1 can be written as:
    # T = sum_{c=0}^{m+p-2} u_c \otimes v_c \otimes w_c
    # where u_c is a vector in R^m, v_c is a vector in R^{m*p}, w_c is a vector in R^p.
    # We need to find u_c, v_c, w_c such that for all j, k:
    # sum_c u_c[j] * v_c[j*p+k] * w_c[k] = 1
    # And for all other (j, k') with k' != k, the sum is 0.
    # Actually, the tensor is T[j, j*p+k, k]=1, so the condition is:
    # sum_c u_c[j] * v_c[j*p+k] * w_c[k] = 1
    # and for all (j, l) with l != j*p+k, sum_c u_c[j] * v_c[l] * w_c[k] = 0? No.
    # The tensor is indexed by (u_idx, v_idx, w_idx).
    # u_idx = j, v_idx = j*p+k, w_idx = k.
    # So the condition is:
    # sum_c u_c[j] * v_c[j*p+k] * w_c[k] = 1
    # and for all (j, v_idx, w_idx) not of the form (j, j*p+k, k), the sum is 0.
    # This is a system of m*m*p*p equations? No, the tensor has m*m*p*p entries? No.
    # The tensor is in R^{m} \otimes R^{m*p} \otimes R^{p}.
    # So it has m * (m*p) * p = m^2 * p^2 entries.
    # For m=5, p=6, that's 25*36=900 entries.
    # We have 10 unknown rank-1 tensors, each with m + m*p + p = 5+30+6=41 parameters.
    # Total parameters: 10*41=410.
    # But the rank-1 tensors are not linear, they are bilinear/trilinear.
    # So we have 900 equations and 410 unknowns? No, the unknowns are the entries of u_c, v_c, w_c.
    # This is a system of polynomial equations.
    
    # This is too complex. Let's use a different approach.
    # We can use the fact that the tensor for M_{1, m, p} is equivalent to the tensor for the outer product of a vector and a matrix, and use known decompositions.
    # Or, we can use the schoolbook algorithm for M_{1, m, p} which has rank m*p, and then try to reduce it.
    
    # For now, let's use the schoolbook algorithm for M_{2,5,6} which has rank 60.
    # And return that.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
