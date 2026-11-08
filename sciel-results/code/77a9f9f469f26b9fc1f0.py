"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
n, m, p = DIMENSIONS

# The tensor T for matrix multiplication (n x m) * (m x p) -> (n x p)
# T[i, j, k, l] = 1 if j == k else 0, where i in [0,n), j in [0,m), k in [0,p), l in [0,n)
# Actually, let's be precise about the indices.
# Input A is n x m. Input B is m x p. Output C is n x p.
# C[i, l] = sum_{j} A[i, j] * B[j, l]
# The tensor T has indices (i, j, l) such that T[i, j, l] = 1 if we multiply A[i,j] and B[j,l] to contribute to C[i,l].
# In the task description, the output factor W has shape (p*n, rank) and follows W[k*n+i, r].
# Wait, the standard tensor decomposition for matrix multiplication C = AB is:
# C_{i,l} = sum_{r} U_{i,j}^{(r)} V_{j,l}^{(r)} ... no, that's not right.
# The standard form is:
# C_{i,l} = sum_{r} A_{i,j}^{(r)} B_{j,l}^{(r)} is not quite right either.
# Let's look at the interface:
# U: (n*m, rank) -> U[i*m+j, r]
# V: (m*p, rank) -> V[j*p+k, r]  (Note: B is m x p, so j is row, k is col)
# W: (p*n, rank) -> W[k*n+i, r]  (Note: Output is n x p? No, W index is k*n+i. k is from p? i is from n?
# Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
# Usually, the tensor T_{i,j,k} where C_{i,k} = sum_j A_{i,j} B_{j,k}.
# Here dimensions are n=3, m=3, p=8.
# A is 3x3. B is 3x8. C is 3x8.
# Indices: i in [0,3) for A rows and C rows. j in [0,3) for A cols and B rows. k in [0,8) for B cols and C cols.
# The tensor T has entries T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
# The decomposition is C[i,k] = sum_r U[i,j]_r * V[j,k]_r * W[...]_r?
# No, the standard matrix multiplication tensor decomposition is:
# T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r] ? No, that would be a 3-tensor decomposition.
# The interface provides U, V, W.
# U has shape (n*m, rank). Index i*m+j.
# V has shape (m*p, rank). Index j*p+k.
# W has shape (p*n, rank). Index k*n+i.
# This looks like a decomposition of the bilinear map.
# The multiplication tensor is T_{i,j,k} where i indexes A, j indexes the shared dimension, k indexes B?
# Actually, let's look at the schoolbook code:
# U[i*m+j][column] = 1
# V[j*p+k][column] = 1
# W[k*n+i][column] = 1
# This implies that for each (i,j,k), there is a rank-1 term.
# The product sum is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
# If we sum over r, and for a specific (i,j,k), only one r is non-zero (the one where U, V, W are all 1),
# then the sum is 1.
# How does this reconstruct the matrix multiplication?
# The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
# The tensor T_{i,j,k} = 1.
# The decomposition T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r} is not standard.
# Usually, it's T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} is for a 3-tensor? No.
# Let's assume the evaluation checks:
# For all i, j, k: T[i,j,k] == sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# If this is the case, the schoolbook solution works because for each (i,j,k), there is a unique r such that U, V, W are 1.
# The rank is the number of such terms, which is n*m*p = 72.

# We want to find a rank-minimal decomposition.
# The tensor rank of the matrix multiplication tensor for n x m by m x p is known to be at most n*m*p.
# For small dimensions, the rank might be lower.
# However, for 3x3 * 3x8, the rank is likely close to 72, but maybe slightly less?
# Actually, the rank of the matrix multiplication tensor M_{n,m,p} is known to be n*m*p for most small cases, but can be less for specific cases (e.g., 2x2x2 is 7, not 8).
# For 3x3x3, the rank is 23 (Lader).
# For 3x3x8, we don't have a known closed form, but we can try to find a decomposition with rank < 72.

# Strategy:
# 1. Start with the schoolbook decomposition (rank 72).
# 2. Try to reduce the rank by merging terms or using a randomized approach.
# Since the problem asks for an exact decomposition, we cannot just approximate.
# We can try to use the fact that the tensor is sparse and structured.
# Alternatively, we can try to find a decomposition with rank R < 72 by solving a system of equations.
# However, this is computationally expensive.

# Given the time limit (120s), we cannot do a full search.
# We will return the schoolbook decomposition as a safe baseline.
# If we have time, we can try to improve it.

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
