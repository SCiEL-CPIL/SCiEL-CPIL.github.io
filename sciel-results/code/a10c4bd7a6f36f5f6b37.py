"""Rewritten program: search for a low-rank exact tensor decomposition of 2x3x10 matrix multiplication."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # 150s limit, leave 15s margin

    n, m, p = DIMENSIONS
    # Tensors: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Product: sum_r U[i*m+j, r] * V[j*p+k, r] = (A*B)[i,k]
    # W[k*n+i, r] must match the coefficient such that the tensor identity holds.
    # Actually the standard decomposition is:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} ? No.
    # The tensor for matrix multiplication M_{n,m,p} is T_{i,j,k} = 1 if we think of it as T[i,j,k].
    # The decomposition is T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} ... wait.
    # Standard: (AB)_{ik} = sum_j A_{ij} B_{jk}.
    # Tensor T_{i,j,k} has support on all (i,j,k).
    # Rank-1 term: u_{i,j} v_{j,k} w_{k,i}? No, the third index of the tensor is k, but the output is indexed by (i,k).
    # Let's stick to the contract:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    # The condition is: for all i, j, k:
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * ???
    # Wait, the tensor T_{i,j,k} is the structure constant.
    # T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The decomposition is T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}?
    # No, W has indices (k, i). So W_{k*n+i, r}.
    # So the condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    # This is a 3-way tensor decomposition (CP decomposition) of the all-ones tensor of size n x m x p?
    # No, the tensor is not all-ones in a simple way. The tensor T has T[i,j,k]=1.
    # Yes, it is the all-ones tensor of size n x m x p.
    # So we need a CP decomposition of the all-ones tensor of size 2x3x10.
    # Rank of all-ones tensor of size n x m x p is 1? No, that's for the matrix.
    # For the tensor, the all-ones tensor has rank 1? No.
    # If T_{i,j,k} = 1, then T = u v w^T where u_i=1, v_j=1, w_k=1.
    # So the rank is 1!
    # Let's check:
    # U[i*m+j, 0] = 1 for all i,j.
    # V[j*p+k, 0] = 1 for all j,k.
    # W[k*n+i, 0] = 1 for all k,i.
    # Then sum_r=0 U[i*m+j,0] V[j*p+k,0] W[k*n+i,0] = 1*1*1 = 1.
    # This works! The rank is 1.
    # Why is the schoolbook rank n*m*p? Because the schoolbook algorithm computes each product term separately.
    # But the tensor decomposition allows mixing.
    # Is the rank of the all-ones tensor really 1?
    # Yes, it is a rank-1 tensor.
    # So the minimal rank is 1.
    # Let's verify the interface.
    # U: (n*m, 1), V: (m*p, 1), W: (p*n, 1).
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # This is valid.
    
    # Let's double check if there are any constraints I missed.
    # "integer/half-integer entries" -> 1 is an integer.
    # "exactly decompose" -> 1*1*1 = 1.
    # So the answer is rank 1.
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
