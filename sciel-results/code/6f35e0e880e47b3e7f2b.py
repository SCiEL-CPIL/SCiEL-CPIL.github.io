import numpy as np
from fractions import Fraction
import time
import random
import itertools

def construct():
    n, m, p = 2, 4, 5
    
    # Deadline guard
    deadline = time.time() + 110
    
    # The tensor for matrix multiplication 2x4x5:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # This is the all-ones tensor of size n x m x p.
    # The rank of the all-ones tensor is max(n, m, p) when we can use arbitrary real entries.
    # With integer/half-integer entries, we need to check if rank max(n,m,p) is achievable.
    #
    # For the all-ones tensor, a rank-r decomposition exists if and only if
    # we can write 1 as a sum of r products of entries from three vectors.
    # Specifically, if we set U[i,j,r], V[j,k,r], W[k,i,r] such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.
    #
    # One approach: use the fact that the all-ones matrix of size a x b has rank 1.
    # We can decompose the 3D all-ones tensor using a matrix factorization approach.
    #
    # Let's try to find a low-rank decomposition.
    # The key insight: if we can find vectors u_i, v_j, w_k (for each dimension)
    # such that sum_r u_i[r] * v_j[r] * w_k[r] = 1 for all i,j,k,
    # then we have a rank-r decomposition.
    #
    # This is equivalent to: the tensor T[i,j,k] = 1 can be written as a sum of r rank-1 tensors.
    # A rank-1 tensor is u_i * v_j * w_k.
    # So we need: sum_r u_i[r] * v_j[r] * w_k[r] = 1 for all i,j,k.
    #
    # This is possible if and only if there exist r vectors such that
    # for each i, the vector (u_i[0], ..., u_i[r-1]) is the same for all i (up to scaling)?
    # No, that's not quite right.
    #
    # Actually, this is a well-known result: the all-ones tensor of size n x m x p
    # has tensor rank max(n, m, p) over the reals.
    # With integer/half-integer constraints, we need to verify.
    #
    # Let's try rank = max(n, m, p) = 5.
    # We need to find 5 rank-1 tensors that sum to the all-ones tensor.
    #
    # Approach: Use the identity that the all-ones matrix J_a x b can be written as
    # J = 1 * 1^T (outer product of all-ones vectors).
    # For the 3D tensor, we can use a similar decomposition.
    #
    # Let's try: U[i,j,r] = a_i[r], V[j,k,r] = b_j[r], W[k,i,r] = c_k[r]
    # where a_i, b_j, c_k are vectors of length r.
    # We need: sum_r a_i[r] * b_j[r] * c_k[r] = 1 for all i,j,k.
    #
    # This is equivalent to: for each i,j,k, the inner product of
    # (a_i * b_j) and c_k (element-wise product) equals 1.
    #
    # Let's try r = max(n,m,p) = 5.
    # We can set:
    # a_i[r] = 1 for all i, r
    # b_j[r] = 1 for all j, r  
    # c_k[r] = 1/r for all k, r? No, entries must be integer/half-integer.
    #
    # Alternative: use a combinatorial design.
    # Let's try to find a decomposition where each term contributes 1/r.
    # If r divides 1, i.e., r = 1, 2, or r such that 1/r is integer/half-integer.
    # 1/r is integer/half-integer only for r = 1, 2.
    #
    # For r = 1: U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1 gives sum = 1. ✓
    # Wait, that's rank 1! Let me verify.
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For r = 0: U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 * 1 * 1 = 1.
    # So rank 1 works!
    
    # Let's verify this is correct.
    # The trivial decomposition has rank n*m*p = 40.
    # But if we set all entries to 1, then for any i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 * 1 * 1 = 1 (for r=0).
    # So the target tensor T[i,j,k] = 1 is satisfied.
    
    # But wait - is the target tensor really all 1s?
    # Let me re-examine the trivial code.
    # In the trivial code, for each (i,j,k), there's exactly one r (idx) where
    # U[i*m+j, idx] = 1, V[j*p+k, idx] = 1, W[k*n+i, idx] = 1.
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    # Yes, the target tensor is all 1s.
    
    # Therefore, rank 1 with all entries = 1 is a valid decomposition!
    
    rank = 1
    U = [[Fraction(1) for _ in range(rank)] for _ in range(n * m)]
    V = [[Fraction(1) for _ in range(rank)] for _ in range(m * p)]
    W = [[Fraction(1) for _ in range(rank)] for _ in range(p * n)]
    
    # Convert to lists of integers/half-integers
    def convert_to_list(arr):
        res = []
        for row in arr:
            res_row = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        res_row.append(int(val))
                    elif val.denominator == 2:
                        res_row.append(val.numerator / 2.0)
                    else:
                        res_row.append(float(val))
                else:
                    res_row.append(val)
            res.append(res_row)
        return res
    
    return {"U": convert_to_list(U), "V": convert_to_list(V), "W": convert_to_list(W)}
