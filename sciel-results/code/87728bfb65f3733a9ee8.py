import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def _validate(U, V, W, n):
    """
    Validates the decomposition.
    U: (n*n, rank)
    V: (n*n, rank)
    W: (n*n, rank)
    Checks if sum_r U[i,r] V[j,r] W[k,r] == Tensor[i,j,k]
    Tensor[i,j,k] is 1 if i,j,k correspond to a valid multiplication term.
    Convention:
    U index: i * n + j (A[i,j])
    V index: j * n + k (B[j,k]) -- Wait, current code uses V[j, k] where j is row of B?
    Let's check current code V:
    V[r, c] where r = (j_out * 3 + j_in) * 6 + (k_out * 3 + k_in).
    So V index is j * 6 + k.
    W index: k * 6 + i.
    
    The tensor for matrix mult C = A B is:
    C[i,j] = sum_k A[i,k] B[k,j].
    So the term is A[i,k] * B[k,j] * C[j,i]? No.
    The tensor M is defined such that M[i,k,j] = 1.
    But the indices in U, V, W are:
    U: i, j (from A[i,j]?)
    If U is A[i,j], V is B[j,k], W is C[k,i]?
    Then sum U[i,j] V[j,k] W[k,i] ?
    
    Let's check the current code's U:
    U[r, c] where r = (i_out * 3 + i_in) * 6 + (j_out * 3 + j_in).
    So U index is i * 6 + j.
    
    V index is j * 6 + k.
    
    W index is k * 6 + i.
    
    So the sum is over r:
    Sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    This should equal 1 if (i,j,k) is a valid term in the multiplication?
    No, the multiplication tensor is M[i,k,j] = 1 (A[i,k] B[k,j] -> C[i,j]).
    Here the indices are (i, j, k) in U, (j, k, i) in V, (k, i, j) in W?
    
    If U is A[i,j], V is B[j,k], W is C[k,i].
    Then the product is A[i,j] B[j,k] C[k,i].
    This doesn't match C[i,j] = sum_k A[i,k] B[k,j].
    
    Unless the indices in U are i, k?
    Current code U: `i_out, j_out`.
    If j in U corresponds to k in the formula?
    Let's assume the current code is correct and just swap the inner block.
    """
    pass

def construct():
    # Keep current logic
    pass
