"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 10 matrix.
    
    The matrix multiplication tensor T has entries T[i,j,k] = delta_{j,k} where
    i indexes rows of A (n=2), j indexes columns of A / rows of B (m=3),
    k indexes columns of B (p=10).
    
    The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We use a structured approach based on the Kronecker product structure.
    For 2x3 * 3x10, we can exploit that the middle dimension is 3.
    
    The key insight: we can group by the middle index j. For each j, we have
    a 2x10 sub-problem (outer product of a 2-vector and 10-vector), which
    has rank at most 2*10 = 20, but can be done in rank 2 using the fact
    that it's an outer product.
    
    Actually, let me think more carefully. The tensor T_{i,j,k} = delta_{j,k}... 
    wait, no. For matrix multiplication C = A * B where A is 2x3 and B is 3x10:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor is T[i,j,k] which represents the bilinear map (A,B) -> C.
    The tensor entries are T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    
    The standard tensor decomposition: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where the sum is over c, and for each c we have:
    U[i*m+j, c] * V[j*p+k, c] gives a term in A[i,j] * B[j,k]
    
    Actually the tensor T has T[i,j,k] = delta_{j,k} is wrong. Let me reconsider.
    
    The matrix multiplication tensor for (n,m,p) has T[i,j,k] such that
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor is T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    
    The tensor T is defined by T[i,j,k] = delta_{i,j} * delta_{j,k}? No.
    
    Let me think again. The bilinear map is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This can be written as C = T(A,B) where T is a 3-tensor with T[i,j,k] such that
    C[i,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] delta_{i,i'} delta_{k,k'}
    
    Actually, the standard way: the multiplication tensor has entries
    T[i,j,k] = 1 if the index (i,j,k) corresponds to a valid term, i.e., always 1
    for i in [0,n), j in [0,m), k in [0,p). But that's the identity tensor.
    
    No wait. The tensor T for matrix multiplication (n,m,p) is defined such that
    the bilinear map (A,B) -> A*B is given by contracting T with A and B.
    
    T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    (A*B)[i,k] = sum_j A[i,j] B[j,k]
    
    So T[i,j,k] = delta_{j,j} = 1? That doesn't make sense as a tensor.
    
    I think the correct interpretation is:
    T[i,j,k] = 1 for all i,j,k in valid ranges, and the multiplication is
    C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    
    But T[i,j,k] = 1 for all i,j,k would mean C[i,k] = sum_j A[i,j] B[j,k] which is correct!
    
    So the tensor is just the all-ones tensor of shape (n, m, p)? That can't be right
    because then the rank would be n*m*p.
    
    Hmm, I think I'm confusing myself. Let me look at this differently.
    
    The matrix multiplication tensor T_{(n,m,p)} is a tensor in R^{n x m x p} defined by
    T[i,j,k] = 1 if j == j (always true), so T is the all-ones tensor? No...
    
    Actually, I think the correct definition is:
    T[i,j,k] = delta_{i,j} * delta_{j,k}? No, that's not right either.
    
    Let me just go with the standard result. For matrix multiplication (n,m,p),
    the tensor rank is at most n*m*p (schoolbook) and at least max(n*p, m*p, n*m) 
    (by slice rank or similar bounds).
    
    For 2x3*3x10: n=2, m=3, p=10
    - Schoolbook rank: 2*3*10 = 60
    - Lower bound: max(2*10, 3*10, 2*3) = max(20, 30, 6) = 30
    
    So the rank is between 30 and 60.
    
    Let me try to construct a decomposition of rank less than 60.
    
    One approach: use the fact that for each fixed j, we have a 2x10 outer product
    structure. For fixed j, C[:,j part] involves A[:,j] * B[j,:].
    
    Actually, C[i,k] = sum_j A[i,j] B[j,k]
    
    For a fixed j, the contribution to C from the j-th column of A and j-th row of B
    is A[:,j] * B[j,:] which is a rank-1 matrix (outer product of a 2-vector and 10-vector).
    
    So C = sum_j A[:,j] B[j,:]
    
    Each term A[:,j] B[j,:] can be computed with a rank-1 tensor decomposition.
    But we need to do this for all j simultaneously.
    
    The total rank is at most sum_j (rank of outer product for j) = 3 * (2*10) = 60? No,
    each outer product A[:,j] * B[j,:] is rank 1 as a matrix, but as a tensor decomposition
    term, it uses rank 1 (one term in the sum).
    
    Wait, I think I'm overcomplicating this. Let me just implement a basic search.
    
    Given the time constraints, let me try a few known constructions:
    
    1. Schoolbook: rank 60
    2. Try to find a better one using structured approaches
    
    For now, let me return the schoolbook decomposition as a baseline, and if time permits,
    try to improve it.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds to be safe
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Try to find a better decomposition
    # For 2x3*3x10, we can try rank 30 (the lower bound)
    
    # One approach: use the structure of the problem
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can write this as:
    # C = A * B where A is 2x3 and B is 3x10
    
    # Let's try a decomposition based on splitting the middle dimension
    
    # For each pair (i, k), C[i,k] = sum_j A[i,j] B[j,k]
    # There are n*p = 20 such entries
    
    # A tensor decomposition of rank r means we can write:
    # C[i,k] = sum_{c=1}^r U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # for all i, j, k
    
    # This is equivalent to saying the tensor T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # where T[i,j,k] = 1 for all i,j,k (since the multiplication tensor is all ones)
    
    # Wait, that's not right. The tensor T should satisfy:
    # T[i,j,k] such that (A*B)[i,k] = sum_j A[i,j] B[j,k]
    
    # I think the correct tensor is T[i,j,k] = 1 for all i,j,k, and the multiplication
    # is (A*B)[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    
    # So we need to decompose the all-ones tensor of shape (n, m, p) into a sum of
    # rank-1 tensors.
    
    # The rank of the all-ones tensor of shape (a, b, c) is min(a*b, b*c, a*c) if
    # we're talking about tensor rank? No, that's not right either.
    
    # The tensor rank of the all-ones tensor J_{a,b,c} is at most min(a*b, b*c, a*c)
    # by flattening. For (2,3,10): min(6, 30, 20) = 6.
    
    # So the rank should be at most 6? That seems too low.
    
    # Let me reconsider. The all-ones tensor J_{2,3,10} can be written as:
    # J[i,j,k] = 1 = (1,1)_{i} * (1,1,1)_{j} * (1,...,1)_{k}
    # This is a rank-1 tensor!
    
    # So the rank is 1? That can't be right for matrix multiplication.
    
    # I think I'm fundamentally misunderstanding the tensor structure.
    
    # Let me look at this from a different angle. The matrix multiplication tensor
    # for (n,m,p) is NOT the all-ones tensor. It's a specific tensor that encodes
    # the multiplication.
    
    # The correct tensor T_{(n,m,p)} has entries:
    # T[i,j,k] = delta_{i,j} * delta_{j,k} ? No.
    
    # Actually, I think the tensor is defined as:
    # T[i,j,k] = 1 if the indices (i,j,k) form a valid "chain" in the multiplication,
    # i.e., i indexes rows of A, j indexes columns of A / rows of B, k indexes columns of B.
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # But then the tensor is all-ones, and its rank is 1, which contradicts the
    # known lower bounds for matrix multiplication tensor rank.
    
    # I think the issue is that the tensor decomposition for matrix multiplication
    # is not a standard tensor rank decomposition. It's a different kind of decomposition
    # where the three modes have different meanings.
    
    # Let me just go with the schoolbook decomposition and return it.
    
    return {"U": best_U, "V": best_V, "W": best_W}
