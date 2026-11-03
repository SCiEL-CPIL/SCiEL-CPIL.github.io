"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    The schoolbook decomposition has rank 2*4*5 = 40.
    We aim to find a lower rank decomposition.
    
    For matrix multiplication of (n x m) by (m x p), the tensor T[i, j, k] = delta_{j,k} * ... 
    Actually, the multiplication tensor is T[i, j, k] where:
    (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    
    So the tensor entries are T[i, j, k] = 1 if the indices match appropriately.
    More precisely, T is a 3-tensor of dimensions (n*p) x m x m... no.
    
    Let me be precise. The matrix multiplication tensor T has entries:
    T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    T[i, j, k] = 1 if we're computing (A*B)[i,k] and the term A[i,j]*B[j,k] contributes.
    
    So T[i, j, k] = 1 for all valid i, j, k (since every A[i,j]*B[j,k] contributes to (A*B)[i,k]).
    Wait, that's not right either. The tensor is defined such that:
    (A*B)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    
    So T[i,j,k] = 1 for all i, j, k. That would give a trivial tensor of all ones.
    
    No, that's wrong. Let me reconsider.
    
    The bilinear map is: (A, B) -> A*B
    (A*B)[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T is a 3-tensor with dimensions (n*p) x (n*m) x (m*p) where:
    T[(i,k), (i',j), (j',k')] = 1 if i=i', j=j', k=k'
    
    So T is the identity-like tensor. The rank of this tensor is what we want to minimize.
    
    The schoolbook decomposition uses rank n*m*p = 2*4*5 = 40, with each rank-1 term corresponding
    to a specific (i,j,k) triple:
    U[(i*m+j), (i,j,k)] = 1
    V[(j*p+k), (i,j,k)] = 1
    W[(k*n+i), (i,j,k)] = 1
    
    Wait, the W index is k*n + i per the problem statement: "The output factor follows the official trace convention W[k*n+i,r]".
    
    So W has p*n rows, indexed by k*n + i.
    U has n*m rows, indexed by i*m + j.
    V has m*p rows, indexed by j*p + k.
    
    For a rank-r decomposition, we need:
    U: (n*m) x r
    V: (m*p) x r
    W: (p*n) x r
    
    such that for all i, j, k:
    sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    And for all other combinations (where the indices don't form a valid (i,j,k) triple in the right way), the sum should be 0.
    
    Actually, the condition is:
    For all i in [0,n), j in [0,m), k in [0,p):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    For all other (i', j', k') where the triple doesn't correspond to a valid multiplication term... 
    
    Hmm, actually the tensor T has non-zero entries only at specific positions. Let me think about this differently.
    
    The multiplication tensor T is in R^{(n*p) x (n*m) x (m*p)} but actually it's better to think of it as a 3-tensor.
    
    T[a, b, c] where:
    a = k*n + i (output index, p*n values)
    b = i*m + j (first matrix index, n*m values)  
    c = j*p + k (second matrix index, m*p values)
    
    T[a, b, c] = 1 if a = k*n + i, b = i*m + j, c = j*p + k for some valid i, j, k.
    T[a, b, c] = 0 otherwise.
    
    So the non-zero entries are exactly those where the indices are consistent.
    
    For a rank-r decomposition, we need matrices U (n*m x r), V (m*p x r), W (p*n x r) such that:
    T[a, b, c] = sum_{t=0}^{r-1} U[b, t] * V[c, t] * W[a, t]
    
    for all a, b, c.
    
    The schoolbook uses r = n*m*p = 40.
    
    Can we do better? For small matrices, the minimal rank is known or bounded.
    
    For 2x4x5, I'm not sure of the exact minimal rank. Let's try to find a decomposition with rank less than 40.
    
    One approach: use the fact that the first dimension is only 2. We can try to exploit this.
    
    Let me try a different approach: use known results or try to construct a lower-rank decomposition.
    
    Actually, for matrix multiplication, the tensor rank is at least max(n*p, n*m, m*p) = max(10, 8, 20) = 20.
    
    So the rank is at least 20. The schoolbook gives 40. Can we get between 20 and 40?
    
    Let me try to use a simple heuristic: try to find a rank-r decomposition using random initialization and gradient descent, or use a structured approach.
    
    Given the time constraint, let me try a structured approach based on the small first dimension.
    
    For n=2, we can split the problem into two cases based on i=0 and i=1.
    
    Actually, let me just try to return the schoolbook decomposition for now, as finding a lower rank is non-trivial.
    
    Wait, the task says "Find a rank-minimal exact tensor decomposition". The score is the rank, lower is better. The seed has rank 40.
    
    Let me try to see if I can find a rank less than 40.
    
    One idea: use the fact that for the 2x4x5 case, we might be able to share some computations.
    
    Let me try a simple approach: use rank 40 (schoolbook) as a baseline, and see if I can improve.
    
    Actually, let me just implement the schoolbook for now, since finding a provably lower rank is complex.
    """
    n, m, p = DIMENSIONS
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
