import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Constructs the standard (schoolbook) tensor decomposition for matrix multiplication.
    This approach directly builds the factor matrices U, V, W such that
    T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r] = 1 if (i,j,k) corresponds to a valid multiplication term, 0 otherwise.
    
    For standard matrix multiplication of A (n x m) and B (m x p), the term C[i, k] involves sum over j of A[i, j] * B[j, k].
    The tensor T has indices i (row of A), j (col of A / row of B), k (col of B).
    T[i, j, k] = 1.
    
    We decompose this into rank R = n * m * p.
    Each term in the sum corresponds to a unique (i, j, k) triple.
    U[i*m + j, r] = 1 if r corresponds to (i, j, *)
    V[j*p + k, r] = 1 if r corresponds to (*, j, k)
    W[k*n + i, r] = 1 if r corresponds to (i, *, k)
    
    Wait, the standard identity is:
    T[i, j, k] = 1.
    We want sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1.
    
    Let r index the triple (i, j, k).
    U_{ij,r} = 1 if r = (i, j, k) else 0? No, U depends on i, j.
    If we set r = (i, j, k):
    U_{ij,r} should be 1 only if the i, j match.
    V_{jk,r} should be 1 only if the j, k match.
    W_{ki,r} should be 1 only if the k, i match.
    
    So for a fixed r corresponding to (i0, j0, k0):
    U_{ij,r} = 1 if i==i0 and j==j0.
    V_{jk,r} = 1 if j==j0 and k==k0.
    W_{ki,r} = 1 if k==k0 and i==i0.
    
    Then for T[i, j, k]:
    Sum over r. Only r=(i, j, k) contributes.
    U_{ij, (i,j,k)} = 1.
    V_{jk, (i,j,k)} = 1.
    W_{ki, (i,j,k)} = 1.
    Product = 1.
    
    For any other (i, j, k), no r matches all three conditions simultaneously?
    If we sum over all r=(i', j', k'):
    Term is non-zero only if i=i', j=j' (for U), j=j', k=k' (for V), k=k', i=i' (for W).
    This implies i=i', j=j', k=k'.
    So only the term r=(i,j,k) survives.
    Value is 1.
    
    This works.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, r] = 1
                # V index: j*p + k
                V[j * p + k, r] = 1
                # W index: k*n + i
                W[k * n + i, r] = 1
                r += 1
                
    def to_list(mat):
        return mat.tolist()
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
