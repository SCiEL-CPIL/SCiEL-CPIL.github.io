import numpy as np
import itertools

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank-45 decomposition for 2x5x5 matrix multiplication
    # This is based on the conjecture that R(2, n, n) = 2n^2 - n for n=5
    # If this fails, fall back to the trivial rank-50 decomposition
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For rank 45, we need to remove 5 terms from the trivial 50-term decomposition
    # The idea is that for each fixed j (column of A, row of B), we can combine
    # some of the k terms. Specifically, for R(2, n, n), the structure is:
    # For each i (row of A, 2 values), j (column of A/row of B, 5 values), k (col of B, 5 values)
    # We can potentially combine terms across k for fixed i,j.
    
    # Actually, let's think about this differently.
    # The tensor is sum_{i,j,k} e_i^A e_j^B e_k^C * e_{ij}^{in} e_{jk}^{out} e_{ki}^{prod}
    # Wait, the standard decomposition is:
    # (A * B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor for the "diagonal" case)
    # No wait, the multiplication tensor for matrix multiplication has:
    # T[i,j, k,l] = delta_{j,k} where i is row of A, j is col of A/row of B, k is col of B, l is row of output... 
    # Let me re-read the interface.
    
    # Interface: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # n=2, m=5, p=5
    # U has 10 rows (i*m+j for i in 0..1, j in 0..4)
    # V has 25 rows (j*p+k for j in 0..4, k in 0..4)
    # W has 10 rows (k*n+i for k in 0..4, i in 0..1)
    
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,k}... 
    # No, that's not right either. Let me think about what the matrix multiplication tensor actually is.
    
    # The multiplication tensor T for (n,m,p) matrices has T[i,j,k,l] = delta_{j,k}
    # where i in 0..n-1 (row of A), j in 0..m-1 (col of A / row of B), k in 0..p-1 (col of B), l in 0..n-1 (row of output)
    # Wait, the output is n x p, so l should be in 0..p-1? No, output row is i (0..n-1), output col is k (0..p-1).
    # So T[i,j,k,l] where l is the output column? No.
    
    # Let me re-derive. (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor has indices (i, j) for A, (j, k) for B, (i, k) for output.
    # T[(i,j), (j,k), (i,k)] = 1 if the middle indices match, else 0.
    # Actually T[(i,j), (j',k), (i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # So the tensor is: T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k, else 0.
    # More precisely: T[(i,j),(j',k),(i',k')] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    
    # The trivial decomposition has rank n*m*p = 2*5*5 = 50, with one term for each (i,j,k).
    # For term (i,j,k): U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
    
    # To get rank 45, we need to find 5 terms that can be combined.
    # This is equivalent to finding a rank-5 sub-tensor that can be decomposed with rank 0 (impossible) 
    # or finding that the whole tensor has rank 45.
    
    # Actually, I don't know a valid rank-45 decomposition. Let me just return the trivial one.
    
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
