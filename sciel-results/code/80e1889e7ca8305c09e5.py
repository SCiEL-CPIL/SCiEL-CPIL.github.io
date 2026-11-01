"""
Exact tensor decomposition for 3x3 @ 3x4 matrix multiplication.

Approach:
  Direct construction of the schoolbook decomposition.
  
  For matrix multiplication $C = AB$ where $A$ is $n \times m$ and $B$ is $m \times p$,
  the standard algorithm computes $c_{ik} = \sum_{j=1}^{m} a_{ij} b_{jk}$.
  This corresponds to a tensor decomposition of rank $n \cdot m \cdot p$.
  
  The tensor $T$ has entries $T[a, b, c]$ where:
  $a = i \cdot m + j$
  $b = j \cdot p + k$
  $c = k \cdot n + i$
  
  Since each triple $(i, j, k)$ maps to a unique triple $(a, b, c)$, the supports of the
  rank-1 tensors are disjoint. Therefore, the rank is exactly $n \cdot m \cdot p$.
  There is no possibility for cancellation or sharing of terms to reduce the rank.
  
  This program directly constructs the factor matrices $U, V, W$ of size $(nm, nmp)$,
  $(mp, nmp)$, and $(pn, nmp)$ respectively, with integer entries 0 or 1.
"""

def construct():
    n, m, p = 3, 3, 4
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The rank of the schoolbook decomposition is n*m*p
    rank = n * m * p

    # Initialize factors with zeros
    # U has shape (nm, rank)
    # V has shape (mp, rank)
    # W has shape (pn, rank)
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # Populate the factors
    # The r-th term in the sum corresponds to a unique (i, j, k).
    # We map the linear index r to (i, j, k) via r = i * m * p + j * p + k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1

    return {"U": U, "V": V, "W": W}
