"""Matrix multiplication 2x4 by 4x5 with rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit

    # The multiplication tensor T has entries T[i, j, k] = delta[i,k] (since (A*B)_{i,k} = sum_j A_{i,j} B_{j,k})
    # where i in [0,n), j in [0,m), k in [0,p).
    # A rank-r decomposition means: T[i,j,k] = sum_{t=0}^{r-1} U[i,t] * V[j,t] * W[k,t]
    # U is (n, r), V is (m, r), W is (p, r)
    # The interface requires U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)
    # with W[k*n+i, r] convention.
    #
    # Wait, let me re-read the interface:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor entry T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # For the standard matrix multiplication tensor:
    # T[i,j,k] = delta_{i,k} (Kronecker delta, 1 if i==k else 0)
    # because (AB)_{ik} = sum_j A_{ij} B_{jk}
    
    # Schoolbook: rank = n*m*p = 40
    # We need to find a lower rank.
    
    # Key insight: The tensor T[i,j,k] = delta_{i,k} is independent of j.
    # This means for each (i,k) pair where i==k, we need sum_t U[i*m+j, t]*V[j*p+k, t]*W[k*n+i, t] = 1 for all j
    # and for i!=k, the sum must be 0 for all j.
    
    # Let's think about this differently. The tensor is T[i,j,k] = delta_{i,k}.
    # This is a "slice" structure: for each j, the (i,k) slice is the identity matrix I_n (but only the top-left n x n block of an n x p matrix... wait n=2, p=5, so it's a 2x5 matrix with 1s at (0,0) and (1,1)).
    
    # Actually T[i,j,k] = 1 if i==k and i < n and k < n, else 0. Since n=2, p=5:
    # T[i,j,k] = 1 if (i=0,k=0) or (i=1,k=1), else 0. Independent of j.
    
    # So for each j, we have a 2x5 matrix E_j where E_j[i,k] = delta_{i,k} for i in {0,1}, k in {0,1,2,3,4}.
    # This is the same for all j!
    
    # This is a very special structure. Let's think about what rank decompositions exist.
    
    # For a single j, we need to represent the 2x5 matrix [[1,0,0,0,0],[0,1,0,0,0]] as a sum of rank-1 terms U_i * V_j * W_k.
    # But the constraint is that the same U, V, W work for ALL j.
    
    # Let me try a direct approach: use a randomized or structured search for low-rank decompositions.
    
    # First, let's try the obvious low-rank approach.
    # Since T is independent of j, maybe we can use the fact that it's essentially an identity embedding.
    
    # Try rank = n*p = 2*5 = 10? Or even lower?
    
    # Actually, let's think: T[i,j,k] = delta_{i,k}. 
    # Consider: U[i*m+j, t] = a_{i,t} (independent of j), V[j*p+k, t] = b_{j,t}, W[k*n+i, t] = c_{k,t}
    # Then sum_t a_{i,t} * b_{j,t} * c_{k,t} = delta_{i,k} for all i,j,k.
    # This must hold for all j, so for each j: sum_t a_{i,t} * b_{j,t} * c_{k,t} = delta_{i,k}.
    
    # If we set b_{j,t} = 1 for all j,t, then we need sum_t a_{i,t} * c_{k,t} = delta_{i,k}, which is just a rank-r decomposition of the 2x5 matrix [[1,0,0,0,0],[0,1,0,0,0]].
    # This matrix has rank 2, so r=2 suffices!
    
    # Wait, but we need integer/half-integer entries. Let's check:
    # a_{0,0}=1, a_{0,1}=0; a_{1,0}=0, a_{1,1}=1
    # c_{0,0}=1, c_{0,1}=0; c_{1,0}=0, c_{1,1}=1; c_{2,0}=0, c_{2,1}=0; c_{3,0}=0, c_{3,1}=0; c_{4,0}=0, c_{4,1}=0
    # b_{j,0}=1, b_{j,1}=1 for all j
    
    # Then sum_t a_{i,t} * b_{j,t} * c_{k,t} = a_{i,0}*1*c_{k,0} + a_{i,1}*1*c_{k,1}
    # For i=0,k=0: 1*1*1 + 0*1*0 = 1 ✓
    # For i=0,k=1: 1*1*0 + 0*1*1 = 0 ✓
    # For i=0,k=2: 1*1*0 + 0*1*0 = 0 ✓
    # For i=1,k=0: 0*1*1 + 1*1*0 = 0 ✓
    # For i=1,k=1: 0*1*0 + 1*1*1 = 1 ✓
    # For i=1,k=2: 0*1*0 + 1*1*0 = 0 ✓
    # All other (i,k) pairs give 0. ✓
    
    # So rank 2 works! Let me verify this is valid.
    
    # U[i*m+j, t] = a_{i,t} where a is 2x2: a = [[1,0],[0,1]]
    # V[j*p+k, t] = b_{j,t} where b is 4x2: b = [[1,1],[1,1],[1,1],[1,1]]
    # W[k*n+i, t] = c_{k,t} where c is 5x2: c = [[1,0],[0,1],[0,0],[0,0],[0,0]]
    
    # All entries are integers. Rank = 2.
    
    # Let me double-check the interface shapes:
    # U: (n*m, rank) = (8, 2) ✓
    # V: (m*p, rank) = (20, 2) ✓
    # W: (p*n, rank) = (10, 2) ✓
    
    # Let me verify more carefully:
    # U[i*m+j, t] = a[i, t]
    # V[j*p+k, t] = b[j, t]  
    # W[k*n+i, t] = c[k, t]
    
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # = sum_t a[i,t] * b[j,t] * c[k,t]
    # = a[i,0]*b[j,0]*c[k,0] + a[i,1]*b[j,1]*c[k,1]
    # = a[i,0]*1*c[k,0] + a[i,1]*1*c[k,1]
    # = a[i,0]*c[k,0] + a[i,1]*c[k,1]
    
    # a = [[1,0],[0,1]], c = [[1,0],[0,1],[0,0],[0,0],[0,0]]
    # a[i,0]*c[k,0] + a[i,1]*c[k,1]
    # i=0: c[k,0]
    # i=1: c[k,1]
    # k=0: i=0 -> 1, i=1 -> 0
    # k=1: i=0 -> 0, i=1 -> 1
    # k=2,3,4: all 0
    
    # So T[0,j,0]=1, T[1,j,1]=1, all others 0. This is exactly delta_{i,k} for i in {0,1}. ✓
    
    # This gives rank 2, which is much better than 40!
    
    # Let me construct this:
    rank = 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # a[i, t]: U[i*m+j, t] = a[i, t]
    # a = identity 2x2
    for i in range(n):
        for j in range(m):
            for t in range(rank):
                U[i * m + j][t] = 1 if i == t else 0
    
    # b[j, t]: V[j*p+k, t] = b[j, t] = 1 for all j, t
    for j in range(m):
        for k in range(p):
            for t in range(rank):
                V[j * p + k][t] = 1
    
    # c[k, t]: W[k*n+i, t] = c[k, t]
    # c = first 2 rows of identity, rest zeros
    for k in range(p):
        for i in range(n):
            for t in range(rank):
                W[k * n + i][t] = 1 if k == t else 0
    
    return {"U": U, "V": V, "W": W}
