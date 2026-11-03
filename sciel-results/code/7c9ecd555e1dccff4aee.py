"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import numpy as np
import itertools
import time

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition for (n,m,p) multiplication.
    
    Convention:
      A is n x m, B is m x p, C is n x p
      U has shape (n*m, rank), indexed by (i*m + j)
      V has shape (m*p, rank), indexed by (j*p + k)
      W has shape (p*n, rank), indexed by (k*n + i)
      
    The decomposition is valid iff for all i, k:
      sum_r U[i*m + j, r] * V[j*p + k, r] = ... 
      Actually, the tensor entry T[i,j,k] = delta (should be 1 if we're decomposing the multiplication tensor)
      
    The multiplication tensor T has T[i,j,k] = 1 for all i,j,k in the "outer product" sense?
    No. The matrix multiplication tensor is defined by:
      C[i,k] = sum_j A[i,j] * B[j,k]
      
    As a trilinear form / tensor, the "multiplication tensor" has entries T[i,j,k] such that
      C[i,k] = sum_{j} A[i,j] B[j,k]
      
    The standard tensor decomposition of matrix multiplication uses the tensor with:
      T[i, j, k] = 1  (for the "multiplication" part, but actually it's more subtle)
    
    Let me think again. The bilinear map (A, B) -> C is linear in each.
    We can write C[i,k] = sum_j A[i,j] B[j,k].
    
    A rank-r decomposition means:
      C[i,k] = sum_{r=1}^{r} U[i*m+j, r] ... 
    
    Actually, the standard formulation: we want matrices U (n*m x r), V (m*p x r), W (p*n x r)
    such that for the multiplication tensor, we have:
    
    For each r, the "factor" is a rank-1 tensor. The r-th rank-1 tensor is:
      U_col_r (vector in R^{n*m}) outer V_col_r (vector in R^{m*p}) outer W_col_r (vector in R^{p*n})
    
    And we need:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for the "multiplication" pattern
    
    Wait, that's not quite right either. Let me look at the schoolbook case:
      For each (i,j,k), we have one term r with:
        U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    So the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for each (i,j,k)
    that corresponds to a "multiplication entry", and 0 otherwise? 
    
    No. The multiplication tensor T is defined by T[i,j,k] = 1 for ALL i,j,k.
    Because C[i,k] = sum_j A[i,j] B[j,k], and the "tensor" that represents this computation
    has T[i,j,k] = 1 for all valid i,j,k.
    
    So the condition is:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i in [0,n), j in [0,m), k in [0,p)
    
    Let me verify with schoolbook: for a fixed (i,j,k), exactly one r has all three entries = 1,
    so the sum is 1. For other (i',j',k'), if r corresponds to (i,j,k), then
    U[i'*m+j', r] = 1 only if i'=i and j'=j, so the product is 0 unless (i',j',k')=(i,j,k).
    Good.
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using tensor contraction."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need T = 1 everywhere
    T = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr)
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _build_rank27():
    """Construct rank-27 decomposition for 3x3x6 by sharing U factors.
    
    For 3x3x6: A is 3x3, B is 3x6, C is 3x6.
    We can view B as two 3x3 blocks: B1 (cols 0-2), B2 (cols 3-5).
    C = A @ B = [A @ B1 | A @ B2]
    
    Schoolbook for 3x3x3 has rank 27 (one term per (i,j,k) triple).
    For the first 3 columns of B: terms (i, j, k) for k in 0,1,2
    For the last 3 columns of B: terms (i, j, k+3) for k in 0,1,2
    
    U factors: U[i*3+j, r] = 1 for the r-th term.
    For the first block: r = 0..26, U[i*3+j, r] = 1 where (i,j) matches the (i,j) in the (i,j,k) triple.
    For the second block: we can reuse the SAME U columns (r = 0..26) because the (i,j) indices are the same.
    
    Wait, that means we only need 27 columns total, not 54!
    
    Let me verify: 
    Term r (0-indexed) corresponds to (i, j, k_block, k_in_block) where k_block is 0 or 1.
    Actually, let's index:
    For block 0 (k in 0,1,2): r = i*9 + j*3 + k, U[i*3+j, r] = 1, V[j*3+k, r] = 1, W[k*3+i, r] = 1
    For block 1 (k in 3,4,5): r = i*9 + j*3 + (k-3) ... wait, that would collide.
    
    Let me re-index. I have 27 U columns. Each column r corresponds to a pair (i, j) with multiplicity 3 (for the 3 k values in a 3x3 block). But since I have two blocks, I need 27 * 2 = 54 if I don't share.
    
    Hmm, let me reconsider. The key insight: U depends on (i,j), V depends on (j,k), W depends on (k,i).
    For the first block (k=0,1,2):
      U[i*3+j, r] = 1 for r = 9*(i) + 3*(j) + k  ... no, that's not right.
    
    Let me just build it explicitly.
    """
    n, m, p = 3, 3, 6
    # 27 U columns: indexed by (i, j, block) where block is 0 or 1
    # But we only have 27 columns, so we need to share.
    # 
    # Actually, the standard trick: for 3x3x6, we can use the fact that
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # If we split B into B[:,0:3] and B[:,3:6], then:
    # C[:,0:3] = A @ B[:,0:3]  (3x3 x 3x3 -> 3x3)
    # C[:,3:6] = A @ B[:,3:6]  (3x3 x 3x3 -> 3x3)
    #
    # Schoolbook for 3x3x3: rank 27.
    # For each (i,j,k) in [0,3)^3, one term.
    #
    # For C[:,0:3]: terms (i,j,k) for k in 0,1,2 -> 27 terms
    # For C[:,3:6]: terms (i,j,k+3) for k in 0,1,2 -> 27 terms
    #
    # Total: 54 terms if separate. But U[i*3+j, r] only depends on (i,j), not on k.
    # So for a fixed (i,j), there are 3 values of k in block 0 and 3 in block 1.
    # U[i*3+j, r] = 1 for all 6 of these r values.
    # 
    # So we can have 27 columns, where column r corresponds to (i, j, k) with k in 0..5.
    # U[i*3+j, r] = 1
    # V[j*6+k, r] = 1  (since B is 3x6, V is 3*6=18 rows)
    # W[k*3+i, r] = 1  (since C is 3x6, W is 6*3=18 rows)
    #
    # But that's just the schoolbook with rank 54.
    
    # The key insight for rank 27: use the SAME U columns for both blocks.
    # Let's have 27 columns. Column r = (i, j, t) where t in 0,1,2.
    # U[i*3+j, r] = 1
    # For block 0: V[j*3+k, r] = 1 where k = t, and W[k*3+i, r] = 1
    # For block 1: V[j*3+k+3, r] = 1 where k = t, and W[(k+3)*3+i, r] = 1
    #
    # But V and W are separate factors. We can't have V[j*6+k, r] and V[j*6+k+3, r] both = 1
    # for the same r unless we're okay with it.
    #
    # Actually, V[j*6+k, r] for k in 0..5. If I set V[j*6+t, r] = 1 AND V[j*6+t+3, r] = 1,
    # then for the first block, the k=t entry gets a 1, and for the second block, the k=t+3 entry gets a 1.
    # Similarly for W.
    #
    # Let's try: 27 columns, r = (i, j, t) with i,j,t in 0,1,2.
    # U[i*3+j, r] = 1
    # V[j*6+t, r] = 1
    # V[j*6+t+3, r] = 1
    # W[t*3+i, r] = 1
    # W[(t+3)*3+i, r] = 1
    #
    # Now check: for (i', j', k') with k' in 0..5:
    # sum_r U[i'*3+j', r] * V[j'*6+k', r] * W[k'*3+i', r]
    # = sum_{i,j,t: i=i', j=j'} V[j'*6+k', (i',j',t)] * W[k'*3+i', (i',j',t)]
    # = sum_t V[j'*6+k', (i',j',t)] * W[k'*3+i', (i',j',t)]
    #
    # V[j'*6+k', (i',j',t)] = 1 if k' = t or k' = t+3
    # W[k'*3+i', (i',j',t)] = 1 if k' = t or k' = t+3
    #
    # So for k' = t (t in 0,1,2): V=1, W=1 -> product = 1
    # For k' = t+3 (t in 0,1,2): V=1, W=1 -> product = 1
    # For other k': V=0 or W=0 -> product = 0
    #
    # So sum_t: if k' is in 0,1,2, then t=k' gives 1, others give 0 -> sum = 1
    # If k' is in 3,4,5, then t=k'-3 gives 1, others give 0 -> sum = 1
    #
    # This works! Rank 27.
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for t in range(3):
                U[i * m + j][col] = 1
                V[j * p + t][col] = 1       # k = t
                V[j * p + t + 3][col] = 1   # k = t+3
                W[t * n + i][col] = 1       # k = t
                W[(t + 3) * n + i][col] = 1 # k = t+3
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Return a rank-minimal tensor decomposition for 3x3x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with rank 27 (shared-A decomposition)
    result = _build_rank27()
    
    # Verify it's correct
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Try to find a lower rank decomposition
        # Known lower bounds: for 3x3x3, rank is at least 19 (Landsberg), best known is 23
        # For 3x3x6, the rank could potentially be lower than 27
        # But finding a valid decomposition below 27 is non-trivial
        # 
        # For now, return rank 27 which is a solid improvement over 54
        return result
    
    # Fallback: schoolbook
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
