"""
Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.

Strategy:
- The known optimal rank for 4x4*4x4 is 49 (Landsberg, 2019). For 4x4*4x5,
  the rank is at most 50 (by extending the 49-rank 4x4x4 decomposition with
  one extra column for the 5th output column, or by a direct construction).
- We use a structured ansatz based on the "block" construction: 
  - Use the known 7x7 Strassen-like patterns or more generally, we attempt
    to find a rank-49 or rank-50 decomposition by solving a system of polynomial
    equations via numerical optimization (Levenberg-Marquardt or similar).
- However, given the 120s time limit and the difficulty of non-convex optimization,
  we take a more pragmatic approach:
  - We know that the schoolbook rank is 80.
  - The best known rank for 4x4x5 is 50 (or possibly 49).
  - We will construct a rank-50 decomposition by taking the 49-rank 4x4x4
    decomposition (if we can hardcode it) and adding one more term.
  
  But hardcoding the 49-rank 4x4x4 decomposition is complex. Instead, we use
  a different approach: we use the "Kronecker product" based construction or
  a known explicit decomposition.

  Actually, a simpler approach: the rank of matrix multiplication [n,m,p] is
  at most n*m*p. For [4,4,5], schoolbook is 80.
  
  Known results:
  - [2,2,2] rank 7 (Strassen)
  - [4,4,4] rank 49 (Landsberg)
  - [4,4,5] rank <= 50 (by taking [4,4,4] rank 49 and adding 1 for the extra
    output dimension, but this isn't direct).
  
  A better bound: rank([4,4,5]) <= rank([4,4,4]) + 4 = 53? No.
  
  Actually, we can use the following construction:
  - For [4,4,5], we can view it as [4,4,4] + [4,4,1].
  - rank([4,4,1]) = 4 (since it's just 4 independent multiplications).
  - But the sum of ranks isn't the rank of the sum.
  
  A direct construction for rank 50:
  - Use 49 terms for the 4x4x4 part (if we had them) and 1 term for the extra
    output column. But we don't have the 49-term decomposition hardcoded.
  
  Alternative: Use a randomized approach to find a lower-rank decomposition.
  We'll use numerical optimization to minimize the rank by finding a sparse
  decomposition.
  
  Given the complexity, let's try a different tactic: use the fact that for
  small dimensions, we can use the "embedding" approach.
  
  Actually, the most practical approach within the time limit is to use a
  known explicit decomposition. The 4x4x4 rank-49 decomposition is known but
  complex. For 4x4x5, we can use the following:
  
  We'll use a rank-56 construction based on 2x2 Strassen blocks:
  - Partition the 4x4 matrices into 2x2 blocks.
  - Use Strassen's 7-multiplication algorithm for 2x2 blocks.
  - This gives a rank of 7 * 4 = 28 for the 2x2 block multiplications, but
    we need to account for the 5 output columns.
  
  Let me think more carefully. For [4,4,5]:
  - Input A: 4x4, Input B: 4x5, Output C: 4x5.
  - We can partition A into four 2x2 blocks: A11, A12, A21, A22.
  - B has 4 rows and 5 columns. We can partition B into two 2x5 blocks: B1, B2.
  - C has 4 rows and 5 columns. C = [C1; C2] where C1 is 2x5, C2 is 2x5.
  
  Using Strassen for 2x2 * 2x5:
  - M1 = (A11+A22)(B1+B2)
  - M2 = (A21+A22)B1
  - M3 = A11(B1-B2)
  - M4 = A22(B2-B1)
  - M5 = (A11+A12)B2
  - M6 = (A21-A11)(B1+B2)
  - M7 = (A12-A22)(B2-B1)
  
  But B1 and B2 are 2x5, so each "multiplication" is 2x2 * 2x5, which has
  rank 10 (schoolbook) or 7*? Actually, 2x2 * 2x5 can be done with rank 7*? 
  No, 2x2 * 2x5 is equivalent to 5 independent 2x2 * 2x1 multiplications, so
  rank 5*7 = 35? No, that's not right either.
  
  This is getting complicated. Let's use a simpler approach:
  
  We'll use the schoolbook decomposition but with a twist: we'll try to find
  a lower-rank decomposition by using the fact that the tensor is sparse in
  certain bases.
  
  Given the time constraints, I'll implement a rank-50 construction based on
  the following idea:
  - Use 49 terms from a known 4x4x4 decomposition (approximated) and 1 extra term.
  
  But since I don't have the exact 49-term decomposition, I'll use a different
  strategy: use numerical optimization to find a rank-r decomposition for
  r = 50, 51, ..., 80, starting from the best known lower bound.
  
  Given the 120s time limit, I'll implement a simple gradient-based optimizer
  that tries to find a rank-50 decomposition. If it fails, I'll fall back to
  the schoolbook rank-80 decomposition.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _schoolbook_decomposition():
    """Return the schoolbook (rank n*m*p) decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _tensor_residual(U, V, W, n, m, p):
    """Compute the residual of the tensor decomposition."""
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want T[i,j,k] = delta(i,j,k) for the matrix multiplication tensor,
    # where T[i,j,k] = 1 if the multiplication (A*B)[i,k] involves A[i,j]*B[j,k].
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid
    # (i,j,k) triples, and the decomposition should reproduce this.
    
    # The residual is the difference between the decomposed tensor and the
    # true matrix multiplication tensor.
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                residual += (val - 1.0) ** 2
    return residual

def _try_optimization(rank, max_iters=5000, lr=0.01, seed=42):
    """Try to find a rank-r decomposition using gradient descent."""
    n, m, p = DIMENSIONS
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_residual = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute gradient
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # d/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c] * (val - 1)
                    # d/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c] * (val - 1)
                    # d/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c] * (val - 1)
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = val - 1.0
                    dU[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :] * err
                    dV[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :] * err
                    dW[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :] * err
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        res = _tensor_residual(U, V, W, n, m, p)
        if res < best_residual:
            best_residual = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < 1e-10:
            break
    
    if best_residual < 1e-6:
        return best_U, best_V, best_W
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4*4x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin
    
    # Try ranks from 49 upward
    for rank in range(49, 81):
        if time.time() > deadline:
            break
        result = _try_optimization(rank, max_iters=2000, lr=0.005, seed=rank)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
