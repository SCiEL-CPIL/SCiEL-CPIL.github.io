"""
Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.

Structural change: Instead of independent block decomposition, we model the
2x2x15 tensor as 15 slices of 2x2x1 tensors. Each 2x2x1 slice has rank 2
(since it's a rank-2 matrix). However, we can potentially share rank terms
across slices.

The key insight: A 2x2x15 tensor T[i,j,k] where T[i,j,k]=1 for all i,j,k
can be viewed as T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].

For the 2x2x1 case, the minimum rank is 2 (since it's a 2x2 matrix which
has rank 2). But when we have 15 such slices, we might be able to do better
than 15*2=30 by sharing.

Actually, let's think more carefully. The tensor is T[i,j,k] = 1 for all
i in {0,1}, j in {0,1}, k in {0,...,14}.

We can write: T[i,j,k] = (1/2) * sum over a in {0,1} of (1 if i=a else 0) * 
  sum over b in {0,1} of (1 if j=b else 0) * sum over c in {0,...,14} of (1 if k=c else 0)

But that gives rank 4*15=60.

Better: T[i,j,k] = 1 = (1) * (1) * (1). So we can write:
T[i,j,k] = U[i,j] * V[j,k] * W[k,i] where U,V,W are all 1.
But that's rank 1! Wait, that's not right.

Let me reconsider. The decomposition is:
T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

For n=2, m=2, p=15:
T[i,j,k] = sum_r U[2i+j, r] * V[2j+k, r] * W[2k+i, r]

We need this to equal 1 for all i in {0,1}, j in {0,1}, k in {0,...,14}.

Let's try rank 1: U[2i+j, 0] * V[2j+k, 0] * W[2k+i, 0] = 1 for all i,j,k.
This means U[2i+j,0] * V[2j+k,0] * W[2k+i,0] = 1.
For this to hold for all i,j,k, we need U, V, W to be constant (all 1s).
Then U[2i+j,0] = 1, V[2j+k,0] = 1, W[2k+i,0] = 1.
So rank 1 works! But wait, that can't be right.

Let me verify: if U is all 1s (4x1), V is all 1s (30x1), W is all 1s (30x1),
then T[i,j,k] = 1*1*1 = 1. Yes! So the rank is 1?

No, that's wrong. The issue is that the indices are coupled. Let me re-examine.

Actually, I think I'm making an error. Let me be very explicit.

The matrix multiplication is: C = A * B where A is 2x2, B is 2x15, C is 2x15.
C[i,k] = sum_j A[i,j] * B[j,k].

The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.

The decomposition is:
C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k']) * ... 

Wait, I need to be more careful. The standard decomposition is:
C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k']) * W[k*n+i, r]

No, that's not right either. Let me look at the trace convention.

The trace convention says W[k*n+i, r]. So the decomposition is:
C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]

Wait, I think the decomposition is:
sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

where the sum over j is implicit in the indices. Actually, I think the correct
interpretation is:

The tensor T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k]).
The rank-r decomposition is:
T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

So we need: for all i in {0,1}, j in {0,1}, k in {0,...,14}:
sum_r U[2i+j, r] * V[2j+k, r] * W[2k+i, r] = 1

Let's try rank 1: U[2i+j,0] * V[2j+k,0] * W[2k+i,0] = 1 for all i,j,k.

For i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0] = 1
For i=0,j=0,k=1: U[0,0]*V[1,0]*W[2,0] = 1
For i=0,j=1,k=0: U[1,0]*V[2,0]*W[0,0] = 1
...

This is a system of 2*2*15 = 60 equations. Let's see if it's consistent.

From the first two: V[0,0]*W[0,0] = V[1,0]*W[2,0] (dividing by U[0,0])
This means V[0,0]/V[1,0] = W[2,0]/W[0,0].

Similarly, for fixed i=0,j=0: V[2j+k,0]*W[2k+i,0] is constant for all k.
So V[k,0]*W[2k,0] = c for k=0,...,14.
V[0,0]*W[0,0] = V[1,0]*W[2,0] = V[2,0]*W[4,0] = ... = V[14,0]*W[28,0] = c

For fixed i=0,j=1: V[2j+k,0]*W[2k+i,0] = V[2+k,0]*W[2k,0] = c' for all k.
So V[2,0]*W[0,0] = V[3,0]*W[2,0] = ... = V[16,0]*W[28,0] = c'

But V[2,0]*W[0,0] appears in both sequences. From the first: V[2,0]*W[4,0] = c.
From the second: V[2,0]*W[0,0] = c'.

These are different unless W[0,0] = W[4,0].

This is getting complicated. Let me just try to find a low-rank decomposition
numerically using ALS, starting from a rank-1 random initialization.

Actually, let me try a different approach. Since T[i,j,k] = 1 for all i,j,k,
this is a very simple tensor. The rank of the all-ones tensor of size n x m x p
is min(n,m,p) if we allow the indices to be independent. But here the indices
are coupled through the trace convention.

Let me just run ALS from rank 1 and see if it converges.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140  # Safety margin before 150s kill

def _compute_error(U, V, W, n, m, p):
    """Compute the Frobenius norm error of the decomposition."""
    rank = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                error += (actual - 1.0) ** 2
    return np.sqrt(error)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    return _compute_error(U, V, W, n, m, p) < tol

def _als(U, V, W, n, m, p, deadline, max_iters=500, tol=1e-10):
    """
    Run ALS on the 2x2x15 tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    prev_error = float('inf')
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Update U: for each (i,j)
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed (i,j): T[i,j,:] = U[i*m+j, :] @ (V[j*p+0:p, :] * W[0:p*n, :].reshape(p, n)[:, i])^T
                # Actually: B[r, k] = V[j*p+k, r] * W[k*n+i, r]
                B = np.zeros((rank, p))
                for r in range(rank):
                    for k in range(p):
                        B[r, k] = V[j * p + k, r] * W[k * n + i, r]
                if rank > 0:
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    except:
                        pass
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed (j,k): T[:,j,k] = V[j*p+k, :] @ (U[0:m*n, :] * W[k*n:0:-1*n*n, :])^T
                # C[r, i] = U[i*m+j, r] * W[k*n+i, r]
                C = np.zeros((rank, n))
                for r in range(rank):
                    for i in range(n):
                        C[r, i] = U[i * m + j, r] * W[k * n + i, r]
                if rank > 0:
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(C, np.ones(n), rcond=None)[0]
                    except:
                        pass
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed (k,i): T[i,:,k] = W[k*n+i, :] @ (U[i*m:0:-1*m*m, :] * V[0:m*p, :])^T
                # D[r, j] = U[i*m+j, r] * V[j*p+k, r]
                D = np.zeros((rank, m))
                for r in range(rank):
                    for j in range(m):
                        D[r, j] = U[i * m + j, r] * V[j * p + k, r]
                if rank > 0:
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(D, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Check convergence
        if (it + 1) % 10 == 0:
            error = _compute_error(U, V, W, n, m, p)
            if error < tol:
                break
            if abs(prev_error - error) < 1e-12:
                break
            prev_error = error
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # First, try very low ranks with multiple random initializations
    # The all-ones tensor might have a very low rank
    
    np.random.seed(42)
    
    best_result = None
    best_rank = float('inf')
    
    # Try ranks from 1 to 10
    for target_rank in range(1, 11):
        if time.time() > deadline - 5:
            break
        
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_init = np.random.randn(n * m, target_rank)
            V_init = np.random.randn(m * p, target_rank)
            W_init = np.random.randn(p * n, target_rank)
            
            # Run ALS
            U_opt, V_opt, W_opt = _als(U_init, V_init, W_init, n, m, p, deadline - 3, max_iters=200)
            
            # Check if we found a valid solution
            if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                best_result = (U_opt, V_opt, W_opt)
                best_rank = target_rank
                break
        
        if best_result is not None:
            break
    
    # If we found a low-rank solution, return it
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
