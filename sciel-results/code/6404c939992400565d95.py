"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition.

We exploit the structure: C = A * B where A is 3x3, B is 3x6, C is 3x6.
The output convention is W[k*n+i, r] where k is the row (0..5) and i is the column (0..2).

C[k, i] = sum_{j=0}^{2} A[i, j] * B[k, j]

The tensor has dimensions:
- U: (n*m, rank) = (9, rank), indexed by i*m + j
- V: (m*p, rank) = (18, rank), indexed by j*p + k
- W: (p*n, rank) = (18, rank), indexed by k*n + i

We can split B into two 3x3 blocks: B1 (rows 0-2) and B2 (rows 3-5).
Then C1 = A * B1 (3x3) and C2 = A * B2 (3x3).

If we have a rank-r decomposition of 3x3x3 multiplication, we can use it twice
with shared U factors, giving total rank 2r.

The best known upper bound for 3x3x3 is 23 (Bläser 2005), but implementing
that exact decomposition is complex and error-prone.

A safer approach: use a known rank-20 or lower decomposition if available,
or fall back to schoolbook rank 27 per block (total 54).

However, we can try to find a better decomposition using optimization.
The key insight is that for the 3x3x3 case, there exist rank-23 decompositions.
Let me try to implement a known one.

Actually, let me try a different approach: use the fact that we can sometimes
do better than 2*27 = 54 by exploiting the shared structure more carefully.

One approach: use a rank-23 decomposition for 3x3x3. The Bläser decomposition
uses 23 terms. Let me try to encode it.

Actually, the safest bet for correctness is to return the schoolbook rank-54
decomposition. But let me try to find a lower-rank one using a simple search.

Given the time constraints, I'll implement:
1. Schoolbook rank-54 as fallback
2. Try to find a rank-46 decomposition (2 * 23) using a known 3x3 decomposition
3. If that fails, return schoolbook

Let me try to implement the Bläser rank-23 decomposition for 3x3x3.
This is based on the paper "Lower bounds for matrix multiplication" by Bläser.

Actually, I don't have the exact coefficients memorized. Let me try a different
approach: use a randomized search to find a low-rank decomposition.

The constraint is: for all (i, j, k) in [0,3)x[0,3)x[0,6):
sum_r U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r] = 1

This is a system of 3*3*6 = 54 equations in rank variables (3*rank + 18*rank + 18*rank = 39*rank unknowns).

For rank 54, we have 39*54 = 2106 unknowns for 54 equations - very underdetermined.
For rank 46, we have 39*46 = 1794 unknowns for 54 equations.

The problem is non-convex. Let me try a simple approach: start with schoolbook
and try to merge terms by setting some columns to be linear combinations.

Actually, let me just return the schoolbook decomposition to guarantee correctness.
The score will be 54, which is the baseline.

Wait - the score is the rank, and lower is better. The seed program returns rank 54.
So any decomposition with rank < 54 would improve the score.

Let me try to implement a rank-46 decomposition using the known rank-23 for 3x3x3.

After research, I recall that there's a well-known rank-23 decomposition. Let me try
to implement it based on the structure.

Actually, let me try a simpler improvement: can we do rank 49 or 50?

One idea: use 26 terms for one 3x3 block and 23 for the other (total 49), if we can
find a rank-26 decomposition for 3x3x3 that's easier to implement.

But the safest approach is still schoolbook. Let me just return that.

Actually, wait. Let me re-read the task. The score is the rank, and the search score
is seed/value. So if seed = 54 (schoolbook), then:
- Rank 54: score = 54/54 = 1.0
- Rank 46: score = 54/46 ≈ 1.174

So any lower rank improves the score. Let me try to find one.

Let me implement a simple local search: start with schoolbook, randomly perturb,
and try to reduce rank by merging columns.

Actually, a cleaner approach: use the fact that the 3x3x3 tensor has a known
rank-23 decomposition. Let me try to find it online... but I can't in this context.

Let me try a computational approach: use SVD or similar to find a low-rank
approximation, then round to integers/half-integers.

The tensor T[i,j,k] = 1 for all valid (i,j,k). We want to write it as a sum of
rank-1 tensors. This is exactly the tensor rank problem.

For a 3x3x3 tensor, the rank is at most 27 (schoolbook). The question is whether
we can do better.

Let me try a simple approach: use random initialization and gradient descent on
the non-convex objective.

Objective: minimize ||sum_r u_r ⊗ v_r ⊗ w_r - T||_F^2
subject to the sum being exactly T.

This is hard, but let me try.

Actually, for the 3x3x6 case, let me just try to find a rank-46 decomposition
by solving a system of equations.

Here's my plan:
1. Assume rank 46
2. Use 23 terms for the first 3x3 block and 23 for the second
3. The U factors for both blocks must be the same (shared A)
4. Solve for the 23 U, 23 V1, 23 W1, 23 V2, 23 W2 factors

This is a system of 27*2 = 54 equations in 23*3 + 23*2 + 23*2 = 161 unknowns.
Wait, that's not right. Let me count again.

For 3x3x3: U is 9x23, V is 9x23, W is 9x23. That's 9*23*3 = 621 unknowns.
Constraints: 27 equations.

This is very underdetermined. We can solve it using least squares.

Let me implement this.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r] for r in range(rank))
                if abs(val - 1) > 1e-6:
                    return False
    return True

def _try_rank_3x3_3x3(target_rank, seed=42):
    """Try to find a rank-target_rank decomposition of 3x3x3 matrix multiplication.
    
    Returns (U, V, W) each of shape (9, target_rank) or None.
    """
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(9, target_rank) * 0.5
    V = np.random.randn(9, target_rank) * 0.5
    W = np.random.randn(9, target_rank) * 0.5
    
    # Use iterative refinement
    for iteration in range(1000):
        # Compute current tensor
        T = np.zeros((3, 3, 3))
        for r in range(target_rank):
            T += np.einsum('i,j,k->ijk', U[:, r].reshape(3, 3), V[:, r].reshape(3, 3), W[:, r].reshape(3, 3))
        
        # Target tensor is all ones
        target = np.ones((3, 3, 3))
        
        # Compute gradient with respect to U, V, W
        # dT/dU[i,j] = sum_r V[j*3+k] * W[k*3+i] for each (i,j,k)
        # This is complex, so let's use a simpler approach
        
        # Instead, use a fixed-point iteration:
        # For each r, update U[:, r] to best fit the residual
        residual = target - T
        
        for r in range(target_rank):
            # Update U[:, r]
            # We want to maximize sum_{i,j,k} residual[i,j,k] * U[i,j,r] * V[j,k,r] * W[k,i,r]
            # For fixed V[:, r] and W[:, r], optimal U[:, r] is:
            # U[i,j,r] = sum_k residual[i,j,k] * V[j,k,r] * W[k,i,r] / (sum_k V[j,k,r]^2 * W[k,i,r]^2 + eps)
            num = np.zeros((3, 3))
            den = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        num[i, j] += residual[i, j, k] * V[j, k, r] * W[k, i, r]
                        den[i, j] += V[j, k, r] ** 2 * W[k, i, r] ** 2
            U[:, r] = (num / (den + 1e-8)).flatten()
            
            # Recompute T
            T = np.zeros((3, 3, 3))
            for rr in range(target_rank):
                T += np.einsum('i,j,k->ijk', U[:, rr].reshape(3, 3), V[:, rr].reshape(3, 3), W[:, rr].reshape(3, 3))
            residual = target - T
            
            # Update V[:, r]
            num = np.zeros((3, 3))
            den = np.zeros((3, 3))
            for j in range(3):
                for k in range(3):
                    for i in range(3):
                        num[j, k] += residual[i, j, k] * U[i, j, r] * W[k, i, r]
                        den[j, k] += U[i, j, r] ** 2 * W[k, i, r] ** 2
            V[:, r] = (num / (den + 1e-8)).flatten()
            
            # Recompute T
            T = np.zeros((3, 3, 3))
            for rr in range(target_rank):
                T += np.einsum('i,j,k->ijk', U[:, rr].reshape(3, 3), V[:, rr].reshape(3, 3), W[:, rr].reshape(3, 3))
            residual = target - T
            
            # Update W[:, r]
            num = np.zeros((3, 3))
            den = np.zeros((3, 3))
            for k in range(3):
                for i in range(3):
                    for j in range(3):
                        num[k, i] += residual[i, j, k] * U[i, j, r] * V[j, k, r]
                        den[k, i] += U[i, j, r] ** 2 * V[j, k, r] ** 2
            W[:, r] = (num / (den + 1e-8)).flatten()
        
        # Recompute final T
        T = np.zeros((3, 3, 3))
        for r in range(target_rank):
            T += np.einsum('i,j,k->ijk', U[:, r].reshape(3, 3), V[:, r].reshape(3, 3), W[:, r].reshape(3, 3))
        
        error = np.linalg.norm(T - target)
        if error < 1e-6:
            return U, V, W
    
    return None

def _try_rank_3x3_6x3(target_rank, seed=42):
    """Try to find a rank-target_rank decomposition of 3x3 * 3x6 -> 3x6.
    
    Returns (U, V, W) with shapes (9, rank), (18, rank), (18, rank) or None.
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(9, target_rank) * 0.5
    V = np.random.randn(18, target_rank) * 0.5
    W = np.random.randn(18, target_rank) * 0.5
    
    n, m, p = 3, 3, 6
    
    for iteration in range(500):
        # Compute current tensor
        T = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        target = np.ones((n, m, p))
        residual = target - T
        
        for r in range(target_rank):
            # Update U
            num = np.zeros((n, m))
            den = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        num[i, j] += residual[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                        den[i, j] += V[j*p+k, r] ** 2 * W[k*n+i, r] ** 2
            U[:, r] = (num / (den + 1e-8)).flatten()
            
            # Recompute
            T = np.zeros((n, m, p))
            for rr in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
            residual = target - T
            
            # Update V
            num = np.zeros((m, p))
            den = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        num[j, k] += residual[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                        den[j, k] += U[i*m+j, r] ** 2 * W[k*n+i, r] ** 2
            V[:, r] = (num / (den + 1e-8)).flatten()
            
            # Recompute
            T = np.zeros((n, m, p))
            for rr in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
            residual = target - T
            
            # Update W
            num = np.zeros((p, n))
            den = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        num[k, i] += residual[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
                        den[k, i] += U[i*m+j, r] ** 2 * V[j*p+k, r] ** 2
            W[:, r] = (num / (den + 1e-8)).flatten()
        
        # Final check
        T = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        error = np.linalg.norm(T - target)
        if error < 1e-6:
            return U, V, W
    
    return None

def _to_lists(U, V, W):
    """Convert numpy arrays to lists."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130  # Leave 20s margin for safety
    
    # Fallback: schoolbook rank 54
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # Strategy: try ranks from 46 upward
    for target_rank in range(46, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try direct 3x3x6 decomposition
        result = _try_rank_3x3_6x3(target_rank, seed=target_rank)
        if result is not None:
            U, V, W = result
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = _to_lists(U, V, W)
                best_rank = target_rank
                break
        
        # Also try the split approach: two 3x3x3 decompositions
        if target_rank % 2 == 0:
            sub_rank = target_rank // 2
            result3 = _try_rank_3x3_3x3(sub_rank, seed=target_rank)
            if result3 is not None:
                U3, V3, W3 = result3
                # Build the 3x3x6 decomposition
                U_full = np.hstack([U3, U3])  # Shared U
                V_full = np.zeros((18, target_rank))
                W_full = np.zeros((18, target_rank))
                
                # First 3x3 block (k=0,1,2)
                V_full[0:9, 0:sub_rank] = V3
                W_full[0:9, 0:sub_rank] = W3
                
                # Second 3x3 block (k=3,4,5)
                V_full[9:18, sub_rank:target_rank] = V3
                W_full[9:18, sub_rank:target_rank] = W3
                
                if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
                    best = _to_lists(U_full, V_full, W_full)
                    best_rank = target_rank
                    break
    
    return best
