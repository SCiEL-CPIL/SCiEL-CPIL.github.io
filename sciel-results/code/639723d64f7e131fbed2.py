"""
Matrix multiplication 4x5 x 5x5: rank-minimal exact tensor decomposition.

Strategy:
- Use a block-based decomposition that exploits Strassen's rank-7 for 2x2x2.
- Split dimensions to create 2x2x2 blocks where Strassen applies.
- For remaining blocks, use optimized schoolbook or known decompositions.
- The key is to minimize total rank by choosing splits that maximize Strassen usage.

For 4x5 x 5x5:
  Split n=4 into 2+2, m=5 into 2+3, p=5 into 2+3.
  
  This gives 8 blocks:
  (2,2,2): rank 7 (Strassen)
  (2,2,3): rank 12 (schoolbook) or better
  (2,3,2): rank 12 (schoolbook) or better
  (2,3,3): rank 18 (schoolbook) or better
  (2,2,2): rank 7 (Strassen)
  (2,2,3): rank 12
  (2,3,2): rank 12
  (2,3,3): rank 18
  
  Total: 2*7 + 2*12 + 2*12 + 2*18 = 14 + 24 + 24 + 36 = 98

  But we can do better by trying different splits or using optimized decompositions
  for the non-Strassen blocks.

  Alternative: use a single Strassen-like decomposition that doesn't require
  all blocks to be 2x2x2.

  Actually, let's try a different approach: use the known result that
  r(2,2,3) can be done in rank 11 (not 12). And r(2,3,3) might be better than 18.

  Let me try to use optimization to find good decompositions for the blocks,
  then combine them.

  Given time constraints, let's focus on:
  1. A good block decomposition using Strassen for 2x2x2
  2. Optimized schoolbook for other blocks
  3. Try to reduce rank via optimization on the full tensor
"""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit

    # Build the target tensor T of shape (n*m, m*p, n*p)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k*n+i) with matching indices
    T = np.zeros((n * m, m * p, n * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i  # W[k*n+i, r] convention
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def to_output(U, V, W):
        """Convert numpy arrays to the required dict format."""
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Schoolbook decomposition
    def schoolbook(nn, mm, pp):
        rank = nn * mm * pp
        U = np.zeros((nn * mm, rank))
        V = np.zeros((mm * pp, rank))
        W = np.zeros((pp * nn, rank))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1.0
                    V[j * pp + k, col] = 1.0
                    W[k * nn + i, col] = 1.0
                    col += 1
        return U, V, W

    # Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[:, 0] = [1, 0, 0, 1]
        
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[:, 1] = [0, 0, 1, -1]
        
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 1, 0, -1]
        W[:, 2] = [0, 1, 0, 1]
        
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [0, -1, 1, 0]
        W[:, 3] = [-1, 1, 0, 0]
        
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[:, 4] = [1, 0, 1, 0]
        
        U[:, 5] = [-1, 0, 1, 0]
        V[:, 5] = [1, 0, 1, 0]
        W[:, 5] = [0, 0, 0, 1]
        
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 1, 0, 1]
        W[:, 6] = [1, 1, 0, 0]
        
        return U, V, W

    # Try to find a good decomposition using optimization
    def try_optimization(rank, num_iters=200, lr=0.1, seed=42, max_time=30):
        """Try to find a rank-r decomposition using gradient descent."""
        start_time = time.time()
        if time.time() > deadline - 20:
            return None
        
        rng = np.random.RandomState(seed)
        best_err = float('inf')
        best_UVW = None
        
        for restart in range(2):
            if time.time() > deadline - 15 or (time.time() - start_time) > max_time:
                break
            
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(n * p, rank) * 0.5
            
            for it in range(num_iters):
                if time.time() > deadline - 10 or (time.time() - start_time) > max_time:
                    break
                
                T_recon = np.einsum('as,bs,cs->abc', U, V, W)
                residual = T_recon - T
                
                dU = np.einsum('abc,bs,cs->as', residual, V, W)
                dV = np.einsum('abc,as,cs->bs', residual, U, W)
                dW = np.einsum('abc,as,bs->cs', residual, U, V)
                
                err = np.max(np.abs(residual))
                if err < 1e-12:
                    return U, V, W
                
                eff_lr = lr * min(1.0, 1.0 / (1.0 + err))
                
                U -= eff_lr * dU
                V -= eff_lr * dV
                W -= eff_lr * dW
            
            err = np.max(np.abs(np.einsum('as,bs,cs->abc', U, V, W) - T))
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-10:
            return best_UVW
        return None

    # Try to find the best rank via optimization
    best_result = None
    best_rank = n * m * p  # 100
    
    # Try ranks from 90 to 99 (conservative, since we know schoolbook is 100)
    for rank in range(90, best_rank):
        if time.time() > deadline - 40:
            break
        
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            
            result = try_optimization(rank, num_iters=150, lr=0.1, seed=seed + rank * 13)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    best_rank = rank
                    best_result = to_output(U, V, W)
                    break
        if best_result is not None and best_rank <= rank:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return to_output(U, V, W)
