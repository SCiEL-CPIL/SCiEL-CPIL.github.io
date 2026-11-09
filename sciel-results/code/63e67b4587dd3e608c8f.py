"""Matrix multiplication 3x3 x 3x6 rank-minimal decomposition.

Strategy:
  - The 3x3x3 matrix multiplication tensor has known rank 23 (Bini-Capovani-Falletti-Roncati).
  - We construct a candidate rank-23 decomposition for 3x3x3 using a systematic search
    over the space of 0/1/-1 coefficient matrices, verifying the tensor identity.
  - Then we extend to 3x3x6 by combining the rank-23 3x3x3 decomposition with a
    trivial rank-3 decomposition for multiplying a 3x1 vector by a 1x6 matrix.
  - Total candidate rank: 23 * 3 = 69? No, that's not right.
  
  Actually, let me think more carefully. The tensor for 3x3 * 3x6 multiplication has
  dimensions (3*3) x (3*6) x (3*6) = 9 x 18 x 18.
  
  The schoolbook rank is 3*3*6 = 54.
  
  A better approach: The 3x3 * 3x6 multiplication can be viewed as computing C = A*B
  where A is 3x3 and B is 3x6. This is equivalent to computing 6 columns of the 3x3
  matrix product. Each column c_k = A * b_k where b_k is a 3x1 vector.
  
  So C[:,k] = A * B[:,k] for k=0,1,2,3,4,5.
  
  The tensor is: sum over i,j,k of A[i,j] * B[j,k] * e_{i,k}
  
  This is a 9 x 18 x 18 tensor (or equivalently, the bilinear map from 9-dim and 18-dim
  to 18-dim space).
  
  One approach: Use the rank-23 algorithm for 3x3x3, and for each of the 6 output columns,
  we need to compute A * b_k. But that would give rank 23*6 = 138, which is worse.
  
  Better approach: The tensor T_{i,j,k} = A[i,j] * B[j,k] can be decomposed as:
  T = sum_r u_r ⊗ v_r ⊗ w_r
  
  where u_r is in R^9 (indexed by (i,j)), v_r is in R^18 (indexed by (j,k)), w_r is in R^18 (indexed by (i,k)).
  
  We can try to find a low-rank decomposition directly.
  
  Let me try a different approach: use the fact that the 3x3x3 tensor has rank 23, and see
  if we can leverage that structure.
  
  Actually, the simplest correct approach that might beat schoolbook:
  
  Consider that B is 3x6. We can write B = [B1 | B2] where B1 and B2 are 3x3.
  Then C = [A*B1 | A*B2].
  
  So we need to compute two 3x3 matrix products. If each has rank 23, total rank = 46.
  
  This gives rank 46 < 54, so score = 54/46 ≈ 1.17.
  
  Let me verify: The tensor for 3x3 * 3x3 has rank 23. The tensor for 3x3 * 3x6 is
  the direct sum of two 3x3 * 3x3 tensors (since B = [B1|B2] and C = [A*B1 | A*B2]).
  
  Wait, that's not quite right. The input A is shared. Let me think again.
  
  The bilinear map is: (A, B) -> C where A is 3x3, B is 3x6, C is 3x6.
  
  C[:, 0:3] = A * B[:, 0:3] and C[:, 3:6] = A * B[:, 3:6].
  
  So the tensor is the direct sum of two 3x3x3 multiplication tensors, but they share the
  same A input. The rank of a direct sum of tensors T1 and T2 is at most rank(T1) + rank(T2),
  and can be less if there's sharing.
  
  In this case, since A is shared, we can't simply say rank = 23 + 23 = 46. We need to be
  more careful.
  
  Actually, the tensor T_{(i,j),(j,k),(i,k)} for the full 3x3*3x6 problem can be written as:
  T = T1 + T2 where T1 handles k=0,1,2 and T2 handles k=3,4,5.
  
  T1_{(i,j),(j,k),(i,k)} = 1 if k<3 else 0, and it's the 3x3x3 tensor for k=0,1,2.
  T2_{(i,j),(j,k),(i,k)} = 1 if k>=3 else 0, and it's the 3x3x3 tensor for k=3,4,5.
  
  Since T1 and T2 have disjoint support in the output index (i,k), rank(T1+T2) <= rank(T1)+rank(T2) = 23+23 = 46.
  
  But can we do better? The A input is shared, so maybe we can save some rank.
  
  For now, let me just implement rank 46 by using two rank-23 decompositions.
  
  I need the actual rank-23 decomposition for 3x3x3. Let me recall or derive it.
  
  The BCFR rank-23 algorithm for 3x3 matrix multiplication:
  
  The 23 terms are given by (u_r, v_r, w_r) where each is a 3x3 matrix (flattened to 9 entries).
  
  Let me try to construct it. Actually, I don't have the exact 23-term decomposition memorized.
  
  Alternative: I can try to find a low-rank decomposition numerically using a search or optimization.
  
  Given the 150-second time limit, let me try a numerical approach:
  1. Start with the schoolbook decomposition (rank 54).
  2. Try to reduce the rank by finding linear dependencies among the terms.
  
  Or, I can try to use a known result: the rank of 3x3x3 is 23. Let me try to find it.
  
  Actually, let me just implement the rank-46 approach by using two copies of a rank-23
  decomposition. But I need the rank-23 decomposition.
  
  Let me try a different approach: use numerical linear algebra to find a low-rank
  approximation and then round to integers/half-integers.
  
  Here's my plan:
  1. Construct the 9x18x18 tensor.
  2. Use a rank-reduction algorithm to find a low-rank decomposition.
  3. Round to half-integers and verify.
  
  Given time constraints, let me try a simpler approach:
  
  I'll use the fact that the 3x3*3x6 tensor can be decomposed as the sum of two
  3x3*3x3 tensors. If I can find a rank-23 decomposition for 3x3*3x3, I can get rank 46.
  
  Let me try to find the rank-23 decomposition using a search. The space of 0/1/-1
  matrices for 3x3 is finite but large. Let me try a randomized approach.
  
  Actually, let me just try to implement a known rank-23 algorithm. I'll use the
  decomposition from the literature.
  
  After some thought, let me try a numerical approach:
  1. Build the tensor.
  2. Use ALS (Alternating Least Squares) to find a low-rank decomposition.
  3. Round and verify.
  
  Let me code this up.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
_DEADLINE_SECONDS = 140

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build the tensor from U, V, W
    # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1  # A[i,j] * B[j,k] contributes 1 to C[i,k]
                actual = 0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _als_decompose(T, rank, max_iter=1000, tol=1e-8, seed=42):
    """Alternating Least Squangles decomposition of a 3D tensor."""
    np.random.seed(seed)
    d1, d2, d3 = T.shape
    U = np.random.randn(d1, rank)
    V = np.random.randn(d2, rank)
    W = np.random.randn(d3, rank)
    
    for it in range(max_iter):
        # Update U
        for i in range(d1):
            # T[i,:,:] = sum_r U[i,r] * V[:,r] * W[:,r] (outer product)
            # Fix V and W, solve for U[i,:]
            # T[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r])
            # This is: T[i,:,:].reshape(-1) = U[i,:] @ (V * W).T (element-wise)
            VW = (V * W).reshape(rank, d2 * d3)  # r x (d2*d3)
            rhs = T[i, :, :].reshape(-1)
            # Solve: rhs = U[i,:] @ VW.T => U[i,:] = rhs @ VW / (VW @ VW.T)
            if rank > 0:
                U[i, :] = np.linalg.lstsq(VW, rhs, rcond=None)[0]
        
        # Update V
        for j in range(d2):
            UW = (U * W).reshape(rank, d1 * d3)
            rhs = T[:, j, :].reshape(-1)
            if rank > 0:
                V[j, :] = np.linalg.lstsq(UW, rhs, rcond=None)[0]
        
        # Update W
        for k in range(d3):
            UV = (U * V).reshape(rank, d1 * d2)
            rhs = T[:, :, k].reshape(-1)
            if rank > 0:
                W[k, :] = np.linalg.lstsq(UV, rhs, rcond=None)[0]
        
        if it % 50 == 0:
            # Compute residual
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            res = np.linalg.norm(T - recon)
            if res < tol:
                break
    
    return U, V, W

def _round_to_half_int(A):
    """Round matrix entries to nearest half-integer."""
    return np.round(2 * A) / 2.0

def _check_half_integer(A):
    """Check if all entries are half-integers."""
    return np.allclose(2 * A, np.round(2 * A), atol=1e-6)

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition
    # Start with rank 46 (two 3x3x3 decompositions)
    # But first, let's try to find a lower rank
    
    T = _build_tensor(n, m, p)
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p  # schoolbook
    
    # Try ranks from 23*2=46 down to 23
    for rank in range(46, 22, -1):
        if time.time() - t0 > _DEADLINE_SECONDS - 20:
            break
        
        # Try multiple random initializations
        for seed in range(10):
            if time.time() - t0 > _DEADLINE_SECONDS - 10:
                break
            
            U, V, W = _als_decompose(T, rank, max_iter=200, tol=1e-8, seed=seed)
            
            # Round to half-integers
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            
            # Check if rounding preserved the decomposition
            if _check_half_integer(U_r) and _check_half_integer(V_r) and _check_half_integer(W_r):
                # Verify
                recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.allclose(recon, T, atol=1e-6):
                    # Convert to lists
                    best_U = U_r.tolist()
                    best_V = V_r.tolist()
                    best_W = W_r.tolist()
                    best_rank = rank
                    break
        
        if best_rank < n * m * p:
            break
    
    # If no low-rank decomposition found, use schoolbook
    if best_U is None:
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
    
    return {"U": best_U, "V": best_V, "W": best_W}
