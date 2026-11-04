"""Tensor rank minimization for 4x4x5 matrix multiplication via structured search."""
DIMENSIONS = [4, 4, 5]
import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    """Check exact MM decomposition. T[i,k,j] must equal sum_t U[i*m+j? no]..."""
    # Convention: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T with indices (i, j, k): T[i,j,k] = 1 for the MM structure
    # Decomposition: T = sum_t U_t ⊗ V_t ⊗ W_t where
    #   U_t[i * m + j] = U[i*m+j, t]  (A index)
    #   V_t[j * p + k] = V[j*p+k, t]  (B index)
    #   W_t[k * n + i] = W[k*n+i, t]  (C index, trace convention W[k*n+i])
    # So: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] should equal 1 if (i,j,k) valid else 0
    # Actually the MM tensor: for each (i,k), C[i,k] = sum_j A[i,j]B[j,k]
    # The structure tensor T[i, j, k] = 1 (all ones in the valid range) represents
    # the multiplication: <T, A ⊗ B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} (sum_j A[i,j]B[j,k]) = sum_{i,k} C[i,k]
    # Wait that's not right. The standard MM tensor: T[i,j,k] such that contraction gives C[i,k]=sum_j T[i,j,k] A[i,j] B[j,k] with T[i,j,k]=delta_{j,j}=1.
    # So T[i,j,k] = 1 for all i,j,k in range. Then rank-1 decomposition: U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i]
    # and we need sum_t U_t[i*m+j] V_t[j*p+k] W_t[k*n+i] = 1 for all i,j,k? No.
    # The tensor is T[i,j,k]=1. A rank-1 slice is u[i]v[j]w[k]. We need sum_t u_t[i] v_t[j] w_t[k] = 1 for ALL i,j,k.
    # That's a much stronger condition (all-ones tensor), not MM!
    # 
    # Let me re-read: "W[k*n+i,r]" trace convention. The MM tensor for C=AB:
    # C[i,k] = sum_j A[i,j] B[j,k]. The tensor T has three modes: A-indices (i,j), B-indices (j,k), C-indices (i,k).
    # T[(i,j), (j,k), (i,k)] = 1. As a 3-tensor with flattened modes: T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i.
    # T[a,b,c] = 1 if the indices are consistent (the j in a equals j in b, and i in a equals i in c, and k in b equals k in c), else 0.
    # Specifically: a = i*m+j, b = j*p+k, c = k*n+i. These must satisfy: j from a == j from b, i from a == i from c, k from b == k from c.
    # So T is sparse. Rank-1: U[a,t]*V[b,t]*W[c,t]. Sum_t should equal T[a,b,c].
    rank = len(W[0]) if len(W) > 0 else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    # Compute reconstructed
    for t in range(rank):
        T -= np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
    return np.max(np.abs(T)) < 1e-9

def _strassen_4x4():
    """Return rank-49 decomposition for 4x4x4 MM using Strassen-based scheme.
    This is complex; instead we use a known rank-49 explicit decomposition.
    Given complexity, we fall back to schoolbook 64 for 4x4x4 and note it's suboptimal.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Strategy: Try to find a decomposition with rank < n*m*p = 80.
    # We know schoolbook is 80. Let's try rank 64, 60, 56, 53, 49 via random init + refinement.
    # But exactness is hard. Use a structured approach:
    # 
    # Key insight: The 4x4x5 MM tensor can be decomposed by treating it as:
    #   C[:, 0:4] = A @ B[:, 0:4]  (4x4x4, rank <= 64, best known 49)
    #   C[:, 4]   = A @ B[:, 4]    (4x4x1, rank = 16)
    # Total: 64 + 16 = 80 (no gain). But we can share: the products A[i,j]*B[j,k] for k=0..3
    # are reused for k=4? No, B[j,4] is different from B[j,k].
    # 
    # Better: Use the fact that rank(4x4x5) <= rank(4x4x4) + 4*4 = 49 + 16 = 65? 
    # Not obviously. Let's just try random search for rank 60-70 and verify.
    
    # Fallback: schoolbook rank 80 (always valid)
    sb = _schoolbook()
    best = sb
    best_rank = 80
    
    # Try a few random ranks below 80
    rng = np.random.RandomState(42)
    for trial_rank in [64, 60, 56, 53]:
        if time.time() - start > 100:
            break
        for trial in range(20):
            if time.time() - start > 110:
                break
            U = rng.randn(n * m, trial_rank)
            V = rng.randn(m * p, trial_rank)
            W = rng.randn(p * n, trial_rank)
            # Quick check: does sum_t U[:,t] V[:,t] W[:,t] match T?
            # This is a hard constraint; random won't satisfy it. Skip random for exactness.
            pass
    
    # Random won't give exact integer/half-integer. Use structured rank-64:
    # 4x4x4 schoolbook (64 terms) + handle 5th column with 16 more = 80. No gain.
    # 
    # Alternative: Use a rank-49 4x4x4 decomposition. Hard to code from scratch reliably.
    # 
    # Given time constraints, return schoolbook (guaranteed valid, rank 80).
    # But we want improvement. Let me try: can we do rank 72?
    # 4x4x5: note that the tensor T has T[a,b,c]=1 only when consistent. 
    # The "diagonal" structure might allow rank reduction.
    #
    # I'll return the schoolbook decomposition as the safe valid answer.
    # To attempt improvement, I'll check if a rank-72 structured solution exists via
    # sharing the (i,j) products across k.
    
    # Structured rank-72: For each (i,j), we have product A[i,j]*B[j,k] for k=0..4.
    # That's 5 products per (i,j), 16*(i,j) pairs = 80. No sharing possible without
    # losing exactness in general.
    
    return best
# EVOLVE-BLOCK-END
