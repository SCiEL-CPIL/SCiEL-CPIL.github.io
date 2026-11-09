"""Direct approach for 3x3 by 3x6 matrix multiplication tensor decomposition.

Strategy:
  1. Build the schoolbook (rank-54) decomposition as a guaranteed fallback.
  2. Attempt to find a lower-rank exact decomposition via:
     a. Known rank-23 decomposition for 3x3*3x3 (Laderman + refinements)
        combined with the 3x6 structure.
     b. Numerical search using gradient-based optimisation on the factors.
  3. Verify any candidate by reconstructing the tensor and checking exactness.
  4. Return the best valid decomposition found.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
TIME_LIMIT = 115  # safety margin before 120s kill


def _build_schoolbook(n=3, m=3, p=6):
    """Return (U, V, W) for the schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _tensor_from_factors(U, V, W):
    """Reconstruct the tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    a_dim = U.shape[0]
    b_dim = V.shape[0]
    c_dim = W.shape[0]
    r = U.shape[1]
    T = np.zeros((a_dim, b_dim, c_dim), dtype=np.float64)
    for r_idx in range(r):
        T += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
    return T


def _target_tensor(n=3, m=3, p=6):
    """Build the target multiplication tensor T[a,b,c] where
       a = i*m+j (A index), b = j'*p+k (B index), c = k'*n+i' (output index).
       T[a,b,c] = 1 iff j == j' (shared contraction index matches).
    """
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    T = np.zeros((a_dim, b_dim, c_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, target, tol=1e-9):
    """Check that the factorisation reproduces the target tensor exactly."""
    T = _tensor_from_factors(U, V, W)
    return np.allclose(T, target, atol=tol, rtol=tol)


def _try_laderman_based(n=3, m=3, p=6):
    """
    Attempt to construct a lower-rank decomposition using known results.
    
    For 3x3 * 3x3, the minimum rank is 23 (Laderman 1977, improved by Strassen).
    For 3x3 * 3x6, we can view B as a 3x3 block matrix [B1, B2] where B1, B2 are 3x3.
    Then C = A * B = A*B1 + A*B2.
    
    If we have a rank-23 decomposition for 3x3*3x3, we can use it twice (for B1 and B2),
    giving rank 46. But we can potentially share some terms.
    
    Actually, a better approach: the 3x3*3x6 multiplication can be seen as computing
    18 output entries, each requiring 3 multiplications. The tensor has structure.
    
    Let me try a different approach: use the fact that the problem is equivalent to
    computing 3 independent 1x3 * 3x6 products (one per row of A).
    Each 1x3 * 3x6 product has rank 18 (schoolbook) but can be done in rank... 
    Actually 1x3 * 3x6: C[k] = sum_j A[j]*B[j,k]. This is a dot product of length 3
    applied to 6 columns. The tensor for this is rank 3*6 = 18? No, it's rank 18
    in schoolbook but the minimum is actually 18 since each output depends on all 3 inputs.
    
    Wait, for a single 1x3 * 3x6: the tensor T[j, j', k] = 1 if j == j'. This is
    just the identity in the j dimension, so rank = 3 (one per j). No wait, the output
    has 6 components. T[j, j', k] = delta_{j,j'} for j,j' in {0,1,2}, k in {0..5}.
    This tensor has rank 3 (sum over j of e_j ⊗ e_j ⊗ (1,1,1,1,1,1)^T).
    
    So for 3 rows of A, total rank = 3 * 3 = 9? That can't be right because we need
    to handle the full 3x3 * 3x6.
    
    Let me reconsider. The full tensor T[(i,j), (j',k), (k',i')] = 1 iff j=j' AND i=i' AND k=k'.
    Wait no. Let me re-derive.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The bilinear form has terms A[i,j] * B[j,k] contributing to C[i,k].
    So the tensor T has a 1 at position ((i,j), (j,k), (i,k)) for each (i,j,k).
    In the indexing convention: a = i*m+j, b = j*p+k, c = k*n+i.
    
    So T[a,b,c] = 1 iff there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i.
    This means j (from a) must equal j (from b), i (from a) must equal i (from c),
    and k (from b) must equal k (from c).
    
    So T[(i,j),(j',k),(k',i')] = 1 iff j=j' AND i=i' AND k=k'.
    
    This is a sum of 54 rank-1 tensors (one per (i,j,k) triple).
    
    Can we do better? The key question is whether these 54 rank-1 tensors are linearly independent.
    If they are, the rank is exactly 54. If not, we might find a lower-rank representation.
    
    For now, let's try a numerical approach to see if we can find a lower rank.
    """
    return None  # Fall through to numerical search


def _numerical_search(n=3, m=3, p=6, max_rank=53, time_limit=110):
    """
    Try to find a lower-rank decomposition using numerical optimisation.
    Start with the schoolbook factors and try to reduce rank by merging terms.
    """
    start_time = time.time()
    target = _target_tensor(n, m, p)
    
    # Try ranks from 53 down to some lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But realistically, for this structured problem, we might find something below 54.
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 54
    
    # First, get the schoolbook as baseline
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    if _verify(U_fb, V_fb, W_fb, target):
        best_U, best_V, best_W = U_fb.copy(), V_fb.copy(), W_fb.copy()
        best_rank = n * m * p
    
    # Try to find lower rank by optimisation
    # Strategy: for each target rank r < 54, try to find factors U (9 x r), V (18 x r), W (18 x r)
    # such that the reconstructed tensor matches the target.
    
    for r in range(53, 17, -1):  # Try ranks 53 down to 18
        if time.time() - start_time > time_limit:
            break
        if r >= best_rank:
            continue
        
        # Initialise with a random perturbation of a subset of schoolbook terms
        # Or use a smarter initialisation
        U_init = np.zeros((n * m, r), dtype=np.float64)
        V_init = np.zeros((m * p, r), dtype=np.float64)
        W_init = np.zeros((p * n, r), dtype=np.float64)
        
        # Copy first r schoolbook terms as initialisation
        for c in range(min(r, n * m * p)):
            i = c // (m * p)
            rem = c % (m * p)
            j = rem // p
            k = rem % p
            U_init[i * m + j, c] = 1.0
            V_init[j * p + k, c] = 1.0
            W_init[k * n + i, c] = 1.0
        
        # Add small random noise to help escape local minima
        U_init += 0.01 * np.random.randn(n * m, r)
        V_init += 0.01 * np.random.randn(m * p, r)
        W_init += 0.01 * np.random.randn(p * n, r)
        
        # Gradient descent on the loss ||T_factors - T_target||^2
        U_cur = U_init.copy()
        W_cur = W_init.copy()
        V_cur = V_init.copy()
        
        lr = 0.01
        iters = 200
        for it in range(iters):
            if time.time() - start_time > time_limit:
                break
            
            T_cur = _tensor_from_factors(U_cur, V_cur, W_cur)
            loss = np.sum((T_cur - target) ** 2)
            
            if loss < 1e-12:
                break
            
            # Compute gradients
            # dL/dU[a,r] = sum_{b,c} (T_cur[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # dL/dV[b,r] = sum_{a,c} (T_cur[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
            # dL/dW[c,r] = sum_{a,b} (T_cur[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
            
            diff = T_cur - target
            gU = np.einsum('abc,br,cr->ar', diff, V_cur, W_cur)
            gV = np.einsum('abc,ar,cr->br', diff, U_cur, W_cur)
            gW = np.einsum('abc,ar,br->cr', diff, U_cur, V_cur)
            
            U_cur -= lr * gU
            V_cur -= lr * gV
            W_cur -= lr * gW
            
            # Adaptive learning rate
            if it % 20 == 0 and it > 0:
                new_loss = np.sum((_tensor_from_factors(U_cur, V_cur, W_cur) - target) ** 2)
                if new_loss > loss * 1.1:
                    lr *= 0.5
                elif new_loss < loss * 0.9:
                    lr *= 1.1
        
        T_final = _tensor_from_factors(U_cur, V_cur, W_cur)
        final_loss = np.sum((T_final - target) ** 2)
        
        if final_loss < 1e-9:
            # Verify more carefully
            if _verify(U_cur, V_cur, W_cur, target, tol=1e-9):
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                best_rank = r
                break  # Found a valid lower-rank decomposition
    
    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build fallback
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    
    # Try numerical search for lower rank
    try:
        U_opt, V_opt, W_opt, opt_rank = _numerical_search(n, m, p, time_limit=TIME_LIMIT - 5)
        if U_opt is not None and opt_rank < n * m * p:
            # Convert to lists for the return interface
            return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    except Exception:
        pass
    
    # Return schoolbook fallback
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


# EVOLVE-BLOCK-START
# (construct is already defined above; this block is kept for interface compatibility)
# EVOLVE-BLOCK-END
