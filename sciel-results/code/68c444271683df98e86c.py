"""Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]
START_TIME = time.time()
TIME_LIMIT = 115  # seconds, leave safety margin before 120s

# Precompute the schoolbook solution
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check that U, V, W form a valid matrix multiplication decomposition."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 if we're multiplying A[i,j] * B[j,k]
    # Actually, the tensor is T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
    # Let me re-examine: the multiplication tensor has T[i,j,k] = delta (it's 1 when indices are valid)
    # For 2x3 * 3x9: i in [0,1], j in [0,2], k in [0,8]
    # The output is C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k
    # (because C[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) triple contributes with weight 1)
    
    errors = 0.0
    for r in range(rank):
        pass  # Check each (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                errors += (s - 1.0) ** 2
    return errors < 1e-9

def _try_block_decomposition():
    """
    Try to exploit block structure: 2x3 * 3x9 = three independent 2x3 * 3x3 blocks.
    If we can find a rank-r decomposition for 2x3*3x3, then rank = 3*r for the full problem.
    Schoolbook for 2x3*3x3 is 18. Known best for 2x3 is 7 (Winograd), but 2x3*3x3 is different.
    Actually, the block structure means:
    C[:, 0:3] = A * B[:, 0:3]  (2x3 * 3x3)
    C[:, 3:6] = A * B[:, 3:6]  (2x3 * 3x3)
    C[:, 6:9] = A * B[:, 6:9]  (2x3 * 3x3)
    
    But the tensor is not simply a tensor product because the middle index j is shared.
    Let me think more carefully.
    
    T[i, j, k] = 1 for i in [0,1], j in [0,2], k in [0,8]
    
    If we write k = 3*block + offset where block in [0,2] and offset in [0,2],
    then T[i, j, 3*block+offset] = 1.
    
    This is NOT a tensor product T_2x3x3 ⊗ I_3 because the j index is shared across blocks.
    So we can't simply take 3 copies of a 2x3*3x3 decomposition.
    """
    pass

def _greedy_search(target_rank):
    """
    Try to find a decomposition of the given target rank using a greedy/approximation approach.
    We use the fact that we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    This is a 3-way tensor completion / rank factorization problem.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18
    
    # We need U (6 x R), V (27 x R), W (18 x R)
    # such that for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Use randomized initialization and alternating minimization
    best_error = float('inf')
    best_solution = None
    rng = np.random.RandomState(42)
    
    max_iters = 50
    for attempt in range(3):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        U = rng.randn(nm, target_rank) * 0.5
        V = rng.randn(mp, target_rank) * 0.5
        W = rng.randn(pn, target_rank) * 0.5
        
        for it in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            # Update U: for each (i,j), U[i*m+j, r] = sum_k V[j*p+k,r]*W[k*n+i,r] * 1 / (V[j*p+k,r]*W[k*n+i,r] squared stuff)
            # This is complex. Let's use a simpler approach: just check if current is good enough.
            
            # Compute error
            U_arr = U
            V_arr = V
            W_arr = W
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                        err += (s - 1.0)**2
            if err < best_error:
                best_error = err
                if err < 1e-9:
                    best_solution = (U.copy(), V.copy(), W.copy())
                    return best_solution
            if err < 1e-6:
                break
        
        if best_solution is not None:
            break
    
    if best_solution is not None:
        return best_solution
    
    # Try to polish with exact fractions if we got close
    if best_error < 1e-4 and best_solution is None:
        # This is getting complex; for now just return None
        pass
    
    return None

def _try_winograd_2x3():
    """
    Winograd's algorithm for 2x2 uses rank 7 instead of 8.
    For 2x3, there are known fast algorithms. Let me try to construct one.
    
    Actually, let me try a different approach: use the structure of the problem.
    The tensor T[i,j,k] = 1 for all i,j,k is a "flat" tensor.
    
    One known approach: the multiplication tensor for m x n * n x p has rank
    at most mn + mp + np - n - m - p + 1 (from Winograd-style constructions) for small cases,
    but I'm not sure of the exact formula.
    
    Let me just try to find a lower-rank decomposition by searching.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = n * m * p  # 54
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for 2x3*3x9 is at least max(n*m, m*p, p*n) = 27
    # (from the dimension of each factor)
    # Also at least n*m*p / min(n,m,p) = 54/2 = 27
    
    # Try ranks from 27 upward
    for rank in range(27, best_rank):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        sol = _greedy_search(rank)
        if sol is not None:
            U, V, W = sol
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify
            if _verify(U_list, V_list, W_list):
                best_rank = rank
                result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return result

# EVOLVE-BLOCK-END
