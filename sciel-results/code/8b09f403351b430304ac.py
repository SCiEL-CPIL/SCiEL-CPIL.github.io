"""Matrix multiplication 3x3 by 3x7 - rank minimization via structural factorization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """Verify that the decomposition correctly computes C = A*B."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all (i,j,k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """Fast vectorized verification."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Build tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r].reshape(n, m)
        v = V_arr[:, r].reshape(m, p)
        w = W_arr[:, r].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
    
    return np.allclose(T, np.ones((n,m,p)), atol=tol)

def _schoolbook(n=3, m=3, p=7):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_based(n=3, m=3, p=7):
    """
    Attempt a Strassen-inspired decomposition.
    
    For 3x3 * 3x3, Strassen uses rank 7.
    For 3x3 * 3x7, we can decompose B into 3x3 and 3x4 blocks.
    
    Strategy: Use Strassen for the 3x3 part and schoolbook for the 3x4 part,
    then try to merge/optimize.
    
    Actually, let's try a different approach: use the fact that for the full
    tensor T[i,j,k]=1, we can try to find a lower rank by using non-trivial
    linear combinations.
    
    For now, try a structured search with small integer/half-integer values.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try ranks from a reasonable lower bound up
    # Lower bound for 3x3x7: at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But actual lower bound is higher. Let's try from 21 upward.
    
    best = None
    best_rank = n * m * p  # 63
    
    # Try some structured patterns
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try a few structured initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize with a pattern that might work
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Use a cyclic pattern based on Strassen-like ideas
            for r in range(target_rank):
                # Distribute the "mass" across entries
                # Use a hash-like pattern
                seed_val = r * 7 + 1
                for idx in range(n*m):
                    # Use a pseudo-random but deterministic pattern
                    if (idx * 31 + r * 17 + 5) % 13 < 7:
                        U[idx, r] = 1 if (idx + r) % 2 == 0 else -1
                
                for idx in range(m*p):
                    if (idx * 23 + r * 11 + 3) % 11 < 6:
                        V[idx, r] = 1 if (idx + r) % 2 == 0 else -1
                
                for idx in range(p*n):
                    if (idx * 19 + r * 13 + 7) % 17 < 9:
                        W[idx, r] = 1 if (idx + r) % 2 == 0 else -1
            
            # Verify
            U_list = [[int(U[i, r]) for r in range(target_rank)] for i in range(n*m)]
            V_list = [[int(V[i, r]) for r in range(target_rank)] for i in range(m*p)]
            W_list = [[int(W[i, r]) for r in range(target_rank)] for i in range(p*n)]
            
            if _verify_fast(U_list, V_list, W_list):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best is not None:
            break
    
    return best, best_rank

def _try_linear_combination(n=3, m=3, p=7):
    """
    Try to find a lower rank by expressing the tensor as a sum of rank-1 terms
    using linear algebra.
    
    The tensor T has shape (n*m, m*p, p*n) in flattened form, but really
    T[i,j,k] = 1 for all valid indices.
    
    We want: T = sum_r u_r v_r w_r where u_r in R^(n*m), v_r in R^(m*p), w_r in R^(p*n)
    
    This is equivalent to finding the tensor rank.
    
    For small cases, we can try to solve a system of equations.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 3, 3, 7
    num_eq = n * m * p  # 63 equations
    
    # For rank r, we have r unknown vectors of lengths n*m, m*p, p*n
    # Total unknowns: r * (n*m + m*p + p*n) = r * (9 + 21 + 21) = r * 51
    # We need r * 51 >= 63, so r >= ceil(63/51) = 2. But this is a very loose bound.
    
    # The actual tensor rank is much higher. Let's try a practical approach.
    
    # Try to use the schoolbook and see if we can merge some terms.
    # For two schoolbook terms (i1,j1,k1) and (i2,j2,k2), can we replace them
    # with a single term?
    
    # This requires finding u, v, w such that:
    # u[i1,j1]*v[j1,k1]*w[k1,i1] + u[i2,j2]*v[j2,k2]*w[k2,i2] = 1 for those (i,j,k)
    # and 0 for all others (in the combined support)
    
    # This is generally not possible for arbitrary pairs, but let's try specific cases.
    
    # For now, just return None and let the main function use schoolbook.
    return None, n * m * p

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try to find a better decomposition
    best, best_rank = _try_strassen_based(n, m, p)
    
    if best is not None and best_rank < n * m * p:
        return best
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
