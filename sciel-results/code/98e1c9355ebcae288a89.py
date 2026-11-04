"""
Schoolbook decomposition for 4x5 by 5x6 matrix multiplication,
with a local-search attempt to reduce rank.
"""
import time
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = 115.0
_SAFETY_MARGIN = 5.0  # seconds before hard kill

def _time_left():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _get_target_tensor(n, m, p):
    """Create the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Check exact decomposition. Returns True if valid."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Target = _get_target_tensor(n, m, p)
    # Check if T matches Target exactly (within tolerance)
    if not np.allclose(T, Target, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _try_local_search(n, m, p, target_rank, max_iters=5000):
    """Try to find a decomposition of rank target_rank via random restart + local search."""
    if _time_left() < 2.0:
        return None
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    best_err = float('inf')
    best = None
    rng = np.random.RandomState(42)
    for restart in range(5):
        if _time_left() < 2.0:
            break
        U = rng.randint(0, 2, size=(dim_u, target_rank))
        V = rng.randint(0, 2, size=(dim_v, target_rank))
        W = rng.randint(0, 2, size=(dim_w, target_rank))
        # Coordinate descent
        for it in range(max_iters):
            if _time_left() < 2.0:
                break
            # Pick random variable to update
            var_type = rng.randint(0, 3)  # 0=U, 1=V, 2=W, 3=skip
            if var_type == 0:
                idx = rng.randint(0, dim_u)
                col = rng.randint(0, target_rank)
                old = U[idx, col]
                for new_val in range(0, 3):
                    U[idx, col] = new_val
                    T = np.einsum('ar,br,cr->abc', U, V, W)
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; c = k*n+i
                                err += abs(T[a,b,c] - 1.0)
                    if err < best_err:
                        best_err = err
                        best = (U.copy(), V.copy(), W.copy())
                        if best_err < 1e-9:
                            U, V, W = best
                            return U.tolist(), V.tolist(), W.tolist()
                    else:
                        U[idx, col] = old
            elif var_type == 1:
                idx = rng.randint(0, dim_v)
                col = rng.randint(0, target_rank)
                old = V[idx, col]
                for new_val in range(0, 3):
                    V[idx, col] = new_val
                    T = np.einsum('ar,br,cr->abc', U, V, W)
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; c = k*n+i
                                err += abs(T[a,b,c] - 1.0)
                    if err < best_err:
                        best_err = err
                        best = (U.copy(), V.copy(), W.copy())
                        if best_err < 1e-9:
                            U, V, W = best
                            return U.tolist(), V.tolist(), W.tolist()
                    else:
                        V[idx, col] = old
            elif var_type == 2:
                idx = rng.randint(0, dim_w)
                col = rng.randint(0, target_rank)
                old = W[idx, col]
                for new_val in range(0, 3):
                    W[idx, col] = new_val
                    T = np.einsum('ar,br,cr->abc', U, V, W)
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; c = k*n+i
                                err += abs(T[a,b,c] - 1.0)
                    if err < best_err:
                        best_err = err
                        best = (U.copy(), V.copy(), W.copy())
                        if best_err < 1e-9:
                            U, V, W = best
                            return U.tolist(), V.tolist(), W.tolist()
                    else:
                        W[idx, col] = old
    return None

def construct():
    n, m, p = 4, 5, 6
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to improve via local search for lower ranks
    # We only try a few ranks below schoolbook to stay within time limits
    # Schoolbook rank is 120. We try down to 115.
    for target_rank in range(best_rank - 1, max(best_rank - 5, 100), -1):
        if _time_left() < 10.0:
            break
        # Use fewer iterations to ensure speed
        result = _try_local_search(n, m, p, target_rank, max_iters=100)
        if result is not None:
            U2, V2, W2 = result
            if _verify(U2, V2, W2, n, m, p):
                best_rank = target_rank
                best_result = {"U": U2, "V": V2, "W": W2}
                break
    
    return best_result

# EVOLVE-BLOCK-END
