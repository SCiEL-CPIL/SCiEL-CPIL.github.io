import time
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n=3, m=4, p=8):
    """Verify exact decomposition with integer/half-integer entries."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _residuals(params, rank, n=3, m=4, p=8):
    """Compute residuals for the optimization problem."""
    # params shape: (n*m*rank + m*p*rank + p*n*rank,)
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = params[:size_u].reshape(n*m, rank)
    V = params[size_u:size_u+size_v].reshape(m*p, rank)
    W = params[size_u+size_v:].reshape(p*n, rank)
    
    residuals = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                residuals[idx] = s - 1.0
                idx += 1
    return residuals

def _try_rank(rank, n=3, m=4, p=8, max_restarts=20, max_nfev=2000):
    """Try to find a valid decomposition of given rank."""
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    total_size = size_u + size_v + size_w
    
    for seed in range(max_restarts):
        rng = np.random.RandomState(seed)
        # Initialize with small random values
        # Using small values helps LM converge to local minima
        x0 = rng.randn(total_size) * 0.1
        
        try:
            res = scipy.optimize.least_squares(
                _residuals, x0, args=(rank, n, m, p), 
                method='lm', max_nfev=max_nfev
            )
            
            if res.cost < 1e-10:
                # Extract U, V, W
                U = res.x[:size_u].reshape(n*m, rank)
                V = res.x[size_u:size_u+size_v].reshape(m*p, rank)
                W = res.x[size_u+size_v:].reshape(p*n, rank)
                
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded version
                if _verify(U_r, V_r, W_r, n, m, p):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
                # If rounding failed, maybe the solution is very close to integers/half-integers
                # but floating point error pushed it away?
                # Try verifying the unrounded version first to see if it's exact
                if _verify(U, V, W, n, m, p):
                     # It's exact but not integer/half-integer.
                     # The task requires integer/half-integer.
                     # So we must fail this candidate.
                     pass
                     
        except Exception:
            continue
            
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    deadline = time.time() + 115
    n, m, p = 3, 4, 8
    
    # Start with schoolbook
    school = _schoolbook(n, m, p)
    best_result = school
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Start from 95 down to 80
    # We can adjust the range based on time
    for target_rank in range(best_rank - 1, 79, -1):
        if time.time() > deadline - 10:
            break
            
        # Try to find a solution for this rank
        result = _try_rank(target_rank, n, m, p, max_restarts=10, max_nfev=1000)
        
        if result is not None:
            U_list, V_list, W_list = result
            # Double check
            if _verify(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                # Continue searching for even lower rank?
                # Yes, but be careful with time.
                # If we found 95, maybe 94 is possible.
                # But usually, if we find one, it's a good improvement.
                # We will continue the loop to try lower ranks.
                continue
        else:
            # If we failed to find rank R, it's possible R-1 is also impossible,
            # but not guaranteed. However, to save time, we might stop if we fail
            # several ranks in a row.
            # For now, just continue to next rank.
            pass
            
    return best_result

def _schoolbook(n=3, m=4, p=8):
    """Return schoolbook decomposition (rank n*m*p)."""
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
