import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def _check_validity(U, V, W, n, m, p, rank):
    """
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    # Check: sum_c W[k*n+i, c] U[i'*m+j, c] V[j'*p+k', c] = 1 if i'=i, j=j', k'=k else 0
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            expected = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                            actual = 0.0
                            for c in range(rank):
                                actual += W[k*n+i, c] * U[i2*m+j, c] * V[j2*p+k2, c]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _compute_error(U, V, W, n, m, p, rank):
    err = 0.0
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            expected = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                            actual = 0.0
                            for c in range(rank):
                                actual += W[k*n+i, c] * U[i2*m+j, c] * V[j2*p+k2, c]
                            err += (actual - expected) ** 2
    return err

def _round_to_half_int(x):
    return round(x * 2) / 2.0

def _try_rank(r, n, m, p, time_limit):
    start_time = time.time()
    U = np.random.uniform(-1, 1, (n*m, r))
    V = np.random.uniform(-1, 1, (m*p, r))
    W = np.random.uniform(-1, 1, (p*n, r))
    
    # Simple gradient descent
    lr = 0.01
    best_err = float('inf')
    
    while time.time() - start_time < time_limit:
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # This is O(n^2 m^2 p^2 r) which is 2^2 * 3^2 * 5^2 * r = 4*9*25*r = 900r
        # For r=20, 18000 operations per gradient step. Very fast.
        
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                expected = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                                actual = 0.0
                                for c in range(r):
                                    term = W[k*n+i, c] * U[i2*m+j, c] * V[j2*p+k2, c]
                                    actual += term
                                    # d(actual)/d(W) = U * V
                                    # d(actual)/d(U) = W * V
                                    # d(actual)/d(V) = W * U
                                    grad_W[k*n+i, c] += 2 * (actual - expected) * U[i2*m+j, c] * V[j2*p+k2, c]
                                    grad_U[i2*m+j, c] += 2 * (actual - expected) * W[k*n+i, c] * V[j2*p+k2, c]
                                    grad_V[j2*p+k2, c] += 2 * (actual - expected) * W[k*n+i, c] * U[i2*m+j, c]
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check if close to solution and try rounding
        err = _compute_error(U, V, W, n, m, p, r)
        if err < best_err:
            best_err = err
        
        if err < 1e-4:
            # Try rounding
            U_r = np.vectorize(_round_to_half_int)(U)
            V_r = np.vectorize(_round_to_half_int)(V)
            W_r = np.vectorize(_round_to_half_int)(W)
            if _check_validity(U_r, V_r, W_r, n, m, p, r):
                return U_r, V_r, W_r
        elif err < 1e-2:
            # Occasionally try rounding even if not super close
            if random.random() < 0.1:
                U_r = np.vectorize(_round_to_half_int)(U)
                V_r = np.vectorize(_round_to_half_int)(V)
                W_r = np.vectorize(_round_to_half_int)(W)
                if _check_validity(U_r, V_r, W_r, n, m, p, r):
                    return U_r, V_r, W_r
                
    return None

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook solution
    rank_school = n * m * p
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W}
    
    deadline = time.time() + 110 # Leave 10s margin
    
    # Try ranks from 16 up to 29
    for r in range(16, rank_school):
        if time.time() > deadline:
            break
        res = _try_rank(r, n, m, p, time_limit=5.0)
        if res is not None:
            U_r, V_r, W_r = res
            # Convert to lists
            return {
                'U': U_r.tolist(),
                'V': V_r.tolist(),
                'W': W_r.tolist()
            }
            
    return {
        'U': best['U'].tolist(),
        'V': best['V'].tolist(),
        'W': best['W'].tolist()
    }
