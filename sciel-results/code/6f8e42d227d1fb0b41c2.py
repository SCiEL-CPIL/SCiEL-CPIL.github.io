"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _is_half_int(x):
    return abs(x * 2 - round(x * 2)) < 1e-9

def _round_to_half_int(x):
    return round(x * 2) / 2.0

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
    return {"U": U, "V": V, "W": W}

def _compute_full_error(U, V, W, n, m, p, rank):
    """Compute the full tensor error (sum of squared errors over all indices)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    total_err = 0.0
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                            total_err += (s - expected) ** 2
    return total_err

def _compute_gradients(U, V, W, n, m, p, rank):
    """
    Compute gradients of the full error with respect to U, V, W.
    Error = sum_{i,ip,j,jp,k,kp} (sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] - delta_{i,ip,j,jp,k,kp})^2
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                            err = s - expected
                            if abs(err) < 1e-12:
                                continue
                            
                            # d(error)/d(U[i*m+j, c]) = 2*err * V[jp*p+k,c] * W[kp*n+ip,c]
                            dU[i*m+j, :] += 2 * err * V[jp*p+k, :] * W[kp*n+ip, :]
                            dV[jp*p+k, :] += 2 * err * U[i*m+j, :] * W[kp*n+ip, :]
                            dW[kp*n+ip, :] += 2 * err * U[i*m+j, :] * V[jp*p+k, :]
    
    return dU, dV, dW

def _optimize_rank(n, m, p, rank, deadline, max_iter=300, lr_init=0.05):
    """
    Gradient descent optimization for a given rank.
    Returns (U, V, W) if converged, else None.
    """
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    best_result = None
    best_err = float('inf')
    
    for attempt in range(5):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(attempt * 987654 + 42)
        
        # Initialize: mix of random and structured
        if attempt == 0:
            # Start near schoolbook-like structure
            U = np.zeros((U_dim, rank))
            V = np.zeros((V_dim, rank))
            W = np.zeros((W_dim, rank))
            for c in range(min(rank, n*m*p)):
                i = c // (m*p)
                j = (c // p) % m
                k = c % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            # Add small random perturbation
            U += np.random.randn(U_dim, rank) * 0.1
            V += np.random.randn(V_dim, rank) * 0.1
            W += np.random.randn(W_dim, rank) * 0.1
        else:
            U = np.random.randn(U_dim, rank) * 0.3
            V = np.random.randn(V_dim, rank) * 0.3
            W = np.random.randn(W_dim, rank) * 0.3
        
        lr = lr_init
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            err = _compute_full_error(U, V, W, n, m, p, rank)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
            
            dU, dV, dW = _compute_gradients(U, V, W, n, m, p, rank)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm < 1e-12:
                break
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Reduce learning rate if error not improving
            new_err = _compute_full_error(U, V, W, n, m, p, rank)
            if new_err > err * 1.1:
                U += lr * dU
                V += lr * dV
                W += lr * dW
                lr *= 0.7
            elif new_err > err:
                U += lr * 0.5 * dU
                V += lr * 0.5 * dV
                W += lr * 0.5 * dW
                lr *= 0.85
            else:
                lr = min(lr * 1.05, 0.5)
    
    if best_result is not None and best_err < 1e-8:
        U, V, W = best_result
        # Round to half-integers
        U_r = np.array([[_round_to_half_int(x) for x in row] for row in U])
        V_r = np.array([[_round_to_half_int(x) for x in row] for row in V])
        W_r = np.array([[_round_to_half_int(x) for x in row] for row in W])
        
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, rank):
            return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    schoolbook_rank = n * m * p  # 36
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But actual lower bounds from tensor rank theory suggest higher
    # Try ranks from a reasonable lower bound up to schoolbook-1
    
    min_rank = max(n*m, m*p, p*n)  # 18
    # Also consider that for matrix multiplication, rank >= n*m*p / max(n,m,p) 
    # but that's not a tight bound. Let's just try from min_rank up.
    
    for rank in range(min_rank, schoolbook_rank):
        if time.time() > deadline - 20:
            break
        
        res = _optimize_rank(n, m, p, rank, deadline)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return result
# EVOLVE-BLOCK-END
