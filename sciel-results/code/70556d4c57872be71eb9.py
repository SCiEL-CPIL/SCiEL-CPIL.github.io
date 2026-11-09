import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]
TIME_LIMIT = 140  # seconds, leave margin before 150s kill

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _to_fraction_list(arr):
    """Convert numpy array to list of Fractions with small denominators."""
    result = []
    for row in arr:
        row_frac = []
        for x in row:
            best = Fraction(int(round(x)), 1)
            for denom in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
                num = round(x * denom)
                if abs(x - num / denom) < 1e-9:
                    frac = Fraction(int(num), denom)
                    if abs(frac.numerator) < abs(best.numerator) or (
                        abs(frac.numerator) == abs(best.numerator) and denom < best.denominator
                    ):
                        best = frac
                    break
            row_frac.append(best)
        result.append(row_frac)
    return result

def _smooth_loss(U, V, W, n, m, p):
    """Compute smooth loss: sum of squared errors in tensor reconstruction."""
    rank = U.shape[1]
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                prod = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                val = np.sum(prod)
                loss += (val - 1.0) ** 2
    return loss

def _try_rank(rank, n, m, p, deadline, max_iters=5000, lr=0.01, seed=None):
    """Try to find a rank-r decomposition using gradient descent."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    prod = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                    val = np.sum(prod)
                    err = val - 1.0
                    
                    dU[u_idx, :] += 2 * err * V[v_idx, :] * W[w_idx, :]
                    dV[v_idx, :] += 2 * err * U[u_idx, :] * W[w_idx, :]
                    dW[w_idx, :] += 2 * err * U[u_idx, :] * V[v_idx, :]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        loss = _smooth_loss(U, V, W, n, m, p)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            break
    
    if best_loss < 1e-6:
        U_frac = _to_fraction_list(best_U)
        V_frac = _to_fraction_list(best_V)
        W_frac = _to_fraction_list(best_W)
        
        U_exact = np.array([[float(x) for x in row] for row in U_frac])
        V_exact = np.array([[float(x) for x in row] for row in V_frac])
        W_exact = np.array([[float(x) for x in row] for row in W_frac])
        
        if _verify_decomposition(U_exact, V_exact, W_exact, n, m, p):
            return {
                "U": U_frac,
                "V": V_frac,
                "W": W_frac
            }
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Schoolbook gives rank 54
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    for rank in range(18, min(best_rank, 40)):
        if time.time() > deadline - 5:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            seed = rank * 1000 + n * 100 + m * 10 + p + attempt * 137
            result = _try_rank(rank, n, m, p, deadline, max_iters=2000, lr=0.005 + 0.002 * attempt, seed=seed)
            if result is not None:
                best_result = result
                best_rank = rank
                break
    
    return best_result
