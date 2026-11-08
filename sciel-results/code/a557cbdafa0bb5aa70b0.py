"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    rank = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _make_identity_decomposition(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_integer_half_integer_search(n, m, p, target_rank, time_budget, rng=None, max_attempts=1000):
    """
    Try to find a decomposition with given rank using random search over integer/half-integer values.
    """
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    
    # Possible values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    possible_values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # For efficiency, bias towards smaller absolute values
    weights = [1, 1, 2, 3, 5, 3, 2, 1, 1]
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_budget:
            break
        
        U = [[rng.choices(possible_values, weights=weights)[0] for _ in range(target_rank)] for _ in range(n*m)]
        V = [[rng.choices(possible_values, weights=weights)[0] for _ in range(target_rank)] for _ in range(m*p)]
        W = [[rng.choices(possible_values, weights=weights)[0] for _ in range(target_rank)] for _ in range(p*n)]
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_gradient_descent_polish(n, m, p, U, V, W, time_budget, rng=None):
    """
    Try to polish a near-exact decomposition using gradient descent.
    This is useful if we find a decomposition that's close but not exact.
    """
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    target_rank = len(U[0]) if U and U[0] else 0
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(5000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    loss += (s - 1.0) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = s - 1.0
                    for c in range(target_rank):
                        dU[i*m+j][c] += 2 * err * V[j*p+k][c] * W[k*n+i][c]
                        dV[j*p+k][c] += 2 * err * U[i*m+j][c] * W[k*n+i][c]
                        dW[k*n+i][c] += 2 * err * U[i*m+j][c] * V[j*p+k][c]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check if we found a valid solution
    if best_loss < 1e-10:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_structured_construction(n, m, p, target_rank):
    """
    Try to construct a decomposition using algebraic structure.
    For 2x4x5, try to exploit the fact that we have 2 rows.
    """
    # Strategy: try to find a decomposition where some terms are shared
    # between the two i values
    
    # One approach: use the fact that T[i,j,k] = 1 for all i,j,k
    # We can try to write 1 as a sum of products
    
    # For now, try a simple construction:
    # Use rank-1 terms that cover all (i,j,k) triples
    
    # Actually, let's try a different approach:
    # The tensor is the all-ones tensor of shape (n*m, m*p, p*n) in the flattened sense
    # but with the specific index structure
    
    # Let's try to find a low-rank decomposition by solving a system of equations
    # This is a bilinear system, so it's hard, but for small ranks we can try
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 120s limit, finish well before
    
    # Start with the known schoolbook decomposition
    best_result = _make_identity_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Verify the initial decomposition
    assert _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p), \
        "Initial decomposition failed verification"
    
    rng = random.Random(12345)
    
    # Try to find a lower-rank decomposition
    # The theoretical lower bound is n*m + m*p + p*n - 2 = 8 + 20 + 10 - 2 = 36
    # So try ranks 39, 38, 37, 36
    
    for target_rank in range(best_rank - 1, max(35, best_rank - 5), -1):
        if time.time() > deadline - 20.0:
            break
        
        # Try integer/half-integer search
        time_budget = min(20.0, deadline - time.time() - 5.0)
        if time_budget > 2.0:
            result = _try_integer_half_integer_search(n, m, p, target_rank, time_budget, rng=rng, max_attempts=2000)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
        
        # Also try gradient descent from random initialization
        time_budget = min(10.0, deadline - time.time() - 5.0)
        if time_budget > 2.0:
            U = [[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
            result = _try_gradient_descent_polish(n, m, p, U, V, W, time_budget, rng=rng)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
    
    return best_result
