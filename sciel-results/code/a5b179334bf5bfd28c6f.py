"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _build_target_tensor(n, m, p):
    """Build the full target tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k), else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _validate_full(U, V, W, n, m, p, tol=1e-6):
    """Full validation: check ALL entries of the tensor decomposition.
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] must match the target tensor exactly.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T = _build_target_tensor(n, m, p)
    for c in range(p * n):
        M_c = (U * W[c, :]) @ V.T
        if np.max(np.abs(M_c - T[:, :, c])) > tol:
            return False
    return True

def _validate_diagonal_only(U, V, W, n, m, p, tol=1e-6):
    """Quick check: only the 45 diagonal entries."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _gradient_descent(n, m, p, rank, n_restarts=5, max_iter=500, lr=0.01, seed=0, deadline=None):
    """
    Gradient descent on the full tensor loss:
    f(U,V,W) = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
    where T is the target tensor (1 on diagonal, 0 elsewhere).
    """
    rng = np.random.RandomState(seed)
    T = _build_target_tensor(n, m, p)
    best_loss = float('inf')
    best_result = None
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        scale = 0.1
        U = rng.randn(n * m, rank) * scale
        V = rng.randn(m * p, rank) * scale
        W = rng.randn(p * n, rank) * scale
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            loss = 0.0
            
            for c in range(p * n):
                M_c = (U * W[c, :]) @ V.T
                err_c = M_c - T[:, :, c]
                loss += np.sum(err_c * err_c)
                
                err_c_at_V = err_c @ V
                dU += 2.0 * (err_c_at_V * W[c, :])
                
                err_cT_at_U = err_c.T @ U
                dV += 2.0 * (err_cT_at_U * W[c, :])
                
                EV = err_c @ V
                dW[c, :] += 2.0 * np.sum(U * EV, axis=0)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss < 1e-12:
                break
        
        if loss < best_loss:
            best_loss = loss
            best_result = (U.copy(), V.copy(), W.copy())
        
        if best_loss < 1e-10:
            break
    
    if best_result is not None and best_loss < 1e-6:
        return best_result, best_loss
    return None, best_loss

def _polish(U, V, W, n, m, p, max_iter=200, lr=0.001, deadline=None):
    """High-precision polish pass with smaller learning rate."""
    T = _build_target_tensor(n, m, p)
    loss = float('inf')
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        loss = 0.0
        for c in range(p * n):
            M_c = (U * W[c, :]) @ V.T
            err_c = M_c - T[:, :, c]
            loss += np.sum(err_c * err_c)
            err_c_at_V = err_c @ V
            dU += 2.0 * (err_c_at_V * W[c, :])
            err_cT_at_U = err_c.T @ U
            dV += 2.0 * (err_cT_at_U * W[c, :])
            EV = err_c @ V
            dW[c, :] += 2.0 * np.sum(U * EV, axis=0)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        if loss < 1e-14:
            break
    return U, V, W, loss

def _try_integer_search(n, m, p, rank, deadline, seed=42):
    """
    Search for a rank-r decomposition with integer/half-integer entries.
    Uses a structured random walk over small integer values.
    Returns (U, V, W) as numpy arrays if found, else None.
    """
    rng = np.random.RandomState(seed)
    values = np.array([-2, -1, -0.5, 0, 0, 0, 0.5, 1, 2], dtype=np.float64)
    
    def compute_loss(U, V, W):
        T = _build_target_tensor(n, m, p)
        loss = 0.0
        for c in range(p * n):
            M_c = (U * W[c, :]) @ V.T
            err_c = M_c - T[:, :, c]
            loss += np.sum(err_c * err_c)
        return loss
    
    best_loss = float('inf')
    best_result = None
    
    for restart in range(10):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        U = np.random.choice(values, size=(n * m, rank))
        V = np.random.choice(values, size=(m * p, rank))
        W = np.random.choice(values, size=(p * n, rank))
        
        loss = compute_loss(U, V, W)
        if loss < best_loss:
            best_loss = loss
            best_result = (U.copy(), V.copy(), W.copy())
            if best_loss < 1e-10:
                break
        
        # Random walk refinement
        for step in range(500):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            mat = rng.randint(0, 2)
            if mat == 0:
                row = rng.randint(0, n * m - 1)
                col = rng.randint(0, rank - 1)
                old = U[row, col]
                new_val = values[rng.randint(0, len(values) - 1)]
                U[row, col] = new_val
            elif mat == 1:
                row = rng.randint(0, m * p - 1)
                col = rng.randint(0, rank - 1)
                old = V[row, col]
                new_val = values[rng.randint(0, len(values) - 1)]
                V[row, col] = new_val
            else:
                row = rng.randint(0, p * n - 1)
                col = rng.randint(0, rank - 1)
                old = W[row, col]
                new_val = values[rng.randint(0, len(values) - 1)]
                W[row, col] = new_val
            
            new_loss = compute_loss(U, V, W)
            if new_loss < loss:
                loss = new_loss
                if loss < best_loss:
                    best_loss = loss
                    best_result = (U.copy(), V.copy(), W.copy())
                    if best_loss < 1e-10:
                        return best_result
            else:
                # Revert
                if mat == 0:
                    U[row, col] = old
                elif mat == 1:
                    V[row, col] = old
                else:
                    W[row, col] = old
    
    if best_result is not None and best_loss < 1e-6:
        return best_result
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    min_possible = max(n * m, m * p, p * n)
    
    # Try ranks from lower to higher
    for rank in range(min_possible, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # First try integer/half-integer search (more likely to find exact solutions)
        result_int = _try_integer_search(n, m, p, rank, deadline, seed=rank * 1000 + 42)
        if result_int is not None:
            U, V, W = result_int
            if _validate_full(U, V, W, n, m, p, tol=1e-5):
                best_rank = rank
                best_result = (U, V, W)
                found = True
        
        if not found:
            # Try gradient descent with multiple restarts
            result, loss = _gradient_descent(
                n, m, p, rank,
                n_restarts=3, max_iter=300, lr=0.01, seed=rank,
                deadline=deadline
            )
            if result is not None:
                U, V, W = result
                if _validate_full(U, V, W, n, m, p, tol=1e-5):
                    # Polish
                    U, V, W, loss2 = _polish(U, V, W, n, m, p, max_iter=100, lr=0.001, deadline=deadline)
                    if _validate_full(U, V, W, n, m, p, tol=1e-4):
                        best_rank = rank
                        best_result = (U, V, W)
                        found = True
        
        if found:
            continue
    
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
