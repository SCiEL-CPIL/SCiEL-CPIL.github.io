<<<<<<< SEARCH
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    # Reconstruct tensor
    # T[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
    # We only need to check the non-zero entries and a sample of zero entries?
    # No, must be exact.
    # For efficiency, we can check the 36 non-zero entries.
    # If those are 1, and we assume the structure is correct, we should also check that no other entries are non-zero.
    # But checking all entries is (9*12*12) = 1296 entries. That's fast.
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    rank = U_np.shape[1]
    
    # Compute tensor
    # T = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
    # This creates a 9x12x12 array.
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = np.sum(U_np[u, :] * V_np[v, :] * W_np[w, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
                
    # Check a few zero entries to ensure no leakage
    # If the rank is low, leakage is likely.
    # We'll check all entries for safety if rank < 36
    if rank < 36:
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    # Skip non-zero entries
                    found = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if u == i * m + j and v == j * p + k and w == k * n + i:
                                    found = True
                    if not found:
                        val = np.sum(U_np[u, :] * V_np[v, :] * W_np[w, :])
                        if not np.isclose(val, 0.0, atol=1e-9):
                            return False
    return True

def _try_find_decomposition(n, m, p, target_rank, time_limit):
    """Try to find a decomposition of target_rank using ALS."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute non-zero indices
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_indices.append((i * m + j, j * p + k, k * n + i))
                
    for attempt in range(10):
        if time.time() > time_limit:
            break
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # ALS iterations
        for it in range(100):
            if time.time() > time_limit:
                break
            # Update U
            # U[u, c] = sum_{v,w} T[u,v,w] V[v,c] W[w,c] / sum_{v,w} V[v,c]^2 W[w,c]^2
            # This is expensive.
            # Instead, use a simple gradient step?
            # Given the small size, we can do a full check.
            
            # Check for integer/half-integer solution
            # Round U, V, W to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
            # Simple update: move towards satisfying equations
            # This is a heuristic and may not work well.
            # I will skip complex optimization and just return None if not found quickly.
            
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a better decomposition
    # Start from rank 27 up to 35
    for r in range(27, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        result = _try_find_decomposition(n, m, p, r, deadline - 5)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 4]

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using vectorized operations."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Compute all tensor entries at once
    # T[u, v, w] = sum_c U[u,c] * V[v,c] * W[w,c]
    T = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if not np.isclose(T[u, v, w], 1.0, atol=1e-9):
                    return False
    
    # Check zero entries
    nm, mp, pn = n * m, m * p, p * n
    nz_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_set.add((i * m + j, j * p + k, k * n + i))
    
    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                if (u, v, w) not in nz_set:
                    if not np.isclose(T[u, v, w], 0.0, atol=1e-9):
                        return False
    return True

def _random_half_integer_matrix(rows, cols, max_val=3):
    """Generate a random matrix with half-integer entries in [-max_val, max_val]."""
    # Values: -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3
    values = np.arange(-max_val, max_val + 0.5, 0.5)
    return values[np.random.randint(0, len(values), size=(rows, cols))]

def _try_find_decomposition_structured(n, m, p, target_rank, time_limit):
    """Try to find a decomposition using structured random search over half-integers."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    best_result = None
    
    while time.time() < time_limit:
        # Generate random half-integer matrices
        U = _random_half_integer_matrix(nm, target_rank, max_val=2)
        V = _random_half_integer_matrix(mp, target_rank, max_val=2)
        W = _random_half_integer_matrix(pn, target_rank, max_val=2)
        
        # Quick check
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
        
        # Check time
        if time.time() > time_limit - 1:
            break
            
    return None

def _try_gradient_descent(n, m, p, target_rank, time_limit):
    """Try gradient descent to find a solution, then round to half-integers."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    for attempt in range(5):
        if time.time() > time_limit - 5:
            break
            
        U = np.random.randn(nm, target_rank) * 0.5
        V = np.random.randn(mp, target_rank) * 0.5
        W = np.random.randn(pn, target_rank) * 0.5
        
        # Gradient descent
        lr = 0.01
        for it in range(200):
            if time.time() > time_limit - 5:
                break
                
            # Compute gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        # T[u,v,w] should be 1
                        val = np.sum(U[u, :] * V[v, :] * W[w, :])
                        err = val - 1.0
                        grad_U[u, :] += err * V[v, :] * W[w, :]
                        grad_V[v, :] += err * U[u, :] * W[w, :]
                        grad_W[w, :] += err * U[u, :] * V[v, :]
                        
            # Also penalize non-zero entries
            nz_set = set()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        nz_set.add((i * m + j, j * p + k, k * n + i))
                        
            for u in range(nm):
                for v in range(mp):
                    for w in range(pn):
                        if (u, v, w) not in nz_set:
                            val = np.sum(U[u, :] * V[v, :] * W[w, :])
                            grad_U[u, :] += val * V[v, :] * W[w, :]
                            grad_V[v, :] += val * U[u, :] * W[w, :]
                            grad_W[w, :] += val * U[u, :] * V[v, :]
                            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check if we're close to a half-integer solution
            if it % 20 == 0:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try structured search for ranks 27 to 35
    for r in range(27, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Try structured random search
        result = _try_find_decomposition_structured(n, m, p, r, min(deadline - 15, time.time() + 10))
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
        # Try gradient descent
        result = _try_gradient_descent(n, m, p, r, min(deadline - 10, time.time() + 10))
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
