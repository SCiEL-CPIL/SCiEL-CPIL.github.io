import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
        
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    # Use einsum for efficiency
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _try_structured_decomposition(n, m, p, rank, time_limit=5.0, seed=42):
    """
    Try to find a valid decomposition using structured patterns.
    Uses gradient-based optimization with random restarts.
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    
    # Try multiple random restarts with gradient descent
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random values
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Gradient descent
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Compute loss: sum of (T[i,j,k] - 1)^2
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            loss = np.sum((T - 1.0) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if loss < 1e-10:
                break
            
            # Compute gradients
            # dT/dU_3d[i,j,c] = sum_k V_3d[j,k,c] * W_3d[k,i,c]
            # dT/dV_3d[j,k,c] = sum_i U_3d[i,j,c] * W_3d[k,i,c]
            # dT/dW_3d[k,i,c] = sum_j U_3d[i,j,c] * V_3d[j,k,c]
            
            dU_3d = np.einsum('jkc,kic->ijc', V_3d, W_3d)
            dV_3d = np.einsum('ijc,kic->jkc', U_3d, W_3d)
            dW_3d = np.einsum('ijc,jkc->kic', U_3d, V_3d)
            
            # Gradient of loss with respect to T is 2*(T-1)
            grad_T = 2.0 * (T - 1.0)
            
            # Chain rule
            grad_U_3d = np.einsum('ijk,jkc,kic->ijc', grad_T, V_3d, W_3d)
            grad_V_3d = np.einsum('ijk,ijc,kic->jkc', grad_T, U_3d, W_3d)
            grad_W_3d = np.einsum('ijk,ijc,jkc->kic', grad_T, U_3d, V_3d)
            
            # Update
            U -= lr * grad_U_3d.reshape(nm, rank)
            V -= lr * grad_V_3d.reshape(mp, rank)
            W -= lr * grad_W_3d.reshape(pn, rank)
            
            # Adaptive learning rate
            if loss < best_loss * 0.9:
                lr *= 0.95
            elif loss > best_loss * 1.1:
                lr *= 0.8
                
        # Check if we found a valid solution
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def _try_sparse_structured(n, m, p, rank, time_limit=5.0, seed=42):
    """
    Try sparse structured decompositions with specific patterns.
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    
    # Try various structured patterns
    patterns = [
        'diagonal', 'block', 'cyclic', 'random_sparse'
    ]
    
    for pattern in patterns:
        if time.time() - start_time > time_limit:
            break
            
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            if pattern == 'diagonal':
                # Each rank dimension handles a subset of (i,j,k)
                for c in range(rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
            elif pattern == 'block':
                # Block structure
                for c in range(rank):
                    i = c % n
                    block_j = (c // n) % m
                    block_k = (c // (n * m)) % p
                    for j in range(block_j, block_j + 1):
                        for k in range(block_k, block_k + 1):
                            U[i * m + j, c] += 1.0 / (1.0)
                            V[j * p + k, c] += 1.0 / (1.0)
                            W[k * n + i, c] += 1.0 / (1.0)
            elif pattern == 'cyclic':
                # Cyclic patterns
                for c in range(rank):
                    i = c % n
                    j = (c + n) % m
                    k = (c + n + m) % p
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
            elif pattern == 'random_sparse':
                # Random sparse initialization
                for c in range(rank):
                    i = rng.randrange(nm)
                    j = rng.randrange(mp)
                    k = rng.randrange(pn)
                    U[i, c] = rng.choice([0.5, 1.0, -1.0])
                    V[j, c] = rng.choice([0.5, 1.0, -1.0])
                    W[k, c] = rng.choice([0.5, 1.0, -1.0])
                    
            # Verify
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # First, get the schoolbook baseline
    baseline = _schoolbook(n, m, p)
    best_result = baseline
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Start from rank 49 down to 30
    for rank in range(49, 29, -1):
        if time.time() > deadline:
            break
        
        # Try structured approach
        result = _try_structured_decomposition(n, m, p, rank, time_limit=3.0, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
            
        # Try sparse structured approach
        result = _try_sparse_structured(n, m, p, rank, time_limit=2.0, seed=rank + 100)
        if result is not None:
            best_result = result
            best_rank = rank
            break
            
    return best_result
