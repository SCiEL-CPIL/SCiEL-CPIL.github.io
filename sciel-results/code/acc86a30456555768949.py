import time
import numpy as np
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using numpy."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, np.ones((n, m, p)), atol=tol)

def _random_search(target_rank, n, m, p, max_iters=5000, deadline=None, seed=None):
    """Random search for a valid decomposition with the given rank."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Generate random matrices with small integer entries
        entries = [-1, 0, 1]
        U = np.array([[random.choice(entries) for _ in range(target_rank)] for _ in range(n*m)], dtype=np.int64)
        V = np.array([[random.choice(entries) for _ in range(target_rank)] for _ in range(m*p)], dtype=np.int64)
        W = np.array([[random.choice(entries) for _ in range(target_rank)] for _ in range(p*n)], dtype=np.int64)
        
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W, True
    
    return None, None, None, False

def _local_search(U, V, W, n, m, p, max_iters=1000, deadline=None):
    """Local search: try flipping individual entries to reduce rank or fix issues."""
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try flipping a random entry in U, V, or W
        choice = random.random()
        if choice < 0.33:
            idx = random.randint(0, n*m-1)
            col = random.randint(0, rank-1)
            best_U[idx, col] *= -1
            if _verify_fast(best_U, best_V, best_W, n, m, p):
                U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
            else:
                best_U[idx, col] *= -1
        elif choice < 0.66:
            idx = random.randint(0, m*p-1)
            col = random.randint(0, rank-1)
            best_V[idx, col] *= -1
            if _verify_fast(best_V, best_V, best_W, n, m, p):
                V = best_V.copy()
            else:
                best_V[idx, col] *= -1
        else:
            idx = random.randint(0, p*n-1)
            col = random.randint(0, rank-1)
            best_W[idx, col] *= -1
            if _verify_fast(best_U, best_V, best_W, n, m, p):
                W = best_W.copy()
            else:
                best_W[idx, col] *= -1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), rank_sb
    
    # Try random search for lower ranks
    for target_rank in range(44, 30, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            U, V, W, success = _random_search(target_rank, n, m, p, max_iters=2000, deadline=deadline - 30, seed=attempt)
            if success:
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
