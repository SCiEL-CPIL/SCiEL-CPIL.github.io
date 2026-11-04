import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
        
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check tensor decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a 3-tensor contraction.
    # We can reshape and use einsum or matrix multiplication.
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # Let's just loop in Python for small n,m,p but use floats for speed? 
    # Or use einsum.
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Compute T[i, j, k]
    # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    # This is not a standard matmul. 
    # We can compute it by looping over i, j, k which is 2*5*5=50 iterations.
    # Each iteration is a dot product of length rank.
    # 50 * rank operations. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Dot product of slices
                val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                if abs(val - 1.0) > 1e-9:
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

def _try_rank_local_search(n, m, p, rank, time_limit=2.0, seed=42):
    """
    Attempt to find a valid decomposition with given rank using a fast, simplified search.
    Given the complexity, this is a best-effort attempt. If it fails, we return None.
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    values = [0, 0.5, -0.5, 1, -1]
    
    start_time = time.time()
    
    # Try a few random restarts with very few steps
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
            
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        # Initialize with sparse random values
        for _ in range(rank * 2):
            U[rng.randrange(nm)][rng.randrange(rank)] = rng.choice(values)
            V[rng.randrange(mp)][rng.randrange(rank)] = rng.choice(values)
            W[rng.randrange(pn)][rng.randrange(rank)] = rng.choice(values)
        
        # Very limited local search
        for step in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Fast verification
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
            # Random mutation
            var_idx = rng.randrange(3)
            c = rng.randrange(rank)
            if var_idx == 0:
                idx = rng.randrange(nm)
                U[idx][c] = rng.choice(values)
            elif var_idx == 1:
                idx = rng.randrange(mp)
                V[idx][c] = rng.choice(values)
            else:
                idx = rng.randrange(pn)
                W[idx][c] = rng.choice(values)
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 10  # Strict deadline to ensure we return well before 120s
    
    # First, get the schoolbook baseline
    baseline = _schoolbook(n, m, p)
    best_result = baseline
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # We try ranks from 49 down to 40 quickly.
    for rank in range(49, 39, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank_local_search(n, m, p, rank, time_limit=1.0, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
            
    return best_result
