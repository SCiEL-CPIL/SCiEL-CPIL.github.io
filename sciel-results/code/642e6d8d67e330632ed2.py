"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS
N, M, P = n, m, p  # aliases

# EVOLVE-BLOCK-START

def _schoolbook():
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

def _check_decomposition(U, V, W, tol=1e-6):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The target tensor has T[i,j,k] = 1 for all i,j,k
    # We compute the actual tensor and compare to all-ones
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Let's compute efficiently
    # For each c, we have a rank-1 tensor: u_c (n*m), v_c (m*p), w_c (p*n)
    # T[i,j,k] += u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    
    # Build target tensor
    target = np.ones((n, m, p))
    
    # Compute actual tensor
    actual = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # uc is indexed by i*m+j
        # vc is indexed by j*p+k
        # wc is indexed by k*n+i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    
    diff = np.abs(actual - target)
    return np.max(diff) < tol, np.max(diff)

def _try_reduce_rank(schoolbook_result, max_rank_target, time_budget=30.0):
    """Try to find a lower-rank decomposition by zeroing out columns and checking."""
    U = [row[:] for row in schoolbook_result["U"]]
    V = [row[:] for row in schoolbook_result["V"]]
    W = [row[:] for row in schoolbook_result["W"]]
    rank = len(U[0])
    
    start_time = time.time()
    best_rank = rank
    best_result = None
    
    # Greedy: try to remove columns one by one
    # For each column, check if it's redundant
    for iteration in range(1000):
        if time.time() - start_time > time_budget:
            break
        if best_rank <= max_rank_target:
            break
        
        # Shuffle column order for diversity
        cols = list(range(best_rank))
        random.shuffle(cols)
        
        for c in cols:
            if time.time() - start_time > time_budget:
                break
            # Try removing column c
            U_test = [row[:c] + row[c+1:] for row in U]
            V_test = [row[:c] + row[c+1:] for row in V]
            W_test = [row[:c] + row[c+1:] for row in W]
            
            valid, _ = _check_decomposition(U_test, V_test, W_test)
            if valid:
                U = U_test
                V = V_test
                W = W_test
                best_rank -= 1
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result if best_result else schoolbook_result

def _random_search(start_rank, time_budget=60.0):
    """Random search for lower-rank decompositions."""
    start_time = time.time()
    best_rank = start_rank
    best_result = None
    
    for trial in range(200):
        if time.time() - start_time > time_budget:
            break
        if best_rank <= 15:  # We know rank 18 is possible, try to beat that
            break
        
        rank = random.randint(15, best_rank - 1)
        
        # Generate random matrices with small integer/rational entries
        U = np.random.randint(-2, 3, size=(n*m, rank))
        V = np.random.randint(-2, 3, size=(m*p, rank))
        W = np.random.randint(-2, 3, size=(p*n, rank))
        
        valid, _ = _check_decomposition(U.tolist(), V.tolist(), W.tolist())
        if valid:
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = n * m * p
    best_result = result
    
    # Try greedy column removal
    if time.time() < deadline - 60:
        reduced = _try_reduce_rank(result, max_rank_target=15, time_budget=40.0)
        if reduced:
            rank = len(reduced["U"][0])
            if rank < best_rank:
                best_rank = rank
                best_result = reduced
    
    # Try random search if we have time
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        random_res = _random_search(best_rank, time_budget=min(remaining, 60.0))
        if random_res:
            rank = len(random_res["U"][0])
            if rank < best_rank:
                best_rank = rank
                best_result = random_res
    
    # Final safety: return best found
    return best_result

# EVOLVE-BLOCK-END
