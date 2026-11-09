"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_fast(U, V, W, n, m, p):
    """Verify decomposition using vectorized numpy. U,V,W are numpy arrays."""
    nmp = n * m * p
    # Precompute index arrays (do this once outside for efficiency, but here for clarity)
    i_idx = np.array([i for i in range(n) for j in range(m) for k in range(p)], dtype=np.int64)
    j_idx = np.array([j for i in range(n) for j in range(m) for k in range(p)], dtype=np.int64)
    k_idx = np.array([k for i in range(n) for j in range(m) for k in range(p)], dtype=np.int64)
    
    u_rows = (i_idx * m + j_idx).astype(np.int64)
    v_rows = (j_idx * p + k_idx).astype(np.int64)
    w_rows = (k_idx * n + i_idx).astype(np.int64)
    
    rank = U.shape[1]
    # For each c: U[u_rows, c] * V[v_rows, c] * W[w_rows, c], then sum over c
    # Use broadcasting: create (nmp, rank) arrays
    U_sub = U[u_rows, :]  # (nmp, rank)
    V_sub = V[v_rows, :]  # (nmp, rank)
    W_sub = W[w_rows, :]  # (nmp, rank)
    
    total = np.sum(U_sub * V_sub * W_sub, axis=1)  # (nmp,)
    
    return bool(np.all(np.abs(total - 1.0) < 1e-9))

def _construct_schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_random_search(rank, deadline, seed_offset=0):
    """Try to find a decomposition of given rank using random search with fast verification."""
    n, m, p = DIMENSIONS
    rows_u = n * m
    rows_v = m * p
    rows_w = p * n
    
    random.seed(42 + seed_offset)
    np.random.seed(42 + seed_offset)
    
    attempts = 0
    max_attempts = 20000
    density = 0.2
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        # Sparse integer entries
        U = np.zeros((rows_u, rank), dtype=np.float64)
        V = np.zeros((rows_v, rank), dtype=np.float64)
        W = np.zeros((rows_w, rank), dtype=np.float64)
        
        for c in range(rank):
            # Each column has a few non-zero entries
            nnz_u = random.randint(1, 3)
            nnz_v = random.randint(1, 3)
            nnz_w = random.randint(1, 3)
            
            u_rows = random.sample(range(rows_u), nnz_u)
            v_rows = random.sample(range(rows_v), nnz_v)
            w_rows = random.sample(range(rows_w), nnz_w)
            
            for r in u_rows:
                U[r, c] = random.choice([-2, -1, 1, 2])
            for r in v_rows:
                V[r, c] = random.choice([-2, -1, 1, 2])
            for r in w_rows:
                W[r, c] = random.choice([-2, -1, 1, 2])
        
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    best = _construct_schoolbook()
    best_rank = schoolbook_rank
    
    for rank in range(78, schoolbook_rank):
        if time.time() > deadline:
            break
        result = _try_random_search(rank, deadline, seed_offset=rank)
        if result is not None:
            best = result
            best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
