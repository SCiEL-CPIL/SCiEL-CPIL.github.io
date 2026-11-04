"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 18
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    for target_rank in [14, 15, 16, 17]:
        if time.time() > deadline - 15:
            break
        result = _find_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    return best_result

def _schoolbook(n, m, p):
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

def _find_decomposition(n, m, p, target_rank, deadline):
    """Find a decomposition of given rank using structured random search."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Number of constraints: n*m*p
    num_constraints = n * m * p
    
    max_attempts = 500
    attempt = 0
    
    while attempt < max_attempts and time.time() < deadline - 10:
        attempt += 1
        
        # Generate random decomposition with small integer/half-integer entries
        U = _rand_mat(n_m, target_rank)
        V = _rand_mat(m_p, target_rank)
        W = _rand_mat(p_n, target_rank)
        
        # Check validity
        if _check(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Periodically do local search
        if attempt % 5 == 0:
            U, V, W = _local_opt(U, V, W, n, m, p, deadline)
            if _check(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # Try a structured approach: fix U to be mostly 0/1 pattern
        if attempt % 20 == 0:
            U, V, W = _structured_init(n, m, p, target_rank)
            if _check(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            U, V, W = _local_opt(U, V, W, n, m, p, deadline)
            if _check(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _rand_mat(rows, cols):
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    return [[random.choice(vals) for _ in range(cols)] for _ in range(rows)]

def _structured_init(n, m, p, rank):
    """Initialize with a sparse structure that might be closer to a valid decomposition."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Start from schoolbook and randomly zero out some columns, then adjust
    U = [[0] * rank for _ in range(n_m)]
    V = [[0] * rank for _ in range(m_p)]
    W = [[0] * rank for _ in range(p_n)]
    
    # Use a subset of schoolbook patterns plus random fill
    schoolbook_patterns = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_patterns.append((i, j, k))
    
    # Randomly select 'rank' patterns, with some overlap
    selected = random.sample(schoolbook_patterns, min(rank, len(schoolbook_patterns)))
    # If rank > num patterns, add random ones
    while len(selected) < rank:
        i = random.randrange(n)
        j = random.randrange(m)
        k = random.randrange(p)
        selected.append((i, j, k))
    
    for col, (i, j, k) in enumerate(selected):
        U[i * m + j][col] = random.choice([1, -1, 0.5, -0.5])
        V[j * p + k][col] = random.choice([1, -1, 0.5, -0.5])
        W[k * n + i][col] = random.choice([1, -1, 0.5, -0.5])
    
    return U, V, W

def _check(U, V, W, n, m, p):
    """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _local_opt(U, V, W, n, m, p, deadline):
    """Local optimization: perturb entries to reduce error."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    def total_error(U, V, W):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                    err += (val - 1.0) ** 2
        return err
    
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    current_err = total_error(U, V, W)
    
    for step in range(200):
        if time.time() > deadline - 10:
            break
        
        # Pick a random matrix and entry
        mat_idx = random.choice([0, 1, 2])
        if mat_idx == 0:
            mat = U; rows = len(U); cols = rank
        elif mat_idx == 1:
            mat = V; rows = len(V); cols = rank
        else:
            mat = W; rows = len(W); cols = rank
        
        r = random.randrange(rows)
        c = random.randrange(cols)
        old = mat[r][c]
        new = random.choice(vals)
        if new == old:
            continue
        mat[r][c] = new
        
        new_err = total_error(U, V, W)
        if new_err < current_err:
            current_err = new_err
        else:
            mat[r][c] = old
    
    return U, V, W
# EVOLVE-BLOCK-END
