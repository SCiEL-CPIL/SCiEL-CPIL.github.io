"""Search-based program for Matrix multiplication 2,5,5.
Attempts to find a rank-minimal exact tensor decomposition via random search
with fallback to schoolbook rank-50.
"""
import random
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# Deadline guard: 150s wall-clock limit, keep 10s safety margin
_DEADLINE_START = None
_TIME_BUDGET = 130.0  # seconds


def _elapsed():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return time.time() - _DEADLINE_START


def _within_budget():
    return _elapsed() < _TIME_BUDGET


def _verify_fast(U, V, W, n, m, p, A_vec, B_vec):
    """Fast verification using precomputed vectors."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    A_mat = A_vec.reshape(n, m)
    B_mat = B_vec.reshape(m, p)
    C_true = (A_mat @ B_mat).flatten()
    
    C_comp = np.zeros(p * n)
    for c in range(rank):
        vA = float(U_np[:, c] @ A_vec)
        vB = float(V_np[:, c] @ B_vec)
        if vA != 0 and vB != 0:
            C_comp += W_np[:, c] * vA * vB
    
    return np.allclose(C_comp, C_true, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """Verify with multiple random inputs."""
    rng = random.Random(12345)
    for trial in range(3):
        A = np.array([rng.uniform(-5, 5) for _ in range(n * m)])
        B = np.array([rng.uniform(-5, 5) for _ in range(m * p)])
        if not _verify_fast(U, V, W, n, m, p, A, B):
            return False
    return True


def _build_schoolbook():
    """Return the schoolbook rank-50 decomposition."""
    n, m, p = DIMENSIONS
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


def _structured_search(target_rank, time_limit, seed=42):
    """Structured random search for integer/half-integer decompositions.
    Uses the structure of the problem: exploit that n=2 is small.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    start_time = time.time()
    
    # Precompute test vectors
    A1 = np.array([(i + 1) * 10 + (j + 1) for i in range(n) for j in range(m)], dtype=float)
    B1 = np.array([(j + 1) * 10 + (k + 1) for j in range(m) for k in range(p)], dtype=float)
    A2 = np.array([((i + 1) * 7 + (j + 1) * 3) for i in range(n) for j in range(m)], dtype=float)
    B2 = np.array([((j + 1) * 11 + (k + 1)) for j in range(m) for k in range(p)], dtype=float)
    
    A_mat1 = A1.reshape(n, m)
    B_mat1 = B1.reshape(m, p)
    C_true1 = (A_mat1 @ B_mat1).flatten()
    
    A_mat2 = A2.reshape(n, m)
    B_mat2 = B2.reshape(m, p)
    C_true2 = (A_mat2 @ B_mat2).flatten()
    
    max_iters = 5000
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if not _within_budget():
            break
        
        # Generate random sparse matrices
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        for c in range(target_rank):
            # Each factor: pick 1-3 nonzero entries in each matrix
            u_count = rng.randint(1, 3)
            v_count = rng.randint(1, 3)
            w_count = rng.randint(1, 3)
            
            u_idxs = rng.sample(range(n * m), u_count)
            v_idxs = rng.sample(range(m * p), v_count)
            w_idxs = rng.sample(range(p * n), w_count)
            
            for idx in u_idxs:
                U[idx][c] = rng.choice([-1, 1])
            for idx in v_idxs:
                V[idx][c] = rng.choice([-1, 1])
            for idx in w_idxs:
                W[idx][c] = rng.choice([-1, 1])
        
        # Quick check with first test vector
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        C_comp = np.zeros(p * n)
        for c in range(target_rank):
            vA = float(U_np[:, c] @ A1)
            vB = float(V_np[:, c] @ B1)
            if vA != 0 and vB != 0:
                C_comp += W_np[:, c] * vA * vB
        
        if not np.allclose(C_comp, C_true1, atol=1e-9):
            continue
        
        # Check with second test vector
        C_comp2 = np.zeros(p * n)
        for c in range(target_rank):
            vA = float(U_np[:, c] @ A2)
            vB = float(V_np[:, c] @ B2)
            if vA != 0 and vB != 0:
                C_comp2 += W_np[:, c] * vA * vB
        
        if not np.allclose(C_comp2, C_true2, atol=1e-9):
            continue
        
        # Full verification
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None


def _dense_random_search(target_rank, time_limit, seed=99):
    """Denser random search with small integer entries."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    start_time = time.time()
    
    A1 = np.array([(i + 1) * 10 + (j + 1) for i in range(n) for j in range(m)], dtype=float)
    B1 = np.array([(j + 1) * 10 + (k + 1) for j in range(m) for k in range(p)], dtype=float)
    A_mat1 = A1.reshape(n, m)
    B_mat1 = B1.reshape(m, p)
    C_true1 = (A_mat1 @ B_mat1).flatten()
    
    max_iters = 3000
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if not _within_budget():
            break
        
        # Dense random with small entries
        U = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        C_comp = np.zeros(p * n)
        for c in range(target_rank):
            vA = float(U_np[:, c] @ A1)
            vB = float(V_np[:, c] @ B1)
            if vA != 0 and vB != 0:
                C_comp += W_np[:, c] * vA * vB
        
        if not np.allclose(C_comp, C_true1, atol=1e-9):
            continue
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix mult.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    _DEADLINE_START = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    best_result = _build_schoolbook()
    best_rank = schoolbook_rank
    
    # Try ranks from 49 down to 30, allocating time based on remaining budget
    # Lower ranks are harder, so give more time to higher ranks first
    ranks_to_try = list(range(schoolbook_rank - 1, max(29, schoolbook_rank - 20), -1))
    
    total_time = _TIME_BUDGET
    start = time.time()
    
    for target_rank in ranks_to_try:
        if not _within_budget():
            break
        
        elapsed = time.time() - start
        remaining = _TIME_BUDGET - elapsed - 5  # 5s safety margin
        if remaining < 2:
            break
        
        # Allocate time: more time for higher ranks (easier to find if possible)
        time_alloc = min(remaining * 0.4, 10.0)
        
        # Try structured search first
        result = _structured_search(target_rank, time_alloc, seed=target_rank * 7 + 1)
        if result is not None:
            best_result = result
            best_rank = target_rank
            continue
        
        # Try dense random search
        if _within_budget():
            result = _dense_random_search(target_rank, time_alloc, seed=target_rank * 13 + 5)
            if result is not None:
                best_result = result
                best_rank = target_rank
                continue
    
    return best_result
