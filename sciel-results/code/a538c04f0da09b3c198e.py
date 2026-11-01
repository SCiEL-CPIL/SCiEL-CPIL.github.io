import time
import random
import sys

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    This is the guaranteed valid solution.
    """
    n, m, p = DIMENSIONS
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

def _search_rank(target_rank, deadline, seed=42, value_set=None):
    """
    Search for a decomposition of the given rank using random sparse initialization.
    Returns None if not found before deadline.
    """
    if value_set is None:
        # Small set of integers/half-integers to keep exact arithmetic feasible
        value_set = [-1, 1, -0.5, 0.5, -2, 2]
    
    n, m, p = 3, 3, 3
    rank = target_rank
    random.seed(seed)
    
    # Limit attempts to avoid wasting too much time
    max_attempts = 10000
    start_time = time.time()
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random sparse matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Each row has 2-4 non-zero entries
        # Using a slightly higher density might help connectivity
        for i in range(n * m):
            num_nz = random.randint(2, 4)
            idxs = random.sample(range(rank), num_nz)
            for idx in idxs:
                U[i][idx] = random.choice(value_set)
                
        for i in range(m * p):
            num_nz = random.randint(2, 4)
            idxs = random.sample(range(rank), num_nz)
            for idx in idxs:
                V[i][idx] = random.choice(value_set)
                
        for i in range(p * n):
            num_nz = random.randint(2, 4)
            idxs = random.sample(range(rank), num_nz)
            for idx in idxs:
                W[i][idx] = random.choice(value_set)
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    # Total time budget is 120s. 
    # We allocate a small portion for search, and keep the rest as safety margin.
    # Given the low probability of finding rank < 27, we keep search time short.
    overall_deadline = time.time() + 110  # 10s safety margin
    
    # Try to find a lower rank decomposition (e.g., 26, 25, 24)
    # We try rank 26 first as it's the most likely candidate if one exists.
    search_time_budget = 5.0  # 5 seconds for search
    
    # Try rank 26
    if time.time() < overall_deadline - search_time_budget:
        r26 = _search_rank(26, time.time() + search_time_budget, seed=101)
        if r26 is not None:
            return r26
        
    # Try rank 25
    if time.time() < overall_deadline - search_time_budget:
        r25 = _search_rank(25, time.time() + search_time_budget, seed=102)
        if r25 is not None:
            return r25
            
    # Try rank 24
    if time.time() < overall_deadline - search_time_budget:
        r24 = _search_rank(24, time.time() + search_time_budget, seed=103)
        if r24 is not None:
            return r24

    # Fallback: use schoolbook rank 27
    # This is the guaranteed valid solution.
    return _construct_schoolbook()
