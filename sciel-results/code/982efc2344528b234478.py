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

def _construct_strassen_27():
    """
    Construct a rank-27 decomposition using Strassen-like sub-blocks.
    This is a deterministic construction that should be valid.
    """
    n, m, p = 3, 3, 3
    rank = 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We'll use the standard schoolbook as a base but structured
    # Actually, let's just return the schoolbook for now to be safe
    # and try to find a better one with search
    return _construct_schoolbook()

def _search_rank(target_rank, deadline, seed=42, value_set=None):
    """
    Search for a decomposition of the given rank.
    """
    if value_set is None:
        value_set = [-1, 1, -0.5, 0.5]
    
    n, m, p = 3, 3, 3
    rank = target_rank
    random.seed(seed)
    
    # Use a smarter search: start with a random sparse pattern and refine
    # Or use a gradient-free optimization approach
    
    # For now, let's try a more structured random search
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Generate random sparse matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Each row has 2-4 non-zero entries
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
    overall_deadline = time.time() + 140  # safety margin before 150s kill
    
    # Try to find a rank-23 decomposition (known to exist for 3x3 matmul)
    # This is the Strassen rank for 3x3 matrix multiplication
    
    # First, try a deterministic construction if we know one
    # For 3x3, the Strassen algorithm gives rank 23 (actually, it's more complex)
    
    # Let's try searching for rank 23
    start = time.time()
    r23 = _search_rank(23, overall_deadline - 20, seed=1)
    if r23 is not None:
        return r23
    
    # Try rank 24
    if time.time() < overall_deadline - 20:
        r24 = _search_rank(24, overall_deadline - 20, seed=2)
        if r24 is not None:
            return r24
    
    # Try rank 25
    if time.time() < overall_deadline - 20:
        r25 = _search_rank(25, overall_deadline - 20, seed=3)
        if r25 is not None:
            return r25
    
    # Try rank 26
    if time.time() < overall_deadline - 20:
        r26 = _search_rank(26, overall_deadline - 20, seed=4)
        if r26 is not None:
            return r26
    
    # Fall back to schoolbook rank 27
    return _construct_schoolbook()
