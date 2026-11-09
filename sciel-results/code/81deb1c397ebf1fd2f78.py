"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time as _time
import random as _random

def construct():
    _start = _time.time()
    _deadline = _start + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    _sb = _schoolbook(n, m, p)
    _best = {"U": _sb[0], "V": _sb[1], "W": _sb[2]}
    _best_rank = n * m * p
    
    # Try to find lower-rank decompositions using known constructions and search
    # Known lower bound: R(5,5,5) >= 49 (from Coppersmith-Winograd type bounds, but actual LB is lower)
    # Try ranks from a reasonable lower bound upward
    for _rank in range(49, min(_best_rank, 75)):
        if _time.time() > _deadline:
            break
        _res = _search_rank(n, m, p, _rank, _deadline)
        if _res is not None:
            _best = _res
            _best_rank = _rank
            break  # Found smallest rank in this range
    
    return _best

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _search_rank(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition. Returns dict or None."""
    # Use a structured search: exploit that the tensor is 1 for all (i,j,k)
    # Try random restarts with small integer entries
    for _attempt in range(200):
        if _time.time() > deadline:
            return None
        _res = _try_random(n, m, p, rank)
        if _res is not None:
            return _res
    return None

def _try_random(n, m, p, rank):
    """Try a random initialization and check if it works."""
    # Initialize with small integers
    U = [[_random.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
    V = [[_random.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
    W = [[_random.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
    
    # Check constraint: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return None
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
