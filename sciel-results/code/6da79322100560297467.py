"""Matrix multiplication 2x3x8 rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
DEADLINE_SECONDS = 115  # safety margin before 120s limit


def _validate(U, V, W, n, m, p):
    """Check exact decomposition using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    u_val = U[ui][c]
                    v_val = V[vj][c]
                    w_val = W[wk][c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        total += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                if total != 1:
                    return False
    return True


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
    return U, V, W


def _try_remove_column(U, V, W, n, m, p, rng):
    """Try to remove a random column. Returns (success, new_U, new_V, new_W)."""
    rank = len(U[0]) if U else 0
    if rank <= 1:
        return (False, U, V, W)
    
    c = rng.randint(0, rank - 1)
    new_rank = rank - 1
    new_U = [[U[i][j] for j in range(rank) if j != c] for i in range(n * m)]
    new_V = [[V[i][j] for j in range(rank) if j != c] for i in range(m * p)]
    new_W = [[W[i][j] for j in range(rank) if j != c] for i in range(p * n)]
    
    if _validate(new_U, new_V, new_W, n, m, p):
        return (True, new_U, new_V, new_W)
    return (False, U, V, W)


def _try_scale_column(U, V, W, n, m, p, rng):
    """Try scaling a column by a small factor and adjusting another."""
    rank = len(U[0]) if U else 0
    if rank < 2:
        return (False, U, V, W)
    
    c1 = rng.randint(0, rank - 1)
    c2 = rng.randint(0, rank - 1)
    if c1 == c2:
        return (False, U, V, W)
    
    # Try scaling c1 by factor s and see if we can compensate
    for s_val in [Fraction(1, 2), Fraction(2), Fraction(3, 2), Fraction(2, 3)]:
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        
        for i in range(n * m):
            new_U[i][c1] = U[i][c1] * s_val
        for i in range(m * p):
            new_V[i][c1] = V[i][c1] * s_val
        for i in range(p * n):
            new_W[i][c1] = W[i][c1] * s_val
        
        if _validate(new_U, new_V, new_W, n, m, p):
            return (True, new_U, new_V, new_W)
    
    return (False, U, V, W)


def _local_search(U, V, W, n, m, p, rng, max_iters=200):
    """Local search to reduce rank."""
    current_U, current_V, current_W = U, V, W
    current_rank = len(current_U[0]) if current_U else 0
    
    for _ in range(max_iters):
        # Try removing a column
        success, nU, nV, nW = _try_remove_column(current_U, current_V, current_W, n, m, p, rng)
        if success:
            current_U, current_V, current_W = nU, nV, nW
            current_rank = len(current_U[0]) if current_U else 0
            continue
        
        # Try scaling
        success, nU, nV, nW = _try_scale_column(current_U, current_V, current_W, n, m, p, rng)
        if success:
            current_U, current_V, current_W = nU, nV, nW
    
    return current_U, current_V, current_W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    rng = random.Random(42)
    
    # Iterative improvement with restarts
    while time.time() - start_time < DEADLINE_SECONDS:
        U, V, W = _schoolbook(n, m, p)
        
        # Local search
        U, V, W = _local_search(U, V, W, n, m, p, rng, max_iters=100)
        current_rank = len(U[0]) if U else 0
        
        if current_rank < best_rank:
            best_rank = current_rank
            best_result = {"U": U, "V": V, "W": W}
        
        # Change seed for next restart
        rng = random.Random(rng.randint(0, 2**31))
    
    return best_result
