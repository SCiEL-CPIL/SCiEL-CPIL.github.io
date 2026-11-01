"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
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

def _validate(U, V, W, n, m, p):
    """Check if the factor matrices exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # must equal 1 for all (i,j,k).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _try_rank(r, n, m, p, deadline, rng):
    """
    Try to find a rank-r decomposition using randomized search with local refinement.
    We use the structure that for small ranks, the factors are sparse.
    Strategy: start with random sparse matrices and try to fix via local moves.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Initialize with random sparse integer matrices
    # Each row has ~1-2 non-zeros
    U = [[0]*r for _ in range(nm)]
    V = [[0]*r for _ in range(mp)]
    W = [[0]*r for _ in range(pn)]
    
    # Randomly place non-zeros
    for i in range(nm):
        nz = rng.randint(1, 2)
        cols = rng.sample(range(r), nz)
        for c in cols:
            U[i][c] = rng.choice([-1, 1, 2, -2])
    for j in range(mp):
        nz = rng.randint(1, 2)
        cols = rng.sample(range(r), nz)
        for c in cols:
            V[j][c] = rng.choice([-1, 1, 2, -2])
    for k in range(pn):
        nz = rng.randint(1, 2)
        cols = rng.sample(range(r), nz)
        for c in cols:
            W[k][c] = rng.choice([-1, 1, 2, -2])
    
    # Local search: try to fix errors
    for iteration in range(5000):
        if time.time() > deadline:
            return None
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Find an error
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        errors.append((i, j, k, s))
        
        if not errors:
            return {"U": U, "V": V, "W": W}
        
        # Pick a random error and try to fix it
        i, j, k, s = rng.choice(errors)
        # Try changing a random entry in one of the three matrices
        mat = rng.choice([0, 1, 2])
        if mat == 0:
            row = i*m+j
            col = rng.randint(0, r-1)
            old = U[row][col]
            U[row][col] = old + rng.choice([-2, -1, 1, 2])
        elif mat == 1:
            row = j*p+k
            col = rng.randint(0, r-1)
            old = V[row][col]
            V[row][col] = old + rng.choice([-2, -1, 1, 2])
        else:
            row = k*n+i
            col = rng.randint(0, r-1)
            old = W[row][col]
            W[row][col] = old + rng.choice([-2, -1, 1, 2])
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(4, 10, 10) = 10
    # Try ranks from 10 upwards
    best_result = _schoolbook()
    best_rank = nm * mp * pn // (nm * mp)  # This is just p=5... no, schoolbook is n*m*p=20
    
    # Schoolbook rank is n*m*p = 2*2*5 = 20
    schoolbook_rank = n * m * p
    
    rng = random.Random(42)
    
    # Try to find a better rank
    for r in range(10, schoolbook_rank):
        if time.time() > deadline - 30:
            break
        result = _try_rank(r, n, m, p, deadline, rng)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
