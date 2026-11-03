"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a dict of non-zero entries."""
    # T[(i,j), (j,k), (k,i)] = 1
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=0):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    T = _build_target_tensor(n, m, p)
    rank = len(U[0]) if U else 0
    # Check all non-zero entries
    for (a, b, c) in T:
        s = 0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if abs(s - 1) > tol:
            return False
    # Check all zero entries
    all_a = range(n * m)
    all_b = range(m * p)
    all_c = range(p * n)
    for a in all_a:
        for b in all_b:
            for c in all_c:
                if (a, b, c) not in T:
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > tol:
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
    return {"U": U, "V": V, "W": W}

def _try_random_rank(r, n, m, p, max_iters=200, deadline=None):
    """Try to find a rank-r decomposition with random initialization."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Random initialization with small integer values
        U = [[random.randint(-2, 2) for _ in range(r)] for _ in range(nm)]
        V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(mp)]
        W = [[random.randint(-2, 2) for _ in range(r)] for _ in range(pn)]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    return None

def _local_search(U, V, W, n, m, p, r, max_iters=500, deadline=None):
    """Local search: randomly perturb one entry and check if it improves validity."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    def residual():
        """Compute max absolute residual."""
        max_res = 0
        for (a, b, c) in T:
            s = 0
            for rr in range(r):
                s += U[a][rr] * V[b][rr] * W[c][rr]
            max_res = max(max_res, abs(s - 1))
        # Sample some zero entries
        for _ in range(20):
            a = random.randint(0, nm - 1)
            b = random.randint(0, mp - 1)
            c = random.randint(0, pn - 1)
            if (a, b, c) not in T:
                s = 0
                for rr in range(r):
                    s += U[a][rr] * V[b][rr] * W[c][rr]
                max_res = max(max_res, abs(s))
        return max_res
    
    best_res = residual()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Pick random matrix, row, column
        mat = random.choice(['U', 'V', 'W'])
        if mat == 'U':
            row = random.randint(0, nm - 1)
            arr = U
        elif mat == 'V':
            row = random.randint(0, mp - 1)
            arr = V
        else:
            row = random.randint(0, pn - 1)
            arr = W
        col = random.randint(0, r - 1)
        old_val = arr[row][col]
        new_val = old_val + random.choice([-2, -1, 1, 2])
        arr[row][col] = new_val
        
        new_res = residual()
        if new_res <= best_res:
            best_res = new_res
            if best_res == 0:
                return True
        else:
            arr[row][col] = old_val
    return False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank solutions
    # Theoretical lower bound: at least the number of non-zero entries in any "slice"
    # For this tensor, we need at least max(nm, mp, pn) = 18? No, that's not right.
    # Actually, the rank must be at least the number of non-zero entries divided by max contribution per term.
    # Each rank-1 term can have at most one "active" (i,j,k) if we use 0/1 entries.
    # But with arbitrary values, one term can affect multiple entries.
    
    # Let's try ranks from a reasonable lower bound up
    # Lower bound: the tensor has 36 non-zero entries. If each rank-1 term covers at most 
    # one entry with value 1, we need at least 36. But with cancellations, we might do better.
    # Actually, for exact decomposition with the specific pattern, let's try from 20 upward.
    
    for r in range(20, best_rank):
        if time.time() > deadline:
            break
        sol = _try_random_rank(r, n, m, p, max_iters=50, deadline=deadline)
        if sol is not None:
            best = sol
            best_rank = r
            break
    
    # Try local search from schoolbook to see if we can reduce
    if time.time() < deadline:
        U = [row[:] for row in best["U"]]
        V = [row[:] for row in best["V"]]
        W = [row[:] for row in best["W"]]
        # This is expensive, skip for now
    
    return best
# EVOLVE-BLOCK-END
