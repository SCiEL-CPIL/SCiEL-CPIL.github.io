"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

DEADLINE_MARGIN = 10.0  # seconds to spare

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact (within tolerance)."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                prod = V_np[b] * W_np[c]
                s = np.dot(U_np[a], prod)
                if abs(s - 1.0) > tol:
                    return False
    
    # Check a sample of invalid triples
    rng = np.random.RandomState(42)
    for _ in range(200):
        a = rng.randint(0, n * m)
        b = rng.randint(0, m * p)
        c = rng.randint(0, p * n)
        i, j = divmod(a, m)
        jp, k = divmod(b, p)
        kn, i2 = divmod(c, n)
        if j == jp and k == kn and i == i2:
            continue
        prod = V_np[b] * W_np[c]
        s = np.dot(U_np[a], prod)
        if abs(s) > tol:
            return False
    return True

def _construct_schoolbook(n, m, p):
    """Standard schoolbook rank n*m*p."""
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

def _als_solve(n, m, p, R, deadline, max_iters=200, tol=1e-10):
    """
    Alternating Least Squares for tensor decomposition.
    Returns (U, V, W) as numpy arrays or None if failed.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build constraint list: (a, b, c, target)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                constraints.append((a, b, c, 1.0))
    
    rng = np.random.RandomState(42)
    num_invalid = 500
    for _ in range(num_invalid):
        a = rng.randint(0, nm)
        b = rng.randint(0, mp)
        c = rng.randint(0, pn)
        i, j = divmod(a, m)
        jp, k = divmod(b, p)
        kn, i2 = divmod(c, n)
        if j == jp and k == kn and i == i2:
            continue
        constraints.append((a, b, c, 0.0))
    
    n_constraints = len(constraints)
    
    # Pre-index constraints by each variable
    cons_by_a = [[] for _ in range(nm)]
    cons_by_b = [[] for _ in range(mp)]
    cons_by_c = [[] for _ in range(pn)]
    for idx, (a, b, c, tgt) in enumerate(constraints):
        cons_by_a[a].append((idx, b, c, tgt))
        cons_by_b[b].append((idx, a, c, tgt))
        cons_by_c[c].append((idx, a, b, tgt))
    
    # Initialize
    U = rng.randn(nm, R) * 0.5
    V = rng.randn(mp, R) * 0.5
    W = rng.randn(pn, R) * 0.5
    
    best_obj = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Solve for U given V, W
        U_new = np.zeros((nm, R))
        for a in range(nm):
            if not cons_by_a[a]:
                continue
            rows = []
            targets = []
            for (idx, b, c, tgt) in cons_by_a[a]:
                rows.append(V[b] * W[c])
                targets.append(tgt)
            A = np.array(rows)
            b_vec = np.array(targets)
            try:
                U_new[a] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
            except:
                U_new[a] = 0
        
        # Solve for V given U, W
        V_new = np.zeros((mp, R))
        for b in range(mp):
            if not cons_by_b[b]:
                continue
            rows = []
            targets = []
            for (idx, a, c, tgt) in cons_by_b[b]:
                rows.append(U[a] * W[c])
                targets.append(tgt)
            A = np.array(rows)
            b_vec = np.array(targets)
            try:
                V_new[b] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
            except:
                V_new[b] = 0
        
        # Solve for W given U, V
        W_new = np.zeros((pn, R))
        for c in range(pn):
            if not cons_by_c[c]:
                continue
            rows = []
            targets = []
            for (idx, a, b, tgt) in cons_by_c[c]:
                rows.append(U[a] * V[b])
                targets.append(tgt)
            A = np.array(rows)
            b_vec = np.array(targets)
            try:
                W_new[c] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
            except:
                W_new[c] = 0
        
        U, V, W = U_new, V_new, W_new
        
        # Compute objective
        obj = 0.0
        for (a, b, c, tgt) in constraints:
            val = np.dot(U[a], V[b] * W[c])
            obj += (val - tgt) ** 2
        
        if obj < best_obj:
            best_obj = obj
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if obj < tol:
            break
    
    if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p  # 84
    
    min_rank = max(n * m, m * p, p * n)  # 28
    
    for target_rank in range(min_rank, min(best_rank, 50)):
        if time.time() > deadline - 5.0:
            break
        
        found = False
        for restart in range(2):
            if time.time() > deadline - 5.0:
                break
            
            result = _als_solve(n, m, p, target_rank, deadline)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
