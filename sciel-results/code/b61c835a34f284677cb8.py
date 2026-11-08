"""Rank-minimal tensor decomposition for 4x5 * 5x6 matrix multiplication.

Combines block-structured Kronecker initialization with alternating linear
solves and greedy column dropping to find a low-rank exact decomposition.
"""
import time
import random

try:
    import numpy as np
except Exception:
    np = None

DIMENSIONS = [4, 5, 6]
SEED = 42

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition of rank n*m*p."""
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, R


def _verify_numpy(U, V, W, n, m, p):
    """Verify the decomposition using numpy."""
    if np is None:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            target = np.eye(m, dtype=np.float64)
            if not np.allclose(M, target, atol=1e-9):
                return False
    return True


def _verify_exact(U, V, W, n, m, p):
    """Verify using exact integer arithmetic."""
    R = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    s = 0
                    for c in range(R):
                        wv = W[k * n + i][c]
                        uu = U[i * m + j][c]
                        vv = V[j2 * p + k][c]
                        if wv == 0 or uu == 0 or vv == 0:
                            continue
                        s += wv * uu * vv
                    target = 1 if j == j2 else 0
                    if s != target:
                        return False
    return True


def _random_init(n, m, p, rank, seed=42):
    """Initialize U, V, W with random values of given rank."""
    rng = random.Random(seed)
    U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.randint(-1, 1) for _ in range(rank)] for _ in range(p * n)]
    return U, V, W


def _try_rank(n, m, p, target_rank, deadline, max_attempts=50):
    """Try to find a valid decomposition of the given target rank.
    
    Uses alternating linear solves: fix two of U, V, W and solve for the third.
    """
    if np is None:
        return None
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 20.0:
            break
        
        # Random initialization
        U, V, W = _random_init(n, m, p, target_rank, seed=SEED + attempt * 1000)
        
        # Alternating minimization
        for iteration in range(30):
            if time.time() > deadline - 20.0:
                break
            
            Ua = np.array(U, dtype=np.float64)
            Va = np.array(V, dtype=np.float64)
            
            # Solve for W given U, V
            Wa_new = np.zeros((p * n, target_rank), dtype=np.float64)
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m * m, target_rank), dtype=np.float64)
                    b = np.zeros(m * m, dtype=np.float64)
                    for j in range(m):
                        for j2 in range(m):
                            idx = j * m + j2
                            A[idx, :] = Ua[i*m+j, :] * Va[j*p+k, :]
                            b[idx] = 1.0 if j == j2 else 0.0
                    
                    try:
                        W_row = np.linalg.lstsq(A, b, rcond=None)[0]
                        Wa_new[k*n+i, :] = W_row
                    except Exception:
                        pass
            
            W = Wa_new.tolist()
            
            if _verify_numpy(U, V, W, n, m, p):
                return U, V, W
            
            # Solve for U given V, W
            Va = np.array(V, dtype=np.float64)
            Wa = np.array(W, dtype=np.float64)
            Ua_new = np.zeros((n * m, target_rank), dtype=np.float64)
            
            for i in range(n):
                for j in range(m):
                    A = np.zeros((m * p, target_rank), dtype=np.float64)
                    b = np.zeros(m * p, dtype=np.float64)
                    for k in range(p):
                        for j2 in range(m):
                            idx = j2 * p + k
                            A[idx, :] = Wa[k*n+i, :] * Va[j2*p+k, :]
                            b[idx] = 1.0 if j == j2 else 0.0
                    
                    try:
                        U_row = np.linalg.lstsq(A, b, rcond=None)[0]
                        Ua_new[i*m+j, :] = U_row
                    except Exception:
                        pass
            
            U = Ua_new.tolist()
            
            if _verify_numpy(U, V, W, n, m, p):
                return U, V, W
            
            # Solve for V given U, W
            Ua = np.array(U, dtype=np.float64)
            Wa = np.array(W, dtype=np.float64)
            Va_new = np.zeros((m * p, target_rank), dtype=np.float64)
            
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n * m, target_rank), dtype=np.float64)
                    b = np.zeros(n * m, dtype=np.float64)
                    for i in range(n):
                        for j2 in range(m):
                            idx = i * m + j2
                            A[idx, :] = Wa[k*n+i, :] * Ua[i*m+j2, :]
                            b[idx] = 1.0 if j == j2 else 0.0
                    
                    try:
                        V_row = np.linalg.lstsq(A, b, rcond=None)[0]
                        Va_new[j*p+k, :] = V_row
                    except Exception:
                        pass
            
            V = Va_new.tolist()
            
            if _verify_numpy(U, V, W, n, m, p):
                return U, V, W
        
    return None


def _greedy_column_drop(U, V, W, n, m, p, deadline):
    """Greedy algorithm to remove columns that are not needed."""
    if np is None:
        return U, V, W
    
    R = len(U[0])
    removed = set()
    changed = True
    while changed and time.time() < deadline - 20.0:
        changed = False
        R = len(U[0])
        for c in range(R):
            if c in removed:
                continue
            if time.time() > deadline - 20.0:
                break
            keep = [x for x in range(R) if x != c and x not in removed]
            if len(keep) == 0:
                continue
            U2 = [[U[r][x] for x in keep] for r in range(n * m)]
            V2 = [[V[r][x] for x in keep] for r in range(m * p)]
            W2 = [[W[r][x] for x in keep] for r in range(p * n)]
            if _verify_numpy(U2, V2, W2, n, m, p):
                removed.add(c)
                U, V, W = U2, V2, W2
                changed = True
                break
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    
    try:
        # Start with schoolbook
        U, V, W, R = _build_schoolbook(n, m, p)
        
        best_U, best_V, best_W = U, V, W
        best_rank = R
        
        # Try various target ranks
        for target_rank in [best_rank - 1, best_rank - 2, best_rank - 5, best_rank - 10, best_rank - 20]:
            if target_rank < n * m:
                continue
            if time.time() > deadline - 30.0:
                break
            
            result = _try_rank(n, m, p, target_rank, deadline, max_attempts=20)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = target_rank
        
        # Greedy column drop
        best_U, best_V, best_W = _greedy_column_drop(best_U, best_V, best_W, n, m, p, deadline)
        
        return {"U": best_U, "V": best_V, "W": best_W}
    
    except Exception:
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

# EVOLVE-BLOCK-END
