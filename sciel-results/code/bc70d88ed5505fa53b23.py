"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_exceeded():
    return _elapsed() > TIME_LIMIT

def _make_tensor(n, m, p):
    """Create the matrix multiplication tensor as a dense array of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _rank1_tensor(u, v, w):
    """Create a rank-1 tensor from vectors u, v, w."""
    return np.einsum('a,b,c->abc', u, v, w)

def _greedy_set_cover(T, candidates, max_iter=1000):
    """
    Greedy set cover: find a set of rank-1 tensors whose sum equals T.
    T: target tensor (n*m, m*p, p*n)
    candidates: list of (u, v, w) tuples
    Returns: list of (u, v, w) that sum to T, or None if not found.
    """
    remaining = T.copy().astype(np.float64)
    selected = []
    for _ in range(max_iter):
        if _time_exceeded():
            break
        # Find candidate that best matches remaining
        best_score = -1
        best_idx = -1
        for idx, (u, v, w) in enumerate(candidates):
            if _time_exceeded():
                break
            R = _rank1_tensor(u, v, w)
            # Score: dot product with remaining
            score = np.sum(R * remaining)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0 or best_score <= 0:
            break
        u, v, w = candidates[best_idx]
        R = _rank1_tensor(u, v, w)
        remaining -= R
        selected.append((u, v, w))
        # Check if done
        if np.allclose(remaining, 0, atol=1e-9):
            break
        # Remove or keep candidate? For set cover, we remove it to avoid reuse.
        # But for linear combinations, we might need to reuse. Let's keep it for now.
    if not np.allclose(remaining, 0, atol=1e-9):
        return None
    return selected

def _build_strassen_candidates(n, m, p):
    """
    Build a pool of candidate rank-1 tensors including Strassen-style combinations.
    Returns list of (u, v, w) numpy arrays.
    """
    candidates = []
    
    # Standard schoolbook candidates
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                candidates.append((u, v, w))
    
    # Strassen-style: for pairs of indices, create linear combinations
    # For the middle dimension j, pair up indices and use Strassen-like formulas
    # This is heuristic; we'll add combinations that might cancel
    
    # Add some Strassen-style candidates for 2x2 sub-blocks
    # For each pair of i's, j's, k's, create Strassen-like terms
    for i1 in range(n):
        for i2 in range(n):
            if i1 >= i2:
                continue
            for j1 in range(m):
                for j2 in range(m):
                    if j1 >= j2:
                        continue
                    for k1 in range(p):
                        for k2 in range(p):
                            if k1 >= k2:
                                continue
                            # Strassen-style for 2x2x2 sub-problem
                            # M1 = (A[i1,j1]+A[i2,j2])(B[j1,k1]+B[j2,k2])
                            u = np.zeros(n * m)
                            v = np.zeros(m * p)
                            w = np.zeros(p * n)
                            u[i1 * m + j1] = 1
                            u[i2 * m + j2] = 1
                            v[j1 * p + k1] = 1
                            v[j2 * p + k2] = 1
                            # W needs to reconstruct both C[i1,k1] and C[i2,k2]
                            # This is tricky; for now just add as candidate
                            w[k1 * n + i1] = 1
                            w[k2 * n + i2] = 1
                            candidates.append((u, v, w))
    
    return candidates

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 60
    
    # Build target tensor
    T = _make_tensor(n, m, p)
    
    # Try greedy set cover with Strassen-style candidates
    candidates = _build_strassen_candidates(n, m, p)
    
    selected = _greedy_set_cover(T, candidates, max_iter=200)
    
    if selected is not None:
        # Convert to list format
        r = len(selected)
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        for c, (u, v, w) in enumerate(selected):
            for idx in range(n * m):
                if u[idx] != 0:
                    U[idx][c] = int(u[idx]) if abs(u[idx] - round(u[idx])) < 1e-9 else float(u[idx])
            for idx in range(m * p):
                if v[idx] != 0:
                    V[idx][c] = int(v[idx]) if abs(v[idx] - round(v[idx])) < 1e-9 else float(v[idx])
            for idx in range(p * n):
                if w[idx] != 0:
                    W[idx][c] = int(w[idx]) if abs(w[idx] - round(w[idx])) < 1e-9 else float(w[idx])
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
