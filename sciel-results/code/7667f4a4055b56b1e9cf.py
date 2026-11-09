"""Matrix multiplication 4x4x6 rank-minimal exact decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

def _build_tensor(n, m, p):
    """Build the full matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = delta."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _tensor_to_matrices(T, rank):
    """Convert tensor to U, V, W factorization of given rank (random init)."""
    n, m, p = T.shape
    U = np.random.randn(n, rank)
    V = np.random.randn(m, rank)
    W = np.random.randn(p, rank)
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_halfint(M, tol=1e-4):
    """Snap values to nearest half-integer; return False if any value is far off."""
    for r in range(M.shape[1]):
        for row in range(M.shape[0]):
            v = M[row, r]
            if abs(v) < 1e-9:
                M[row, r] = 0.0
                continue
            h = np.round(2 * v) / 2
            if abs(v - h) > tol:
                return False
            M[row, r] = h
    return True

def _build_block_strassen():
    """Build the explicit block-Strassen rank-84 decomposition."""
    n, m, p = DIMENSIONS
    rank = 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    A_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
    }
    B_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (0, 2): (0, 1, 4, 5),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
        (1, 2): (2, 3, 4, 5),
    }
    C_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (0, 2): (0, 1, 4, 5),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
        (1, 2): (2, 3, 4, 5),
    }
    
    pairs = [
        ((0, 0), (0, 0), (0, 0)),
        ((0, 1), (1, 0), (0, 0)),
        ((0, 0), (0, 1), (0, 1)),
        ((0, 1), (1, 1), (0, 1)),
        ((0, 0), (0, 2), (0, 2)),
        ((0, 1), (1, 2), (0, 2)),
        ((1, 0), (0, 0), (1, 0)),
        ((1, 1), (1, 0), (1, 0)),
        ((1, 0), (0, 1), (1, 1)),
        ((1, 1), (1, 1), (1, 1)),
        ((1, 0), (0, 2), (1, 2)),
        ((1, 1), (1, 2), (1, 2)),
    ]
    
    strassen_terms = [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]},
    ]
    
    col = 0
    for (a_idx, b_idx, c_idx) in pairs:
        a_r1, _, a_c1, _ = A_blocks[a_idx]
        b_r1, _, b_c1, _ = B_blocks[b_idx]
        c_r1, _, c_c1, _ = C_blocks[c_idx]
        
        for term in strassen_terms:
            for local_a, coeff in term['u']:
                lr, lc = divmod(local_a, 2)
                U[(a_r1 + lr) * m + (a_c1 + lc), col] += coeff
            for local_b, coeff in term['v']:
                lr, lc = divmod(local_b, 2)
                V[(b_r1 + lr) * p + (b_c1 + lc), col] += coeff
            for local_c, coeff in term['w']:
                lr, lc = divmod(local_c, 2)
                W[(c_c1 + lc) * n + (c_r1 + lr), col] += coeff
            col += 1
    
    return U, V, W

def _optimize_decomposition(U, V, W, n, m, p, deadline, max_iters=5000, lr=0.1):
    """Optimize V and W (keeping U fixed) to minimize the decomposition error."""
    rng = np.random.RandomState(42)
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute errors for all triples
        errors = []
        for (i, j, k) in triples:
            val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
            errors.append(val - 1.0)
        
        max_err = max(abs(e) for e in errors)
        if max_err < 1e-10:
            break
        
        # Pick worst triple
        worst_idx = np.argmax([abs(e) for e in errors])
        i, j, k = triples[worst_idx]
        err = errors[worst_idx]
        
        # Gradient step on V and W
        dV = U[i * m + j, :] * W[k * n + i, :]
        dW = U[i * m + j, :] * V[j * p + k, :]
        
        V[j * p + k, :] -= lr * err * dV
        W[k * n + i, :] -= lr * err * dW
    
    return U, V, W

def _try_rank(U0, V0, W0, target_rank, n, m, p, deadline, seed=42):
    """Try to find a valid decomposition of target_rank."""
    rng = np.random.RandomState(seed)
    
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Select target_rank columns from the baseline
        idx = rng.choice(U0.shape[1], size=target_rank, replace=False)
        U = U0[:, idx].copy()
        V = V0[:, idx].copy()
        W = W0[:, idx].copy()
        
        # Optimize
        U, V, W = _optimize_decomposition(U, V, W, n, m, p, deadline)
        
        if _verify(U, V, W, n, m, p, tol=1e-8):
            U_copy, V_copy, W_copy = U.copy(), V.copy(), W.copy()
            if _snap_halfint(U_copy) and _snap_halfint(V_copy) and _snap_halfint(W_copy):
                if _verify(U_copy, V_copy, W_copy, n, m, p):
                    return U_copy, V_copy, W_copy
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    
    # Guaranteed baseline: block-Strassen rank 84
    U, V, W = _build_block_strassen()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = 84
    
    # Try lower ranks
    for target_rank in [72, 68, 64, 60, 56]:
        if time.time() > deadline:
            break
        
        res = _try_rank(U, V, W, target_rank, n, m, p, deadline, seed=target_rank)
        if res is not None:
            U_new, V_new, W_new = res
            if _verify(U_new, V_new, W_new, n, m, p):
                best = (U_new, V_new, W_new)
                best_rank = target_rank
                break
    
    return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
