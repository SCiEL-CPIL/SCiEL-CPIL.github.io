"""Rewrite for 2x2 @ 2x16 multiplication; EVOLVE-BLOCK contains the search."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    if not np.allclose(T, target, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_block():
    n, m, p = 2, 2, 16
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for t in range(num_blocks):
        intermediates = [
            {'U': [(0, 1), (3, 1)], 'V': [(2*t, 1), (16+2*t+1, 1)], 'W': [(4*t, 1), (4*t+3, 1)]},
            {'U': [(2, 1), (3, 1)], 'V': [(2*t, 1)], 'W': [(4*t+1, 1), (4*t+3, -1)]},
            {'U': [(0, 1)], 'V': [(2*t+1, 1), (16+2*t+1, -1)], 'W': [(4*t+2, 1), (4*t+3, 1)]},
            {'U': [(3, 1)], 'V': [(16+2*t, 1), (2*t, -1)], 'W': [(4*t, 1), (4*t+1, 1)]},
            {'U': [(0, 1), (1, 1)], 'V': [(16+2*t+1, 1)], 'W': [(4*t, -1), (4*t+2, 1)]},
            {'U': [(2, 1), (0, -1)], 'V': [(2*t, 1), (2*t+1, 1)], 'W': [(4*t+3, 1)]},
            {'U': [(1, 1), (3, -1)], 'V': [(16+2*t, 1), (16+2*t+1, 1)], 'W': [(4*t, 1)]}
        ]
        for inter in intermediates:
            for idx, coeff in inter['U']:
                U[idx][col] += coeff
            for idx, coeff in inter['V']:
                V[idx][col] += coeff
            for idx, coeff in inter['W']:
                W[idx][col] += coeff
            col += 1
    return U, V, W

def _build_candidate(pairs):
    """pairs: list of (i,j,k,r) meaning contribution 1 to tensor index (i,j,k) from rank r."""
    n, m, p = 2, 2, 16
    if not pairs:
        return None, None, None
    max_r = max(r for _, _, _, r in pairs) + 1
    U = np.zeros((n*m, max_r))
    V = np.zeros((m*p, max_r))
    W = np.zeros((p*n, max_r))
    for (i, j, k, r) in pairs:
        U[i*m+j, r] += 1
        V[j*p+k, r] += 1
        W[k*n+i, r] += 1
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    T = np.einsum('ar,br,cr->abc', U, V, W)
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return bool(np.allclose(T, target, atol=1e-9))

def _construct_via_search(n, m, p, deadline):
    """Try to beat rank 56 by local search over small rank."""
    total = n * m * p
    base_rank = 56
    best_rank = base_rank
    best_U, best_V, best_W = _strassen_block()

    # Start from Strassen, attempt local search to reduce rank by merging/zeroing
    U, V, W = best_U, best_V, best_W

    # We'll attempt to drop rank by 1 by merging two columns that are near-linearly
    # dependent in the tensor space. Simple greedy: try all pairs of columns, check
    # if their tensor difference is small; if so, merge.
    def _tensor_diff(U1, V1, W1, U2, V2, W2):
        # Compute tensor of column r1 - r2 (scalar multiple 1, -1)
        T = np.einsum('ar,br,cr->abc',
                      np.stack([U1, -U2], axis=1),
                      np.stack([V1, -V2], axis=1),
                      np.stack([W1, -W2], axis=1))
        # Actually need sum over r of (U[:,r1]*V[:,r1]*W[:,r1] - U[:,r2]*V[:,r2]*W[:,r2])
        # Do it properly:
        T1 = np.einsum('ar,br,cr->abc', U1.reshape(4,1,1), V1.reshape(4,1,1), W1.reshape(4,1,1))
        T2 = np.einsum('ar,br,cr->abc', U2.reshape(4,1,1), V2.reshape(4,1,1), W2.reshape(4,1,1))
        return T1 - T2

    # For speed, build a lookup of tensor vectors for each rank-1 component
    # Each rank r contributes T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    nU, nV, nW = n*m, m*p, p*n
    # Precompute flattened tensors
    def _rank1_tensor(u, v, w):
        return np.outer(np.outer(u, v), w).flatten()

    # Build list of rank-1 tensors
    rank_tensors = []
    for r in range(U.shape[1]):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        if np.allclose(u, 0) or np.allclose(v, 0) or np.allclose(w, 0):
            rank_tensors.append(np.zeros(nU*nV*nW))
        else:
            rank_tensors.append(_rank1_tensor(u, v, w))

    # Target tensor
    target = np.zeros(nU*nV*nW)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                target[a*nV*nW + b*nW + c] = 1.0

    # Current sum
    current_sum = np.zeros(nU*nV*nW)
    for t in rank_tensors:
        current_sum += t

    # Try to merge pairs
    merged = list(range(U.shape[1]))  # active rank indices
    # We'll do a greedy: for each pair (i,j), compute ||T_i + T_j - target||^2 vs ||current_sum||^2
    # Actually, we want to see if removing one column and adjusting another can keep us exact.
    # Simpler: try all pairs (i,j), compute T_new = current_sum - T_i - T_j + c*T_merged
    # This is hard. Let's try a different approach: random restarts with rank < 56.

    # Random search for rank r from 40 to 55
    for r in range(40, 56):
        if time.time() > deadline:
            break
        for attempt in range(20):
            if time.time() > deadline:
                break
            U_r = np.zeros((nU, r))
            V_r = np.zeros((nV, r))
            W_r = np.zeros((nW, r))
            for col in range(r):
                # Random sparse patterns
                U_r[np.random.randint(0, nU, 2), col] = 1
                V_r[np.random.randint(0, nV, 2), col] = 1
                W_r[np.random.randint(0, nW, 2), col] = 1
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
                break
        if time.time() > deadline:
            break

    if best_rank < base_rank:
        return best_U, best_V, best_W, best_rank

    return U, V, W, base_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # First try Strassen (rank 56)
    U, V, W = _strassen_block()
    if _verify(U, V, W, n, m, p):
        # Try to improve via local search if time allows
        if time.time() < deadline - 5:
            U2, V2, W2, r2 = _construct_via_search(n, m, p, deadline)
            if r2 < 56:
                return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
