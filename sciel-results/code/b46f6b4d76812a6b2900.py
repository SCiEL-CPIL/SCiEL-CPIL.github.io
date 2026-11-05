import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]
DEADLINE_MARGIN = 10.0  # seconds to spare

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check exact decomposition: all 45 positive entries = 1, all other entries = 0."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    # All 45 positive constraints: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(np.dot(U[i * m + j, :], V[j * p + k, :]), W[k * n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    # Negative constraints: all other (u, v, w) combinations must sum to 0
    # Total entries: (n*m)*(m*p)*(p*n) = 9*15*15 = 2025, of which 45 are positive
    # Check all 2025 entries
    for u in range(n * m):
        i_u = u // m
        j_u = u % m
        for v in range(m * p):
            j_v = v // p
            k_v = v % p
            # if j_u != j_v: must be 0 (j-link broken)
            for w in range(p * n):
                k_w = w // n
                i_w = w % n
                # Skip positive entries
                if i_u == i_w and j_u == j_v and k_v == k_w:
                    continue
                s = np.dot(np.dot(U[u, :], V[v, :]), W[w, :])
                if abs(s) > 1e-9:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using tensor reconstruction via einsum."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    # Reconstruct full tensor T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    # Check 45 positive entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > 1e-9:
                    return False
    # Check all negative entries
    for u in range(n * m):
        i_u = u // m
        j_u = u % m
        for v in range(m * p):
            j_v = v // p
            k_v = v % p
            for w in range(p * n):
                k_w = w // n
                i_w = w % n
                if i_u == i_w and j_u == j_v and k_v == k_w:
                    continue
                if abs(T[u, v, w]) > 1e-9:
                    return False
    return True


def _try_rank(R, n, m, p, time_budget, rng):
    """Try to find a rank-R decomposition using randomized cut-and-paste."""
    if R <= 0:
        return None
    if R >= n * m * p:
        U, V, W = _build_schoolbook(n, m, p)
        return U[:, :R] if R >= n*m*p else None

    start = time.time()
    best = None

    for attempt in range(200):
        if time.time() - start > time_budget:
            break
        # Randomly group the 45 schoolbook terms into R groups
        # Each group will be combined into one tensor term
        indices = list(range(n * m * p))
        rng.shuffle(indices)
        # Assign each index to one of R groups
        # Strategy: first R-1 groups get at least 1 element, last gets the rest
        groups = [[] for _ in range(R)]
        for idx, i in enumerate(indices):
            if idx < R - 1:
                groups[idx].append(i)
            else:
                groups[-1].append(i)
        # Ensure no empty groups (for R < 45, last group gets 45 - (R-1) elements)
        # Actually let's redistribute more evenly
        sizes = [1] * R
        remaining = n * m * p - R
        pos = 0
        while remaining > 0:
            sizes[pos % R] += 1
            remaining -= 1
            pos += 1
        groups = []
        pos = 0
        for s in sizes:
            groups.append(indices[pos:pos + s])
            pos += s

        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))

        for c, group in enumerate(groups):
            # For each term in the group, pick random coefficients
            for idx in group:
                # idx = (i * m + j) * p + (j * p + k)
                # Actually idx maps to (i, j, k) via:
                # i = idx // (m * p)
                # rem = idx % (m * p)
                # j = rem // p
                # k = rem % p
                i = idx // (m * p)
                rem = idx % (m * p)
                j = rem // p
                k = rem % p
                # Random coefficient (to create linear combinations)
                coeff = rng.standard_normal()
                U[i * m + j, c] += coeff
                V[j * p + k, c] += coeff
                W[k * n + i, c] += coeff

        # Try a few variations: also try with coeff = 1 for all (pure grouping)
        # And try with random signs
        if attempt % 3 == 1:
            # All coefficients = 1
            U[:] = 0
            V[:] = 0
            W[:] = 0
            for c, group in enumerate(groups):
                for idx in group:
                    i = idx // (m * p)
                    rem = idx % (m * p)
                    j = rem // p
                    k = rem % p
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
        elif attempt % 3 == 2:
            # Random signs
            U[:] = 0
            V[:] = 0
            W[:] = 0
            for c, group in enumerate(groups):
                for idx in group:
                    i = idx // (m * p)
                    rem = idx % (m * p)
                    j = rem // p
                    k = rem % p
                    s = 1.0 if rng.random() < 0.5 else -1.0
                    U[i * m + j, c] += s
                    V[j * p + k, c] += s
                    W[k * n + i, c] += s

        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W

    return None


def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = 150.0 - DEADLINE_MARGIN

    # Schoolbook baseline (rank 45)
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find lower rank decompositions
    # Known lower bounds: for M_{nmp}, rank >= max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But realistically for 3x3x5, the rank is likely close to 45 or a bit less
    # Try ranks from 44 down to, say, 30 (limited by time)
    rng = np.random.default_rng(42)

    # Allocate time: try each candidate rank with limited budget
    ranks_to_try = list(range(n * m * p - 1, max(n * m * p - 15, 14), -1))
    # That's ranks 44 down to 30 (15 ranks)
    # Give each rank a time budget
    time_per_rank = 3.0  # 15 * 3 = 45s total, leaving plenty of margin

    for R in ranks_to_try:
        if time.time() > deadline:
            break
        result = _try_rank(R, n, m, p, time_per_rank, rng)
        if result is not None:
            U, V, W = result
            best_rank = R
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # Stop: we found the best (lowest) rank in our search order
            break

    return best_result
