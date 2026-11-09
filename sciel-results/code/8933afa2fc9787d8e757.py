"""Matrix multiplication 2,3,10: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _to_halfint_list(mat):
    """Convert a float matrix to a list-of-lists of integers/half-integers."""
    doubled = np.round(mat * 2) / 2.0
    out = []
    for row in doubled:
        r = []
        for val in row:
            fval = float(val)
            if abs(fval - round(fval)) < 1e-9:
                r.append(int(round(fval)))
            else:
                r.append(fval)
        out.append(r)
    return out


def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check some zero positions
    rng = np.random.default_rng(12345)
    for _ in range(100):
        a = rng.integers(0, n * m)
        b = rng.integers(0, m * p)
        c = rng.integers(0, p * n)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        if not (j_a == j_b and k_b == k_c and i_a == i_c):
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > tol:
                return False
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition: rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _block_decompose(n1, m, p1, n2, p2, rank1, U1, V1, W1, rank2, U2, V2, W2):
    """
    Combine two block decompositions.
    Block 1: n1 x m x p1, Block 2: n2 x m x p2.
    Total: n1+n2 x m x (p1+p2) = n x m x p.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n = n1 + n2
    p = p1 + p2
    total_rank = rank1 + rank2
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block 1: rows 0..n1*m-1 of U, columns 0..p1 of V, columns 0..p1*n of W (for k in 0..p1-1, i in 0..n1-1)
    # U1 has shape (n1*m, rank1)
    U[:n1*m, :rank1] = U1
    # V1 has shape (m*p1, rank1) - maps to rows 0..m*p1-1 of V
    V[:m*p1, :rank1] = V1
    # W1 has shape (p1*n1, rank1) - maps to W[k*n + i] where k in 0..p1-1, i in 0..n1-1
    # In the combined W, index is k*n + i for k in 0..p-1, i in 0..n-1
    # For block 1: k in 0..p1-1, i in 0..n1-1, so W index = k*n + i
    for k in range(p1):
        for i in range(n1):
            W1_idx = k * n1 + i
            W_combined_idx = k * n + i
            W[W_combined_idx, :rank1] = W1[W1_idx, :rank1]
    
    # Block 2: rows n1*m..n*m-1 of U, columns m*p1..m*p of V, columns p1*n..p*n of W
    U[n1*m:, rank1:] = U2
    V[m*p1:, rank1:] = V2
    # W2 has shape (p2*n2, rank2) - maps to W[k*n + i] where k in p1..p-1, i in n1..n-1
    for k in range(p2):
        for i in range(n2):
            W2_idx = k * n2 + i
            W_combined_idx = (p1 + k) * n + (n1 + i)
            W[W_combined_idx, rank1:] = W2[W2_idx, :rank2]
    
    return U, V, W


def _try_als(n, m, p, target_rank, max_iter=100, tol=1e-8, n_restarts=10, deadline=None):
    """
    Try to find a rank-target_rank decomposition using ALS with random restarts.
    Returns (U, V, W) numpy arrays if found, else None.
    """
    rng = np.random.default_rng()
    best = None
    best_res = np.inf
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        U = rng.standard_normal((n * m, target_rank))
        V = rng.standard_normal((m * p, target_rank))
        W = rng.standard_normal((p * n, target_rank))
        
        prev_res = np.inf
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            # Solve for U
            for a in range(n * m):
                i = a // m
                j = a % m
                M = np.zeros((p, target_rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                U[a, :], res, rank_a, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
            
            # Solve for V
            for b in range(m * p):
                j = b // p
                k = b % p
                M = np.zeros((n, target_rank))
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    M[i, :] = U[a, :] * W[c, :]
                V[b, :], res, rank_v, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
            
            # Solve for W
            for c in range(p * n):
                k = c // n
                i = c % n
                M = np.zeros((m, target_rank))
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    M[j, :] = U[a, :] * V[b, :]
                W[c, :], res, rank_w, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
            
            # Check residual
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            for _ in range(20):
                a = rng.integers(0, n * m)
                b = rng.integers(0, m * p)
                c = rng.integers(0, p * n)
                i_a = a // m; j_a = a % m
                j_b = b // p; k_b = b % p
                k_c = c // n; i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_a == i_c):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > max_err:
                        max_err = abs(val)
            
            if max_err < tol:
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                if _validate(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                else:
                    pass
            
            if max_err < best_res:
                best_res = max_err
                best = (U.copy(), V.copy(), W.copy())
            
            if max_err < tol * 10:
                break
    
    return best


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find a low-rank decomposition
    for target_rank in [20, 22, 24, 30, 40, 50]:
        if time.time() > deadline - 15:
            break
        result = _try_als(n, m, p, target_rank, max_iter=80, tol=1e-8, n_restarts=15, deadline=deadline - 15)
        if result is not None:
            U, V, W = result
            if np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W)):
                if _validate(U, V, W, n, m, p):
                    return {
                        "U": _to_halfint_list(U),
                        "V": _to_halfint_list(V),
                        "W": _to_halfint_list(W)
                    }
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": _to_halfint_list(U),
        "V": _to_halfint_list(V),
        "W": _to_halfint_list(W)
    }

# EVOLVE-BLOCK-END
