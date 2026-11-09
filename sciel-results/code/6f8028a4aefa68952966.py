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
    # Check all 1-positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check all 0-positions
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
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


def _structured_init(n, m, p, rank, rng):
    """
    Create a structured initialization that respects the tensor structure.
    We initialize based on the "identity" pattern of the multiplication tensor.
    """
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each rank component, assign it to a subset of (i,j,k) triples
    # We want to cover all n*m*p = 60 triples with `rank` components
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_triples = len(triples)
    
    # Assign triples to rank components
    for r in range(rank):
        # Each component handles a subset of triples
        # Use a round-robin or block assignment
        start = r * n_triples // rank
        end = (r + 1) * n_triples // rank
        if start >= end:
            # Add some randomness
            idx = rng.integers(0, n_triples)
            i, j, k = triples[idx]
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
        else:
            for t in range(start, end):
                i, j, k = triples[t]
                # Scale to maintain the sum
                U[i * m + j, r] = 1.0 / (end - start)
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                # Actually, let's use a simpler approach
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    
    # Add small random perturbations to break symmetry
    U += rng.standard_normal((n * m, rank)) * 0.01
    V += rng.standard_normal((m * p, rank)) * 0.01
    W += rng.standard_normal((p * n, rank)) * 0.01
    
    return U, V, W


def _als_step(U, V, W, n, m, p, target_rank, deadline=None):
    """
    One full ALS iteration: update U, V, W in sequence.
    Returns the max residual.
    """
    # Update U
    for a in range(n * m):
        i = a // m
        j = a % m
        M = np.zeros((p, target_rank))
        for k in range(p):
            b = j * p + k
            c = k * n + i
            M[k, :] = V[b, :] * W[c, :]
        U[a, :], res, rank_a, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
    
    # Update V
    for b in range(m * p):
        j = b // p
        k = b % p
        M = np.zeros((n, target_rank))
        for i in range(n):
            a = i * m + j
            c = k * n + i
            M[i, :] = U[a, :] * W[c, :]
        V[b, :], res, rank_v, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
    
    # Update W
    for c in range(p * n):
        k = c // n
        i = c % n
        M = np.zeros((m, target_rank))
        for j in range(m):
            a = i * m + j
            b = j * p + k
            M[j, :] = U[a, :] * V[b, :]
        W[c, :], res, rank_w, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
    
    # Compute max residual
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
    
    # Sample some 0-positions
    rng = np.random.default_rng(42)
    for _ in range(50):
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
            if abs(val) > max_err:
                max_err = abs(val)
    
    return max_err


def _try_als(n, m, p, target_rank, max_iter=200, tol=1e-10, n_restarts=20, deadline=None):
    """
    Try to find a rank-target_rank decomposition using ALS with multiple initializations.
    Returns (U, V, W) numpy arrays if found, else None.
    """
    rng = np.random.default_rng()
    best = None
    best_res = np.inf
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Structured init
            U, V, W = _structured_init(n, m, p, target_rank, rng)
        elif restart % 3 == 1:
            # Random init
            U = rng.standard_normal((n * m, target_rank))
            V = rng.standard_normal((m * p, target_rank))
            W = rng.standard_normal((p * n, target_rank))
        else:
            # Sparse random init
            U = rng.standard_normal((n * m, target_rank))
            V = rng.standard_normal((m * p, target_rank))
            W = rng.standard_normal((p * n, target_rank))
            # Zero out some entries
            for r in range(target_rank):
                mask = rng.random(n * m) < 0.5
                U[mask, r] = 0
                mask = rng.random(m * p) < 0.5
                V[mask, r] = 0
                mask = rng.random(p * n) < 0.5
                W[mask, r] = 0
        
        prev_res = np.inf
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            max_err = _als_step(U, V, W, n, m, p, target_rank, deadline)
            
            if max_err < tol:
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                if _validate(U_snap, V_snap, W_snap, n, m, p, tol=1e-5):
                    return U_snap, V_snap, W_snap
            
            if max_err < best_res:
                best_res = max_err
                best = (U.copy(), V.copy(), W.copy())
            
            if max_err < prev_res * 0.999:
                prev_res = max_err
            else:
                # Stagnation, break early
                if it > 20:
                    break
    
    return best


def _polish(U, V, W, n, m, p, max_iter=100, tol=1e-12, deadline=None):
    """
    High-precision polish step with tighter tolerance.
    """
    target_rank = U.shape[1]
    prev_res = np.inf
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            i = a // m
            j = a % m
            M = np.zeros((p, target_rank))
            for k in range(p):
                b = j * p + k
                c = k * n + i
                M[k, :] = V[b, :] * W[c, :]
            U[a, :], res, rank_a, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
        
        # Update V
        for b in range(m * p):
            j = b // p
            k = b % p
            M = np.zeros((n, target_rank))
            for i in range(n):
                a = i * m + j
                c = k * n + i
                M[i, :] = U[a, :] * W[c, :]
            V[b, :], res, rank_v, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
        
        # Update W
        for c in range(p * n):
            k = c // n
            i = c % n
            M = np.zeros((m, target_rank))
            for j in range(m):
                a = i * m + j
                b = j * p + k
                M[j, :] = U[a, :] * V[b, :]
            W[c, :], res, rank_w, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
        
        # Compute max residual
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
        
        if max_err < tol:
            return U, V, W
        
        if max_err < prev_res * 0.999:
            prev_res = max_err
        else:
            break
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    # Try to find a low-rank decomposition, starting from small ranks
    for target_rank in [18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 45, 50]:
        if time.time() > deadline - 20:
            break
        
        result = _try_als(n, m, p, target_rank, max_iter=150, tol=1e-10, n_restarts=25, deadline=deadline - 20)
        if result is not None:
            U, V, W = result
            if np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W)):
                # Polish step
                U, V, W = _polish(U, V, W, n, m, p, max_iter=50, tol=1e-12, deadline=deadline - 5)
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                if _validate(U_snap, V_snap, W_snap, n, m, p, tol=1e-5):
                    return {
                        "U": _to_halfint_list(U_snap),
                        "V": _to_halfint_list(V_snap),
                        "W": _to_halfint_list(W_snap)
                    }
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": _to_halfint_list(U),
        "V": _to_halfint_list(V),
        "W": _to_halfint_list(W)
    }

# EVOLVE-BLOCK-END
