"""Combined program: schoolbook fallback + principled linear-in-W search with deadline guard."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    rank = U_np.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.einsum('c,c,c->', U_np[i*m+j, :], V_np[j*p+k, :], W_np[k*n+i, :])
                if abs(val - 1.0) > tol:
                    ok = False
    return ok


def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using einsum over all indices at once."""
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_np.reshape(n, m, -1)
    V3 = V_np.reshape(m, p, -1)
    W3 = W_np.reshape(p, n, -1)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.all(np.abs(T - 1.0) < tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _try_rank_densified(rank, n, m, p, deadline, max_attempts, seed_offset=0):
    """
    Fix random U,V with denser entries; solve linear system for W per (k,i) block.
    Use a wider range of values for U,V to increase chances of consistency.
    """
    rng = np.random.RandomState(42 + seed_offset)
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        # Strategy: try different density levels with better distributions
        stage = attempt % 6
        if stage == 0:
            # Dense binary: {-1, 1}
            U = rng.choice([-1, 1], size=(n*m, rank)).astype(float)
            V = rng.choice([-1, 1], size=(m*p, rank)).astype(float)
        elif stage == 1:
            # Ternary: {-1, 0, 1} with more zeros
            U = rng.choice([-1, 0, 1], size=(n*m, rank), p=[0.25, 0.5, 0.25]).astype(float)
            V = rng.choice([-1, 0, 1], size=(m*p, rank), p=[0.25, 0.5, 0.25]).astype(float)
        elif stage == 2:
            # Half-integer dense
            vals = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
            U = rng.choice(vals, size=(n*m, rank))
            V = rng.choice(vals, size=(m*p, rank))
        elif stage == 3:
            # Sparse with larger values
            U = rng.choice([-2, -1, 0, 1, 2], size=(n*m, rank), p=[0.1, 0.2, 0.4, 0.2, 0.1]).astype(float)
            V = rng.choice([-2, -1, 0, 1, 2], size=(m*p, rank), p=[0.1, 0.2, 0.4, 0.2, 0.1]).astype(float)
        elif stage == 4:
            # Very dense half-integers
            vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            U = rng.choice(vals, size=(n*m, rank))
            V = rng.choice(vals, size=(m*p, rank))
        else:
            # Gaussian-like, rounded to half-integers
            U = 0.5 * np.round(2.0 * rng.randn(n*m, rank) * 0.8)
            V = 0.5 * np.round(2.0 * rng.randn(m*p, rank) * 0.8)
        
        # For each (k,i), solve for W[k*n+i, :] of length rank
        # Equations over j: sum_c U[i*m+j,c]*V[j*p+k,c]*Wc[c] = 1, j=0..m-1
        W = np.zeros((p*n, rank))
        feasible = True
        for k in range(p):
            for i in range(n):
                # Build m x rank matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                rhs = np.ones(m)
                # Solve A @ Wc = rhs (least squares), then check residual
                try:
                    Wc, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    feasible = False
                    break
                # Check consistency
                if np.max(np.abs(A @ Wc - rhs)) > 1e-7:
                    feasible = False
                    break
                # Round to nearest half-integer to satisfy constraint
                Wc_rounded = 0.5 * np.round(2.0 * Wc)
                # Check if rounding introduced significant error
                if np.max(np.abs(A @ Wc_rounded - rhs)) > 1e-7:
                    # Try a local search: perturb nearby half-integers
                    found = False
                    for pert_attempt in range(10):
                        if time.time() > deadline:
                            return None
                        # Randomly flip a few entries by ±0.5
                        Wc_pert = Wc_rounded.copy()
                        num_flips = rng.randint(1, 5)
                        idx = rng.choice(rank, size=num_flips, replace=False)
                        for ii in idx:
                            Wc_pert[ii] += rng.choice([-0.5, 0.5])
                        if np.max(np.abs(A @ Wc_pert - rhs)) < 1e-7:
                            Wc_rounded = Wc_pert
                            found = True
                            break
                    if not found:
                        feasible = False
                        break
                W[k*n+i, :] = Wc_rounded
        if feasible:
            # Verify fully with rounded values
            if _verify_fast(U, V, W, n, m, p):
                return U, V, W
    return None


def _try_rank_with_polish(rank, n, m, p, deadline, max_attempts, seed_offset=0):
    """
    Similar to _try_rank_densified but with an additional polish phase:
    after finding a nearly-feasible solution, do a local refinement of W.
    """
    rng = np.random.RandomState(1000 + seed_offset)
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        # Generate random U,V with varied strategies
        if attempt % 3 == 0:
            U = rng.choice([-1, 0, 1], size=(n*m, rank), p=[0.3, 0.4, 0.3]).astype(float)
            V = rng.choice([-1, 0, 1], size=(m*p, rank), p=[0.3, 0.4, 0.3]).astype(float)
        elif attempt % 3 == 1:
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = rng.choice(vals, size=(n*m, rank))
            V = rng.choice(vals, size=(m*p, rank))
        else:
            # Sparser with larger magnitude
            U = rng.choice([-2, -1, 0, 1, 2], size=(n*m, rank), p=[0.1, 0.2, 0.4, 0.2, 0.1]).astype(float)
            V = rng.choice([-2, -1, 0, 1, 2], size=(m*p, rank), p=[0.1, 0.2, 0.4, 0.2, 0.1]).astype(float)
        
        # For each (k,i), solve for W
        W = np.zeros((p*n, rank))
        feasible = True
        max_residual = 0.0
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                rhs = np.ones(m)
                try:
                    Wc, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    feasible = False
                    break
                resid = np.max(np.abs(A @ Wc - rhs))
                max_residual = max(max_residual, resid)
                if resid > 1e-5:
                    feasible = False
                    break
                Wc_rounded = 0.5 * np.round(2.0 * Wc)
                if np.max(np.abs(A @ Wc_rounded - rhs)) > 1e-5:
                    # Try local refinement
                    found = False
                    for refine_attempt in range(8):
                        if time.time() > deadline:
                            return None
                        Wc_refine = Wc_rounded.copy()
                        num_flips = rng.randint(1, 4)
                        idx = rng.choice(rank, size=num_flips, replace=False)
                        for ii in idx:
                            Wc_refine[ii] += rng.choice([-0.5, 0.5])
                        if np.max(np.abs(A @ Wc_refine - rhs)) < 1e-5:
                            Wc_rounded = Wc_refine
                            found = True
                            break
                    if not found:
                        feasible = False
                        break
                W[k*n+i, :] = Wc_rounded
        if feasible and _verify_fast(U, V, W, n, m, p):
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 135.0  # safety margin below 150s kill

    # Best-so-far: schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    U0_list = [[float(x) for x in row] for row in U0]
    V0_list = [[float(x) for x in row] for row in V0]
    W0_list = [[float(x) for x in row] for row in W0]
    best = {"U": U0_list, "V": V0_list, "W": W0_list}
    best_rank = n * m * p

    # Try ranks from 124 down to ~50
    for rank in range(best_rank - 1, 48, -1):
        if time.time() > deadline - 5.0:
            break
        remaining = deadline - 5.0 - time.time()
        # Allocate more attempts to densified search (it has more variety)
        attempts_densified = max(10, int(remaining * 3))
        attempts_polish = max(8, int(remaining * 2))
        
        # Try densified search first
        result = _try_rank_densified(rank, n, m, p, deadline - 5.0, attempts_densified, seed_offset=rank)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            break
        
        # Try polish-based search
        if time.time() > deadline - 5.0:
            break
        result = _try_rank_with_polish(rank, n, m, p, deadline - 5.0, attempts_polish, seed_offset=rank + 100)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            break

    return best
# EVOLVE-BLOCK-END
