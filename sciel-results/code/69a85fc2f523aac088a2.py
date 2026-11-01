"""Matrix multiplication 2x2x2: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-10):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0.0
                for c in range(rank):
                    acc += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(acc - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition(n, m, p):
    """Return the rank-7 Strassen decomposition for 2x2 matrix multiplication."""
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return (np.array(U_data, dtype=float),
            np.array(V_data, dtype=float),
            np.array(W_data, dtype=float))

def _schoolbook_decomposition(n, m, p):
    """Return the rank-8 schoolbook decomposition (always valid)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _attempt_rank6_search(n, m, p, time_budget):
    """
    Attempt to find a rank-6 decomposition by solving the polynomial system.
    Uses a randomized approach: set up the 8 tensor equations and try to
    solve via iterative refinement starting from multiple random initializations.
    Returns (U, V, W) if a valid rank-6 decomposition is found, else None.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    rank = 6

    # The tensor T[i,j,k] = 1 for all i,j,k in {0,1} (8 equations)
    # We need: for each (i,j,k): sum_{c=0}^{5} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1

    # Use scipy if available for more robust nonlinear solving
    try:
        from scipy.optimize import fsolve
    except ImportError:
        fsolve = None

    best_res = None
    best_norm = float('inf')

    # Multiple random restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Random initialization
        np.random.seed(restart * 7919)
        U_init = np.random.randn(n*m, rank)
        V_init = np.random.randn(m*p, rank)
        W_init = np.random.randn(p*n, rank)

        # Scale to make entries O(1)
        U_init /= np.max(np.abs(U_init)) if np.max(np.abs(U_init)) > 0 else 1
        V_init /= np.max(np.abs(V_init)) if np.max(np.abs(V_init)) > 0 else 1
        W_init /= np.max(np.abs(W_init)) if np.max(np.abs(W_init)) > 0 else 1

        if fsolve is not None:
            # Build the system of 8 equations in 72 unknowns (4*6 + 4*6 + 4*6)
            # Since we have more unknowns than equations, the system is underdetermined.
            # Fix some variables to make it determined, or use least squares.
            # Strategy: fix first column of U, V, W to specific values to break
            # the scaling/symmetry, then solve for the rest.

            # Total unknowns: 4*6*3 = 72, equations: 8
            # This is highly underdetermined. Use a different approach:
            # Fix the first row of U, first row of V, first row of W partially.

            # Alternative: use the fact that rank-6 would be non-integer typically.
            # The system T = sum_c u_c ⊗ v_c ⊗ w_c with 6 terms.

            # Let's parameterize: fix U[:,0], V[:,0], W[:,0] to 1,1,1 (first product is schoolbook-like)
            # Then solve for the remaining 5 products.
            # Actually, let's just try to minimize the residual.

            # Build residual function
            def residual(flat):
                U_r = flat[0:24].reshape(4, 6)
                V_r = flat[24:48].reshape(4, 6)
                W_r = flat[48:72].reshape(4, 6)
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                            res.append(val - 1.0)
                return np.array(res)

            x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
            result, success, info, _ = fsolve(residual, x0, full_output=True, xtol=1e-12)

            if success:
                U_r = result[0:24].reshape(4, 6)
                V_r = result[24:48].reshape(4, 6)
                W_r = result[48:72].reshape(4, 6)
                res_norm = np.max(np.abs(residual(result)))
                if res_norm < 1e-10:
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p, rank):
                        return U_r, V_r, W_r
                if res_norm < best_norm:
                    best_norm = res_norm
                    best_res = (U_r, V_r, W_r)
        else:
            # Without scipy, use a simple gradient-free approach: random search with local perturbation
            # This is unlikely to find a solution if one exists, but let's try.
            best_local = (U_init, V_init, W_init)
            best_local_norm = float('inf')
            for it in range(100):
                if time.time() > deadline - 5.0:
                    break
                U_c, V_c, W_c = best_local
                # Compute residual
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_c[i*m+j,c]*V_c[j*p+k,c]*W_c[k*n+i,c] for c in range(rank))
                            res.append(val - 1.0)
                res_norm = max(abs(r) for r in res)
                if res_norm < 1e-10:
                    if _verify_decomposition(U_c, V_c, W_c, n, m, p, rank):
                        return U_c, V_c, W_c
                if res_norm < best_local_norm:
                    best_local_norm = res_norm
                    best_local = (U_c.copy(), V_c.copy(), W_c.copy())
                # Perturb
                scale = max(0.1, (1 - it/100) * 0.5)
                U_c = U_c + scale * np.random.randn(4, 6) * 0.01
                V_c = V_c + scale * np.random.randn(4, 6) * 0.01
                W_c = W_c + scale * np.random.randn(4, 6) * 0.01

            if best_local_norm < best_norm:
                best_norm = best_local_norm
                best_res = best_local

    # If we found something close but not exact, try to polish
    if best_res is not None and best_norm < 1e-6:
        U_r, V_r, W_r = best_res
        # Try to round to nearby rational values
        # Check if entries are close to simple fractions
        for U_r, V_r, W_r in [(best_res, best_res, best_res)]:
            pass
        # Just check if it's valid
        if _verify_decomposition(best_res[0], best_res[1], best_res[2], n, m, p, rank, tol=1e-9):
            return best_res

    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    The proven optimal rank is 7 (Strassen). Attempts rank-6 search first as a
    sanity check, then returns the verified Strassen decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2

    # Compute deadline with safety margin
    start_time = time.time()
    total_budget = 100.0  # Leave 20s margin for the 120s limit
    deadline = start_time + total_budget

    # Attempt rank-6 search with a time budget
    rank6_budget = min(60.0, deadline - time.time())
    if rank6_budget > 2.0:
        result6 = _attempt_rank6_search(n, m, p, rank6_budget)
        if result6 is not None:
            U, V, W = result6
            if _verify_decomposition(U, V, W, n, m, p, 6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Check time
    if time.time() > deadline - 5.0:
        # Emergency: return Strassen immediately
        U, V, W = _strassen_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Return the verified rank-7 Strassen decomposition
    U, V, W = _strassen_decomposition(n, m, p)

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, 7):
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
