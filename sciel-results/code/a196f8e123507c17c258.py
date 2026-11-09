import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math
import sys

def construct():
    start_time = time.time()
    # Hard deadline: leave margin for the 630s kill limit.
    deadline = start_time + 580

    m = 12
    n = 1

    # Precomputation: Transformation matrix from standard basis to Laguerre basis
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)

    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    # Weights for the Laguerre norm: <L_k, L_l> = delta_kl * (k+1) for alpha=1
    weights = np.arange(13) + 1

    # Positive-definite matrix parameterization:
    # D = D0 + X.T @ X, where D0 is a fixed PD baseline and X is 6x12.
    # Eigenvalues of D are the 12 positive roots.
    # D0 is chosen so that eigenvalues span a reasonable range.
    # D0 = diag(d_i) where d_i are log-spaced values.
    d_baseline = np.logspace(np.log10(1.0), np.log10(1000.0), 12)
    D0 = np.diag(d_baseline)

    def roots_from_X(X_flat):
        """Convert flat X vector (size 72) to 12 positive roots via eigenvalues of D0 + X.T@X."""
        X = X_flat.reshape(6, 12)
        D = D0 + X.T @ X
        # Symmetrize to handle numerical asymmetry
        D = 0.5 * (D + D.T)
        try:
            eigvals = np.linalg.eigvalsh(D)
        except np.linalg.LinAlgError:
            return None
        # Eigenvalues should be positive (D0 is PD, X.T@X is PSD)
        if np.any(eigvals <= 0) or np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
            return None
        return eigvals

    def compute_cost(X_flat):
        """
        Compute the cost function (upper bound on C4) given X parameters.
        Cost = 1 / ||P||^2_Laguerre
        """
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e20

        roots = roots_from_X(X_flat)
        if roots is None:
            return 1e20

        # Build polynomial P(x) = prod(x - r_i)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]

        # Compute Laguerre coefficients c
        c = A_T_inv @ p

        # Compute norm squared in Laguerre basis
        norm_sq = np.sum(c**2 * weights)

        if norm_sq <= 1e-10:
            return 1e20

        return 1.0 / norm_sq

    best_val = 1e20
    best_X = None

    # X is 6x12 = 72 variables.
    # Bounds: X entries in [-10, 10] should be sufficient.
    bounds_X = [(-10.0, 10.0)] * 72

    def update_best(val, x):
        nonlocal best_val, best_X
        if val < best_val:
            best_val = val
            best_X = x.copy()

    # --- Phase 1: Global Search (Differential Evolution) ---
    # Single longer DE run with larger population for thorough exploration.
    de_seeds = [1, 2, 3]

    for seed in de_seeds:
        if time.time() - start_time > deadline - 90:
            break
        try:
            res = differential_evolution(
                compute_cost,
                bounds_X,
                seed=seed,
                maxiter=200,
                popsize=20,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            # Polish the result
            if time.time() - start_time < deadline - 15:
                try:
                    res_polish = minimize(
                        compute_cost,
                        res.x,
                        method='L-BFGS-B',
                        bounds=bounds_X,
                        options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-10}
                    )
                    val = res_polish.fun
                    x = res_polish.x
                except:
                    val = res.fun
                    x = res.x
            else:
                val = res.fun
                x = res.x

            update_best(val, x)
        except Exception:
            pass

    # --- Phase 2: Local Search from Diverse Initial Points ---
    # Generate diverse initial X matrices.
    rng = np.random.RandomState(42)

    initial_points = []

    # Seed 1: Zero X (roots = eigenvalues of D0)
    initial_points.append(np.zeros(72))

    # Seed 2: Small random perturbations
    for _ in range(15):
        X_init = rng.uniform(-2.0, 2.0, 72)
        initial_points.append(X_init)

    # Seed 3: Larger random perturbations
    for _ in range(10):
        X_init = rng.uniform(-5.0, 5.0, 72)
        initial_points.append(X_init)

    # Seed 4: Structured patterns
    for scale in [0.5, 1.0, 2.0, 5.0]:
        X_struct = np.zeros((6, 12))
        for i in range(6):
            X_struct[i, :] = scale * (np.cos(np.arange(12) * (i + 1) * 0.3) + 0.5)
        initial_points.append(X_struct.flatten())

    for i, init_x in enumerate(initial_points):
        if time.time() - start_time > deadline - 25:
            break

        try:
            # L-BFGS-B
            res_lf = minimize(
                compute_cost,
                init_x,
                method='L-BFGS-B',
                bounds=bounds_X,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-9}
            )
            update_best(res_lf.fun, res_lf.x)

            # Nelder-Mead fine-tuning
            if time.time() - start_time < deadline - 10:
                res_nm = minimize(
                    compute_cost,
                    res_lf.x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                update_best(res_nm.fun, res_nm.x)

        except Exception:
            pass

    # --- Final Fallback ---
    if best_X is None:
        # If everything failed, use zero X (roots = eigenvalues of D0)
        best_X = np.zeros(72)
        best_val = compute_cost(best_X)

    # Convert best X to roots
    best_roots = roots_from_X(best_X)

    if best_roots is None:
        # Ultimate fallback
        best_roots = np.logspace(0, 3, 12)

    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)

    return list(best_roots)

# Main entry point
if __name__ == "__main__":
    # For local testing
    roots = construct()
    print(roots)
