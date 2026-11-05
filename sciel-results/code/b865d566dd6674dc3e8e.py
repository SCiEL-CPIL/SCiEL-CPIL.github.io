"""Climate dynamics: unresolved-tendency closure via spline-informed polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import UnivariateSpline


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Remove any non-finite entries
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Normalize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Normalize y
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Strategy: Use a piecewise cubic spline to capture local oscillatory structure,
    # then project onto degree-5 polynomial basis.
    
    # First, get a direct polynomial fit as baseline
    V = np.vander(x_norm, N=6, increasing=True)
    
    # Try ridge regression with various lambda values
    best_coeff_norm = None
    best_mse = np.inf
    
    for lam in [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
        if time.time() > deadline - 20.0:
            break
        try:
            A = V.T @ V + lam * np.eye(6)
            b = V.T @ y_norm
            coeff_norm = np.linalg.solve(A, b)
            pred_norm = V @ coeff_norm
            mse = np.mean((pred_norm - y_norm) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_coeff_norm = coeff_norm.copy()
        except Exception:
            continue
    
    if best_coeff_norm is None:
        # Fallback: least squares
        try:
            best_coeff_norm, _, _, _ = np.linalg.lstsq(V, y_norm, rcond=None)
        except Exception:
            best_coeff_norm = np.zeros(6)
    
    # Now try spline-informed approach:
    # Fit a cubic spline with quantile-based knots, then project to polynomial
    if time.time() < deadline - 30.0:
        try:
            # Determine number of knots based on data size
            # Use quantile-based knot placement
            n_knots_options = [8, 12, 16, 20]
            
            best_spline_coeff_norm = None
            best_spline_mse = best_mse  # must beat direct polynomial to be used
            
            for n_knots in n_knots_options:
                if time.time() > deadline - 20.0:
                    break
                
                # Place knots at quantiles of x_norm
                quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]  # interior knots
                knots = np.quantile(x_norm, quantiles)
                
                # Add boundary knots
                x_min, x_max = np.min(x_norm), np.max(x_norm)
                all_knots = np.concatenate(([x_min - 0.1], knots, [x_max + 0.1]))
                
                # Fit cubic spline with these knots
                # Use scipy's UnivariateSpline with fixed knots
                # We'll use a weighted least squares approach with B-spline basis
                
                # Build B-spline basis matrix
                from scipy.interpolate import BSpline
                from scipy.special import comb
                
                # Degree 3, knots
                k = 3
                # Need enough knots: n_knots + k + 1 total
                # BSpline expects knots of length n_basis + k + 1
                # For n interior knots, total knots = n_interior + k + 2
                n_basis = len(all_knots) - k - 1
                if n_basis < 1:
                    continue
                
                # Evaluate B-spline basis at x_norm
                # BSpline.basis_element gives a single basis function
                # We need to build the full basis matrix
                B = np.zeros((len(x_norm), n_basis))
                for j in range(n_basis):
                    try:
                        B[:, j] = BSpline.basis_element(all_knots, lb=j, k=k)(x_norm)
                    except Exception:
                        continue
                
                # Remove columns that are all zeros (degenerate basis functions)
                col_sums = np.abs(B).sum(axis=0)
                valid_cols = col_sums > 1e-12
                if np.sum(valid_cols) < 2:
                    continue
                B = B[:, valid_cols]
                
                # Fit spline coefficients
                try:
                    spline_coeffs, _, _, _ = np.linalg.lstsq(B, y_norm, rcond=None)
                    spline_pred = B @ spline_coeffs
                    spline_mse = np.mean((spline_pred - y_norm) ** 2)
                    
                    if spline_mse < best_spline_mse:
                        best_spline_mse = spline_mse
                        # Now project this spline onto degree-5 polynomial basis
                        # Evaluate spline on a dense grid and fit polynomial to those values
                        if time.time() < deadline - 15.0:
                            # Dense evaluation grid
                            x_eval = np.linspace(x_min - 0.5, x_max + 0.5, 200)
                            y_spline_eval = np.zeros_like(x_eval)
                            for j in range(n_basis):
                                if valid_cols[j]:
                                    try:
                                        y_spline_eval += spline_coeffs[j] * BSpline.basis_element(all_knots, lb=j, k=k)(x_eval)
                                    except Exception:
                                        continue
                            
                            # Fit degree-5 polynomial to spline values
                            V_eval = np.vander(x_eval, N=6, increasing=True)
                            try:
                                poly_coeffs, _, _, _ = np.linalg.lstsq(V_eval, y_spline_eval, rcond=None)
                                # Check how well this polynomial approximates the spline
                                poly_pred_eval = V_eval @ poly_coeffs
                                proj_mse = np.mean((poly_pred_eval - y_spline_eval) ** 2)
                                
                                # Also check against original data
                                V_orig = np.vander(x_norm, N=6, increasing=True)
                                poly_pred_orig = V_orig @ poly_coeffs
                                orig_mse = np.mean((poly_pred_orig - y_norm) ** 2)
                                
                                if orig_mse < best_spline_mse:
                                    best_spline_mse = orig_mse
                                    best_spline_coeff_norm = poly_coeffs.copy()
                            except Exception:
                                pass
                except Exception:
                    continue
        except Exception:
            pass
    
    # Choose the best between direct polynomial and spline-projected polynomial
    final_coeff_norm = best_coeff_norm.copy()
    
    if best_spline_coeff_norm is not None:
        # Compare: use the one with lower training MSE
        V_check = np.vander(x_norm, N=6, increasing=True)
        mse_direct = np.mean((V_check @ best_coeff_norm - y_norm) ** 2)
        mse_spline_proj = np.mean((V_check @ best_spline_coeff_norm - y_norm) ** 2)
        
        if mse_spline_proj < mse_direct:
            final_coeff_norm = best_spline_coeff_norm.copy()
    
    # Transform back to original scale
    try:
        best_coeff = _transform_poly_to_original(final_coeff_norm, x_mean, x_std, y_mean, y_std)
    except Exception:
        # Fallback
        try:
            V_full = np.vander(x, N=6, increasing=True)
            best_coeff, _, _, _ = np.linalg.lstsq(V_full, y, rcond=None)
        except Exception:
            best_coeff = np.zeros(6)
    
    # Ensure finite output
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.where(np.isfinite(best_coeff), best_coeff, 0.0)
    
    return best_coeff


def _transform_poly_to_original(coeff_norm, x_mean, x_std, y_mean, y_std):
    """
    Transform polynomial coefficients from normalized space to original space.
    y = y_mean + y_std * sum_{i=0}^{5} c_i * ((x - x_mean)/x_std)^i
    """
    d = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            binom_val = _binom(i, j)
            term = coeff_norm[i] / (x_std ** i) * binom_val * ((-x_mean) ** (i - j))
            s += term
        d[j] = y_std * s
    d[0] += y_mean
    return d


def _binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
# EVOLVE-BLOCK-END
