"""
Strategy: Replace the global degree-5 monomial polynomial with a piecewise cubic
spline (natural boundary conditions) defined by quantile-based knots derived from
the training data. This better captures the non-polynomial, oscillatory structure
of the Lorenz-96 subgrid tendency while maintaining smoothness. The spline is
then approximated by a degree-5 polynomial via least-squares on a dense grid,
which is the required output interface.
"""
import numpy as np
import time
from scipy.interpolate import CubicSpline

def construct(training_x, training_subgrid_tendency):
    """
    Fit a piecewise cubic spline to the unresolved fast-scale forcing,
    then approximate the spline with a degree-5 polynomial for the output interface.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
        
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove any potential NaNs/Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        # Too few points for a meaningful spline fit
        # Fall back to a simple polynomial fit
        try:
            coeffs = np.polyfit(x, y, 5)
            # polyfit returns descending powers, we need ascending
            return coeffs[::-1]
        except:
            return np.zeros(6)

    # Sort by x for spline construction
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Remove duplicate x values (spline requires strictly increasing x)
    unique_mask = np.concatenate([True, np.diff(x_sorted) > 1e-12])
    x_unique = x_sorted[unique_mask]
    y_unique = y_sorted[unique_mask]
    
    if len(x_unique) < 10:
        try:
            coeffs = np.polyfit(x_unique, y_unique, 5)
            return coeffs[::-1]
        except:
            return np.zeros(6)

    # Determine knot locations using quantiles
    # Use a reasonable number of knots, e.g., 10-20 quantiles
    n_knots = min(15, len(x_unique) // 3)
    if n_knots < 5:
        n_knots = 5
    
    # Generate knot positions at quantiles of the training data
    quantiles = np.linspace(0, 1, n_knots + 1)[1:-1]  # Exclude endpoints
    knot_positions = np.quantile(x_unique, quantiles)
    
    # Ensure knots are unique and within the data range
    knot_positions = np.unique(knot_positions)
    if len(knot_positions) < 4:
        # Fallback: use evenly spaced knots
        knot_positions = np.linspace(x_unique[0], x_unique[-1], 10)[1:-1]
        knot_positions = np.unique(knot_positions)
    
    # Include endpoints in the knot array for the spline
    all_knots = np.concatenate([[x_unique[0]], knot_positions, [x_unique[-1]]])
    
    # Construct the cubic spline with natural boundary conditions
    try:
        cs = CubicSpline(all_knots, y_unique, bc_type='natural')
    except Exception:
        # Fallback to polynomial fit if spline construction fails
        try:
            coeffs = np.polyfit(x_unique, y_unique, 5)
            return coeffs[::-1]
        except:
            return np.zeros(6)

    # Check time
    if time.time() - start_time > DEADLINE:
        # Return best found so far - fallback to polynomial
        try:
            coeffs = np.polyfit(x_unique, y_unique, 5)
            return coeffs[::-1]
        except:
            return np.zeros(6)

    # Now approximate the spline with a degree-5 polynomial
    # Evaluate the spline on a dense grid and fit a polynomial
    # Use a grid that covers the range of x
    n_grid = 500
    x_grid = np.linspace(x_unique[0], x_unique[-1], n_grid)
    y_grid = cs(x_grid)
    
    # Fit a degree-5 polynomial to the spline values
    try:
        # Use polyfit which returns coefficients in descending powers
        coeffs_desc = np.polyfit(x_grid, y_grid, 5)
        # Convert to ascending powers
        coeffs_asc = coeffs_desc[::-1]
    except Exception:
        # Fallback
        try:
            coeffs = np.polyfit(x_unique, y_unique, 5)
            return coeffs[::-1]
        except:
            return np.zeros(6)

    # Final sanity check: ensure coefficients are finite
    if not np.all(np.isfinite(coeffs_asc)):
        # Fallback to simple polynomial fit
        try:
            coeffs = np.polyfit(x_unique, y_unique, 5)
            coeffs_asc = coeffs[::-1]
        except:
            coeffs_asc = np.zeros(6)

    return coeffs_asc
