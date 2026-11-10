# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_area(positions, n=64, n_heights=20000):
    """Compute the union area via quadrature as specified."""
    x = np.asarray(positions, dtype=np.float64)
    if x.shape != (n,):
        return np.inf
    if not np.all(np.isfinite(x)):
        return np.inf
    
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    
    # L_i = x_i + a_i * y for each triangle and height
    # Shape: (n_heights, n)
    L = x[None, :] + a[None, :] * y[:, None]
    w = (1.0 - y) / n  # Shape: (n_heights,)
    
    # Sort L along axis 1
    L_sorted = np.sort(L, axis=1)
    
    # Union length = w + sum of min(gap, w) for consecutive pairs
    # gaps = L_sorted[:, 1:] - L_sorted[:, :-1], shape (n_heights, n-1)
    gaps = L_sorted[:, 1:] - L_sorted[:, :-1]
    # min(gaps, w[:, None])
    overlaps = np.minimum(gaps, w[:, None])
    union_lengths = w[:, None] + overlaps.sum(axis=1)
    
    area = np.mean(union_lengths)
    if area <= 0:
        return np.inf
    return area


def _objective(positions):
    """Objective to minimize: the area."""
    return _compute_area(positions)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses scipy.optimize.minimize with SLSQP from multiple starting points
    (Keich baseline + random perturbations) to find a locally optimal
    configuration that minimizes the union area.
    """
    n = 64
    k = int(np.log2(n))
    
    # Generate Keich baseline
    j = np.arange(1, k + 1)
    keich = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    best_positions = keich.copy()
    best_area = _compute_area(keich)
    
    # Try multiple random restarts with SLSQP
    rng = np.random.default_rng(42)
    
    # Starting points: Keich + various perturbations
    start_points = [keich.copy()]
    
    # Add perturbed versions
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
        perturbed = keich + rng.normal(0, scale, size=n)
        start_points.append(perturbed)
    
    # Also try purely random starts
    for _ in range(5):
        start_points.append(rng.uniform(0, 1, size=n))
    
    for start in start_points:
        result = minimize(
            _objective,
            start,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if result.success and np.all(np.isfinite(result.x)):
            area = _compute_area(result.x)
            if area < best_area:
                best_area = area
                best_positions = result.x.copy()
    
    return best_positions.astype(np.float64)


# EVOLVE-BLOCK-END
