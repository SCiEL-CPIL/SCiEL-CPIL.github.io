# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _compute_volume(positions: np.ndarray) -> float:
    """
    Compute the Monte Carlo union volume for a given set of 64 base positions.
    
    Parameters:
        positions: array of shape (64, 2) with base positions (x_k, y_k)
    
    Returns:
        V: estimated union volume
    """
    N = 8
    a = positions[:, 0]  # x_k
    b = positions[:, 1]  # y_k
    
    i_idx = np.arange(64) // N  # i for each tube
    j_idx = np.arange(64) % N   # j for each tube
    
    # Box bounds
    X0 = np.min(a)
    X1 = np.max(a + (i_idx + 1) / N)
    Y0 = np.min(b)
    Y1 = np.max(b + (j_idx + 1) / N)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return np.inf
    
    # Monte Carlo estimate
    rng = np.random.default_rng(12345)
    mc_a = rng.uniform(X0, X1, 500000)
    mc_b = rng.uniform(Y0, Y1, 500000)
    mc_c = rng.uniform(0, 1, 500000)
    
    # Check if each point is inside at least one tube
    # Tube k at height c occupies:
    #   x in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
    #   y in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    
    # For efficiency, vectorize over tubes
    # x_k + c*i_k/N for all k, c
    # Shape: (64, 500000)
    x_left = a[:, None] + mc_c[None, :] * i_idx[:, None] / N
    x_right = x_left + 1.0 / N
    y_left = b[:, None] + mc_c[None, :] * j_idx[:, None] / N
    y_right = y_left + 1.0 / N
    
    # Check membership: mc_a in [x_left, x_right] AND mc_b in [y_left, y_right]
    # Shape: (64, 500000)
    in_x = (mc_a[None, :] >= x_left) & (mc_a[None, :] <= x_right)
    in_y = (mc_b[None, :] >= y_left) & (mc_b[None, :] <= y_right)
    in_tube = in_x & in_y  # (64, 500000)
    
    # Point is in union if in at least one tube
    in_union = np.any(in_tube, axis=0)  # (500000,)
    
    fraction = np.mean(in_union)
    V = fraction * box_area
    
    if V <= 0:
        return np.inf
    
    return V


def _objective(x: np.ndarray) -> float:
    """
    Objective function for the optimizer: minimize the union volume.
    x is a 128-dimensional vector: first 64 elements are x_k, next 64 are y_k.
    """
    positions = x.reshape(64, 2)
    V = _compute_volume(positions)
    if not np.isfinite(V) or V <= 0:
        return 1e10
    return V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses Differential Evolution to optimize all 128 base position coordinates
    simultaneously, directly minimizing the 3D Monte Carlo union volume.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order.
    """
    N = 8
    
    # Define bounds for each coordinate
    # We need reasonable bounds. Let's think about what range makes sense.
    # The Keich construction gives values in [0, 1) roughly.
    # Let's use a broader range to allow the optimizer to explore.
    # Based on the structure, x_k should be in some range that allows
    # the sheared tubes to overlap well.
    
    # Let's use bounds [-1, 2] for each coordinate to give the optimizer room.
    # Actually, let's think more carefully. The box [X0, X1] x [Y0, Y1] 
    # is determined by the positions. If all positions are close together,
    # the box is small and V is small. But the shearing means different tubes
    # move in different directions, so we need spread.
    
    # Let's use a reasonable initial range. The Keich construction gives
    # values in [0, ~0.5]. Let's use [-0.5, 1.5] as bounds.
    
    bounds = [(-0.5, 1.5)] * 128
    
    # Run differential evolution
    # Use a reasonable number of generations and population
    # The objective is expensive (500k MC samples per eval), so we need to be careful.
    
    # Let's use a smaller population and fewer generations to stay within time limits.
    # Each eval takes ~0.1-0.5 seconds, so 1000 evals would take 100-500 seconds.
    
    result = differential_evolution(
        _objective,
        bounds,
        maxiter=50,
        popsize=15,
        tol=1e-4,
        seed=42,
        polish=False,
        init='latinhypercube'
    )
    
    positions = result.x.reshape(64, 2)
    
    # Verify the result is valid
    V = _compute_volume(positions)
    if not np.isfinite(V) or V <= 0:
        # Fallback to Keich construction
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
