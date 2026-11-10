# EVOLVE-BLOCK-START
import numpy as np
import time


def _area_fast(x, num_points=20000):
    """Compute union area of 64 triangles using the specified quadrature."""
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(num_points) + 0.5) / num_points
    # L[i, m] = x[i] + a[i] * y[m]
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    w = (1.0 - y) / n
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    clipped = np.minimum(diffs, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(clipped, axis=0)
    return np.mean(union_lengths)


def _subgradient_step(x, i, num_points=5000):
    """
    Compute subgradient of area w.r.t. x[i] and find optimal x[i].
    Returns the best x[i] value that minimizes area.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(num_points) + 0.5) / num_points
    w = (1.0 - y) / n
    
    # L for all triangles at each height
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # For coordinate i, L_i(y) = x[i] + a[i]*y
    # We need to find how changing x[i] affects the union length
    
    # The subgradient w.r.t. x[i] at each height y is:
    # For each height, the contribution to union length from interval i depends on
    # whether the interval [L_i, L_i+w] is in the "gap" between sorted intervals
    # or overlaps with neighbors.
    
    # Simpler approach: just do a 1D scan over a range of x[i] values
    # and pick the one that gives minimum area.
    
    # Estimate range: x[i] should be near its current value
    # The triangle spans from x[i] to x[i]+a[i], so x[i] in roughly [-1, 1]
    # but let's be more conservative and search around current value
    
    x_i_current = x[i]
    # Search range: current +/- 0.5 should cover most cases
    search_range = np.linspace(x_i_current - 0.5, x_i_current + 0.5, 200)
    
    best_xi = x_i_current
    best_area = np.inf
    
    for xi in search_range:
        x_temp = x.copy()
        x_temp[i] = xi
        area = _area_fast(x_temp, num_points=num_points)
        if area < best_area:
            best_area = area
            best_xi = xi
    
    return best_xi, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses Keich construction as initial guess, then refines with coordinate descent.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    # Step 1: Keich construction (baseline)
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    # Evaluate baseline area
    best_area = _area_fast(x, num_points=20000)
    best_x = x.copy()
    
    # Step 2: Coordinate descent optimization
    # Use a coarser quadrature for speed during optimization
    opt_points = 2000
    
    for iteration in range(20):  # Max 20 passes
        if time.time() > deadline - 30:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 30:
                break
            
            # Try to improve x[i]
            # Use a finer search around current value
            x_i_current = x[i]
            
            # Adaptive search range
            search_range = np.linspace(x_i_current - 0.3, x_i_current + 0.3, 100)
            
            best_xi = x_i_current
            best_local_area = _area_fast(x, num_points=opt_points)
            
            for xi in search_range:
                x_temp = x.copy()
                x_temp[i] = xi
                area = _area_fast(x_temp, num_points=opt_points)
                if area < best_local_area - 1e-12:
                    best_local_area = area
                    best_xi = xi
            
            if best_xi != x_i_current:
                x[i] = best_xi
                improved = True
        
        # Check full area after each pass
        full_area = _area_fast(x, num_points=20000)
        if full_area < best_area:
            best_area = full_area
            best_x = x.copy()
        
        if not improved:
            break
    
    # Step 3: Final fine-tuning with full quadrature
    if time.time() < deadline - 10:
        for i in range(n):
            if time.time() > deadline - 10:
                break
            x_i_current = best_x[i]
            search_range = np.linspace(x_i_current - 0.05, x_i_current + 0.05, 50)
            best_xi = x_i_current
            best_local_area = _area_fast(best_x, num_points=20000)
            
            for xi in search_range:
                x_temp = best_x.copy()
                x_temp[i] = xi
                area = _area_fast(x_temp, num_points=20000)
                if area < best_local_area - 1e-14:
                    best_local_area = area
                    best_xi = xi
            
            if best_xi != x_i_current:
                best_x[i] = best_xi
    
    # Ensure validity
    if not np.all(np.isfinite(best_x)) or best_x.size != 64:
        return np.array(positions, dtype=np.float64)
    
    return best_x


# EVOLVE-BLOCK-END
