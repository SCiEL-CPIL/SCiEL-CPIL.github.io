"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(positions):
    """Compute the packing radius for given positions."""
    if positions.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    # Check all points are inside unit circle
    dists = np.linalg.norm(positions, axis=1)
    if np.any(dists >= 1.0):
        return 0.0
    # Min distance to boundary
    min_boundary = 1.0 - np.max(dists)
    if min_boundary <= 0:
        return 0.0
    # Min pairwise distance
    # Compute all pairwise distances
    n = positions.shape[0]
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists_matrix, np.inf)
    min_pairwise = np.min(dists_matrix)
    if min_pairwise <= 0:
        return 0.0
    return min(min_boundary, min_pairwise / 2.0)


def _objective(positions, target_r):
    """Objective function: minimize overlap violations and boundary violations."""
    positions = positions.reshape(N, 2)
    # Distance to boundary
    dists = np.linalg.norm(positions, axis=1)
    boundary_violations = np.maximum(0.0, dists - (1.0 - target_r))
    # Pairwise violations
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists_matrix, np.inf)
    pairwise_violations = np.maximum(0.0, 2.0 * target_r - dists_matrix)
    # Sum of squared violations
    obj = np.sum(boundary_violations ** 2) + np.sum(pairwise_violations ** 2)
    return obj


def _ring_init(n, r0=0.5):
    """Initialize points in concentric rings."""
    points = np.zeros((n, 2))
    # Place points in rings
    ring_radii = [0.0]
    ring_counts = [1]
    remaining = n - 1
    r = 0.3
    while remaining > 0:
        circumference = 2 * np.pi * r
        count = max(1, int(circumference / (2 * r0 * 1.1)))
        if remaining <= count:
            count = remaining
        ring_radii.append(r)
        ring_counts.append(count)
        remaining -= count
        r += 0.25
    idx = 0
    for ri, rc in zip(ring_radii, ring_counts):
        if rc == 1 and ri == 0.0:
            points[idx] = [0.0, 0.0]
            idx += 1
        else:
            angles = np.linspace(0, 2 * np.pi, rc, endpoint=False)
            points[idx:idx+rc, 0] = ri * np.cos(angles)
            points[idx:idx+rc, 1] = ri * np.sin(angles)
            idx += rc
    return points[:n]


def _random_init(n):
    """Random initialization inside unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _hex_lattice_init(n):
    """Hexagonal lattice initialization."""
    spacing = 0.25
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.sqrt(x**2 + y**2) <= 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= n:
        # Select n points closest to center
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists, kind='stable')[:n]
        return points[indices]
    else:
        return _random_init(n)


def construct():
    """Construct the circle packing."""
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    best_positions = None
    best_score = 0.0
    
    # Try multiple initializations
    inits = [
        _ring_init(N),
        _hex_lattice_init(N),
        _random_init(N),
    ]
    
    # Add a few more random inits
    for _ in range(3):
        inits.append(_random_init(N))
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Try to find good radius via bisection on target_r
        # First, get a rough estimate from the initial configuration
        rough_score = _score(init_pos)
        
        # Bisection search for optimal radius
        lo, hi = 0.0, 0.5
        best_r = 0.0
        best_pos_for_r = init_pos.copy()
        
        # Quick bisection with limited iterations
        for bisect_iter in range(15):
            if time.time() > deadline:
                break
            mid_r = (lo + hi) / 2.0
            # Try to optimize with target_r = mid_r
            x0 = init_pos.reshape(-1).copy()
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(mid_r,),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                optimized_pos = result.x.reshape(N, 2)
                # Check if feasible
                if _objective(optimized_pos, mid_r) < 1e-6:
                    # Feasible at this radius
                    actual_score = _score(optimized_pos)
                    if actual_score > best_score:
                        best_score = actual_score
                        best_positions = optimized_pos.copy()
                    best_r = mid_r
                    best_pos_for_r = optimized_pos.copy()
                    lo = mid_r
                else:
                    hi = mid_r
            except Exception:
                hi = mid_r
        
        # Refine around best_r found
        if best_r > 0:
            for refine_iter in range(5):
                if time.time() > deadline:
                    break
                # Try slightly larger radius
                test_r = best_r * 1.01
                x0 = best_pos_for_r.reshape(-1).copy()
                try:
                    result = minimize(
                        _objective,
                        x0,
                        args=(test_r,),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    optimized_pos = result.x.reshape(N, 2)
                    if _objective(optimized_pos, test_r) < 1e-8:
                        actual_score = _score(optimized_pos)
                        if actual_score > best_score:
                            best_score = actual_score
                            best_positions = optimized_pos.copy()
                        best_r = test_r
                        best_pos_for_r = optimized_pos.copy()
                    else:
                        break
                except Exception:
                    break
    
    # If we found a good solution, do final refinement
    if best_positions is not None and best_score > 0:
        # Final polish: maximize the actual score directly
        # Use the current best as starting point
        x0 = best_positions.reshape(-1).copy()
        
        # We'll do a few more optimization steps with increasing target radii
        current_r = best_score
        for polish_iter in range(10):
            if time.time() > deadline:
                break
            test_r = current_r * 1.005
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(test_r,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                optimized_pos = result.x.reshape(N, 2)
                if _objective(optimized_pos, test_r) < 1e-8:
                    actual_score = _score(optimized_pos)
                    if actual_score > best_score:
                        best_score = actual_score
                        best_positions = optimized_pos.copy()
                        x0 = optimized_pos.reshape(-1).copy()
                        current_r = test_r
                    else:
                        break
                else:
                    break
            except Exception:
                break
    
    # Return the best solution found
    if best_positions is not None:
        return best_positions
    else:
        # Fallback: return a valid but simple configuration
        return _hex_lattice_init(N)

# EVOLVE-BLOCK-END
