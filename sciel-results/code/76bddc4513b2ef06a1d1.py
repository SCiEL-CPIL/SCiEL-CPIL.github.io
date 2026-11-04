import numpy as np
from scipy.optimize import differential_evolution

N = 14

def _score_from_angles(angles):
    """Compute min triangle area / convex hull area for points on unit circle at given angles."""
    angles = np.sort(angles) % (2 * np.pi)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Convex hull area (for points on circle, hull is the polygon in angular order)
    # Shoelace formula
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    
    if area < 1e-12:
        return 0.0
    
    # Minimum triangle area over all triples
    min_area = float('inf')
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Triangle area via cross product
                ax = pts[i, 0]
                ay = pts[i, 1]
                bx = pts[j, 0] - ax
                by = pts[j, 1] - ay
                cx = pts[k, 0] - ax
                cy = pts[k, 1] - ay
                tri_area = 0.5 * abs(bx * cy - by * cx)
                if tri_area < min_area:
                    min_area = tri_area
    
    return min_area / area


def _fitness(x):
    """Negative score for minimization. x is N angles in [0, 2*pi)."""
    angles = np.mod(x, 2 * np.pi)
    # Ensure minimum separation to avoid degenerate triangles
    sorted_a = np.sort(angles)
    gaps = np.diff(sorted_a)
    if len(gaps) > 0:
        min_gap = np.min(gaps)
    else:
        min_gap = 2 * np.pi
    # Wrap-around gap
    if len(sorted_a) > 1:
        gaps = np.append(gaps, 2 * np.pi - sorted_a[-1] + sorted_a[0])
        min_gap = np.min(gaps)
    
    if min_gap < 1e-6:
        return 0.0  # Penalize degeneracy
    
    s = _score_from_angles(angles)
    return -s


def construct():
    """Place 14 points on a circle to maximize min triangle area / convex hull area."""
    
    # Bounds: each angle in [0, 2*pi)
    bounds = [(0, 2 * np.pi) for _ in range(N)]
    
    # Multiple random starts for differential evolution
    best_score = -float('inf')
    best_x = None
    
    n_starts = 3
    for start_idx in range(n_starts):
        if start_idx == 0:
            # Uniform start
            x0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
        else:
            x0 = np.random.uniform(0, 2 * np.pi, N)
        
        # Use differential evolution with modest population
        de_result = differential_evolution(
            _fitness,
            bounds,
            x0=x0,
            seed=start_idx * 42 + 7,
            maxiter=80,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 0.9),
            recombination=0.7,
            init='sobol' if start_idx > 0 else 'latinhypercube',
            polish=True
        )
        
        if de_result.fun < 0:  # valid (since we negate)
            s = -de_result.fun
            if s > best_score:
                best_score = s
                best_x = de_result.x
    
    if best_x is None:
        # Fallback: uniform
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    else:
        angles = np.mod(best_x, 2 * np.pi)
    
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts
