# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in unit square.
    Uses differential evolution to find a high-quality packing.
    """
    n = 10
    tol = 1e-7

    def objective(params):
        # params: [s0, cx0, cy0, s1, cx1, cy1, ..., s9, cx9, cy9]
        s = params[0::3]
        cx = params[1::3]
        cy = params[2::3]
        
        # Boundary penalty: squares must be in [0,1]^2
        penalty = 0.0
        for i in range(n):
            if s[i] <= 0:
                penalty += 1e6
            # Check boundaries
            if cx[i] - s[i]/2 < -tol:
                penalty += 1e6 * (abs(cx[i] - s[i]/2 + tol) + 1)
            if cx[i] + s[i]/2 > 1 + tol:
                penalty += 1e6 * (abs(cx[i] + s[i]/2 - 1 - tol) + 1)
            if cy[i] - s[i]/2 < -tol:
                penalty += 1e6 * (abs(cy[i] - s[i]/2 + tol) + 1)
            if cy[i] + s[i]/2 > 1 + tol:
                penalty += 1e6 * (abs(cy[i] + s[i]/2 - 1 - tol) + 1)
        
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    # Overlapping - penalty proportional to overlap
                    overlap_x = t - dx
                    overlap_y = t - dy
                    penalty += 1e6 * (overlap_x * overlap_y + overlap_x + overlap_y)
        
        # We want to MAXIMIZE sum(s), so minimize -sum(s) + penalty
        return -np.sum(s) + penalty

    # Initial guess: mixed sizes
    # One large square, rest smaller
    s_init = np.array([0.45] + [0.25]*9)
    # Place large square bottom-left
    cx_init = np.array([0.225] + [0.5]*9)
    cy_init = np.array([0.225] + [0.5]*9)
    
    # Spread out the smaller squares
    positions = [
        (0.7, 0.2), (0.7, 0.7), (0.2, 0.7),
        (0.4, 0.4), (0.6, 0.4), (0.4, 0.6),
        (0.8, 0.4), (0.4, 0.8), (0.8, 0.8)
    ]
    for k, (px, py) in enumerate(positions):
        cx_init[k+1] = px
        cy_init[k+1] = py
    
    x0 = np.concatenate([s_init, cx_init, cy_init])
    
    # Bounds: s in (0, 1), cx in (0, 1), cy in (0, 1)
    bounds = []
    for i in range(n):
        bounds.append((1e-4, 1.0))   # side
        bounds.append((0.0, 1.0))     # cx
        bounds.append((0.0, 1.0))     # cy
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=200,
        popsize=15,
        tol=1e-8,
        mutation=(0.3, 1.0),
        recombination=0.7,
        polish=True,
        init='sobol',
    )
    
    params = result.x
    s = params[0::3]
    cx = params[1::3]
    cy = params[2::3]
    
    # Ensure feasibility: clip and adjust
    for i in range(n):
        s[i] = max(s[i], 1e-4)
        # Clip centers to keep squares inside
        half = s[i] / 2
        cx[i] = np.clip(cx[i], half, 1.0 - half)
        cy[i] = np.clip(cy[i], half, 1.0 - half)
    
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides


# EVOLVE-BLOCK-END
