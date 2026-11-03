"""Improved program for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _validate(arr):
    """Check if the array of circles is valid: disjoint interiors, within rectangle perimeter <= 4."""
    if len(arr) != N:
        return False
    # Compute bounding box
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    
    if np.any(r <= 0):
        return False
    
    # Bounding box of the circles
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    w = x_max - x_min
    h = y_max - y_min
    
    # Perimeter constraint: 2*(w + h) <= 4
    if 2 * (w + h) > 4 + 1e-9:
        return False
    
    # Check disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return False
    
    return True

def _objective(params, w_rect, h_rect):
    """
    Objective function for optimization.
    params: 3*N array [x1, y1, r1, x2, y2, r2, ...]
    We want to maximize sum of radii, so minimize -sum(r).
    Add penalties for violations.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Ensure positive radii
    r = np.maximum(r, 1e-6)
    
    # Compute actual bounding box
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    w = x_max - x_min
    h = y_max - y_min
    
    # Perimeter penalty
    perim = 2 * (w + h)
    perim_penalty = max(0, perim - 4) * 1000
    
    # Overlap penalty
    overlap_penalty = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_penalty += (min_dist - dist) ** 2
    
    # Keep circles within a reasonable box to avoid drift
    # The rectangle dimensions are variable, but we can constrain the center of mass
    # to be near origin and scale appropriately
    
    # We want to maximize sum(r), so objective = -sum(r) + penalties
    obj = -np.sum(r) + perim_penalty + overlap_penalty * 100
    
    return obj

def _hex_initial(w, h):
    """Generate hexagonal lattice initial positions."""
    points = []
    # Hexagonal packing
    dx = 1.0  # will be scaled
    dy = dx * np.sqrt(3) / 2
    
    # Generate a grid of hexagonal points
    n_cols = 8
    n_rows = 6
    for i in range(n_cols):
        for j in range(n_rows):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            points.append([x, y])
    
    points = np.array(points)
    # Scale to fit in rectangle
    points[:, 0] = (points[:, 0] - points[:, 0].mean()) * (w * 0.9) / (points[:, 0].ptp() + 1e-9) + w/2
    points[:, 1] = (points[:, 1] - points[:, 1].mean()) * (h * 0.9) / (points[:, 1].ptp() + 1e-9) + h/2
    
    # Select N points closest to center
    center = np.array([w/2, h/2])
    dists = np.sum((points - center)**2, axis=1)
    idx = np.argsort(dists)[:N]
    
    return points[idx]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    best_score = 0
    best_result = None
    
    # Try different aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # For perimeter 4: 2*(w + h) = 4, so w + h = 2
        # h = ar * w, so w + ar*w = 2, w = 2/(1+ar), h = 2*ar/(1+ar)
        w_rect = 2.0 / (1 + ar)
        h_rect = 2.0 * ar / (1 + ar)
        
        # Initial guess: hexagonal lattice
        init_points = _hex_initial(w_rect, h_rect)
        
        # Initial radius: small
        r_init = 0.05 * np.ones(N)
        
        # Build initial parameter vector
        x0 = np.zeros(3 * N)
        x0[0::3] = init_points[:, 0]
        x0[1::3] = init_points[:, 1]
        x0[2::3] = r_init
        
        # Use differential evolution for global optimization
        bounds = []
        for i in range(N):
            bounds.append((0, w_rect))  # x
            bounds.append((0, h_rect))  # y
            bounds.append((0.001, 1.0))  # r
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(w_rect, h_rect),
                maxiter=100,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                timeout=20
            )
            
            if result.success or result.fun < 0:
                params = result.x
                x = params[0::3]
                y = params[1::3]
                r = params[2::3]
                
                # Normalize: center the configuration
                cx = (np.min(x - r) + np.max(x + r)) / 2
                cy = (np.min(y - r) + np.max(y + r)) / 2
                x = x - cx + w_rect/2
                y = y - cy + h_rect/2
                
                arr = np.column_stack([x, y, r])
                
                if _validate(arr):
                    score = np.sum(arr[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = arr
        
        except Exception:
            pass
        
        # Also try local refinement with L-BFGS-B
        if time.time() > deadline:
            break
        
        try:
            # Local refinement from the DE result or initial
            if result.fun < 0:
                x0_local = result.x
            else:
                x0_local = x0
            
            res_local = minimize(
                _objective,
                x0_local,
                args=(w_rect, h_rect),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res_local.fun < 0:
                params = res_local.x
                x = params[0::3]
                y = params[1::3]
                r = params[2::3]
                
                cx = (np.min(x - r) + np.max(x + r)) / 2
                cy = (np.min(y - r) + np.max(y + r)) / 2
                x = x - cx + w_rect/2
                y = y - cy + h_rect/2
                
                arr = np.column_stack([x, y, r])
                
                if _validate(arr):
                    score = np.sum(arr[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = arr
        
        except Exception:
            pass
    
    # If no valid result found, return a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
