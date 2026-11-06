"""Improved packing: hexagonal lattice with optimized spacing + local refinement."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _score(centers):
    """Compute the minimum radius (score) for a set of centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    # Boundary constraint
    min_boundary = min(0.5 - np.max(np.abs(centers), axis=1))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists) / 2.0
    return min(min_boundary, min_dist)

def _hex_packing(rows, spacing, offset_y, shift_pattern):
    """Generate centers for a hexagonal-like packing.
    rows: list of (count, x_start_offset) or just counts
    spacing: horizontal/vertical spacing parameter
    offset_y: vertical offset of the whole pattern
    shift_pattern: how much to shift alternating rows
    """
    centers = []
    for i, count in enumerate(rows):
        # Each row is shifted horizontally for hex packing
        x_shift = (i % 2) * spacing * 0.5 if shift_pattern == 'alternate' else 0.0
        # Distribute 'count' circles evenly in a row
        if count > 1:
            x_positions = np.linspace(-0.5 + spacing/2, 0.5 - spacing/2, count)
        else:
            x_positions = np.array([0.0])
        x_positions += x_shift
        y_pos = offset_y + i * spacing * np.sqrt(3) / 2
        for x in x_positions:
            centers.append([x, y_pos])
    return np.array(centers)

def _build_from_rows(rows, params):
    """Build centers from row structure with given parameters."""
    spacing = params[0]
    offset_y = params[1]
    # Compute total height
    n_rows = len(rows)
    total_height = (n_rows - 1) * spacing * np.sqrt(3) / 2
    # Center vertically
    y_base = -total_height / 2 + offset_y
    centers = []
    for i, count in enumerate(rows):
        if count > 1:
            # Spread circles across available width, accounting for spacing
            # Max width for count circles with spacing s: (count-1)*s
            max_width = (count - 1) * spacing
            # Center the row
            x_start = -max_width / 2
            x_positions = x_start + np.arange(count) * spacing
        else:
            x_positions = np.array([0.0])
        
        # Alternate shift for hex packing
        if i % 2 == 1 and count > 1:
            x_positions += spacing * 0.5
        
        y_pos = y_base + i * spacing * np.sqrt(3) / 2
        for x in x_positions:
            centers.append([x, y_pos])
    return np.array(centers)

def _local_refinement(centers, max_iter=2000, deadline=None):
    """Local refinement using gradient-free optimization on flattened centers."""
    n = len(centers)
    
    def objective(x):
        c = x.reshape(-1, 2)
        # Boundary
        min_b = 0.5 - np.max(np.abs(c), axis=1)
        min_b = min_b.min()
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(d, 1e10)
        min_d = d.min() / 2.0
        return -min(min_b, min_d)
    
    x0 = centers.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14})
    return result.x.reshape(-1, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    # Try several row configurations
    # 50 circles: various row splits
    row_configs = [
        [7, 6, 7, 6, 7, 6, 7, 4],      # 50
        [8, 7, 7, 7, 7, 7, 7],           # 50
        [6, 7, 7, 7, 7, 7, 7, 2],        # 50
        [5, 6, 7, 7, 7, 7, 7, 4],        # 50
        [4, 7, 7, 7, 7, 7, 7, 4],        # 50
        [7, 7, 7, 7, 7, 7, 8],           # 50
        [6, 6, 7, 7, 7, 7, 7, 3],        # 50
        [5, 7, 7, 7, 7, 7, 7, 3],        # 50
        [8, 6, 7, 7, 7, 7, 8],           # 50
        [7, 6, 7, 7, 7, 6, 7, 3],        # 50
        [6, 7, 6, 7, 7, 7, 6, 4],        # 50
        [7, 7, 6, 7, 7, 6, 7, 3],        # 50
    ]
    
    best_centers = None
    best_score = 0.0
    
    for rows in row_configs:
        if sum(rows) != N:
            continue
        if time.time() > deadline:
            break
        
        # Initial spacing estimate
        n_rows = len(rows)
        max_count = max(rows)
        # Spacing limited by width and height
        s_w = 1.0 / max_count
        s_h = 1.0 / (n_rows * np.sqrt(3) / 2)
        s_init = min(s_w, s_h) * 0.9
        
        # Use differential evolution to optimize spacing and offset
        bounds = [(0.05, 0.15), (-0.1, 0.1)]
        
        def de_objective(params):
            centers = _build_from_rows(rows, params)
            # Check validity
            if np.any(np.abs(centers) > 0.5 + 1e-9):
                return 1e10
            return -_score(centers)
        
        try:
            de_result = differential_evolution(
                de_objective, bounds, 
                maxiter=100, popsize=15, tol=1e-8,
                seed=42, polish=False, init='latinhypercube'
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        best_params = de_result.x
        centers = _build_from_rows(rows, best_params)
        
        # Local refinement
        try:
            centers = _local_refinement(centers, max_iter=1000)
        except Exception:
            pass
        
        # Check validity and score
        if np.any(~np.isfinite(centers)):
            continue
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            continue
        
        s = _score(centers)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
    
    # If no valid result, fall back to grid
    if best_centers is None or best_score <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Final check
    if time.time() > deadline:
        return best_centers
    
    # One more refinement pass if time allows
    try:
        refined = _local_refinement(best_centers, max_iter=3000)
        if np.all(np.isfinite(refined)) and np.all(np.abs(refined) <= 0.5 + 1e-9):
            s2 = _score(refined)
            if s2 > best_score:
                best_centers = refined
    except Exception:
        pass
    
    return best_centers
# EVOLVE-BLOCK-END
