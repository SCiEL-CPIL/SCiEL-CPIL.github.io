"""Improved packing: hexagonal arrangement + local optimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    # Try multiple row configurations and pick the best
    best_cents = None
    best_r = 0
    
    # Configuration: rows with 10,10,10,10,10 (5 rows of 10)
    # Hexagonal: alternate rows offset
    for n_rows in [5, 6, 7, 8]:
        for offset_pattern in [0, 1]:
            # Distribute N circles into n_rows
            base = N // n_rows
            extra = N % n_rows
            row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
            
            # Calculate spacing
            # For hexagonal: horizontal spacing = 2r, vertical spacing = 2r * sqrt(3)/2
            # We need to fit in [-0.5, 0.5]^2
            # Try to find r such that circles fit
            
            # Use a parameter s = spacing factor
            # Horizontal: (max_row_count - 1) * 2r + 2r <= 1  => r <= 1/(2*max_row_count)
            # Vertical: (n_rows - 1) * 2r*sqrt(3)/2 + 2r <= 1 => r <= 1/(2 + (n_rows-1)*sqrt(3))
            
            max_rc = max(row_counts)
            r_horiz = 1.0 / (2.0 * max_rc)
            r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
            r = min(r_horiz, r_vert)
            
            if r <= best_r:
                continue
            
            # Build positions
            centers = []
            total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
            y_start = -0.5 + r + total_y / 2.0
            
            for row_idx in range(n_rows):
                rc = row_counts[row_idx]
                y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
                x_span = (rc - 1) * 2 * r
                x_start = -0.5 + r + x_span / 2.0
                offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
                for col in range(rc):
                    x = x_start + col * 2 * r + offset
                    centers.append([x, y])
            
            centers = np.array(centers[:N])
            r_actual = _score(centers)
            
            if r_actual > best_r:
                best_r = r_actual
                best_cents = centers.copy()
    
    if best_cents is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)
    
    return best_cents, best_r

def _local_optimize(cents, time_limit=100.0):
    """Local optimization using gradient-free method."""
    start_time = time.time()
    best_cents = cents.copy()
    best_r = _score(best_cents)
    
    n = len(cents)
    
    # Use Nelder-Mead style or simple perturbation
    # Try random perturbations with acceptance
    rng = np.random.default_rng(42)
    step_size = 0.005
    
    while time.time() - start_time < time_limit:
        # Generate perturbation
        perturbation = rng.normal(0, step_size, size=cents.shape)
        new_cents = best_cents + perturbation
        
        # Check if all points are still in the square (with margin)
        if np.any(np.abs(new_cents) > 0.5 - 1e-9):
            continue
        
        new_r = _score(new_cents)
        if new_r > best_r:
            best_r = new_r
            best_cents = new_cents.copy()
            step_size = max(step_size * 0.95, 0.001)
        else:
            step_size = max(step_size * 0.98, 0.001)
    
    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    # Step 1: Create hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # Step 2: Local optimization
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        cents, r1 = _local_optimize(best_cents, time_limit=min(remaining - 5, 100))
        if r1 > best_r:
            best_r = r1
            best_cents = cents
    
    return best_cents
# EVOLVE-BLOCK-END
