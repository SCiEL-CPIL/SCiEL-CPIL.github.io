"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

DEADLINE = 150.0  # seconds
START_TIME = None


def _score(centers):
    """Compute the radius: min of (half min pair distance, min wall distance)."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(centers)):
        return -1.0
    # Wall distances
    wall = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    wall_min = wall.min()
    # Pair distances
    d = centers[:, 0:1] - centers[0:1, 0]
    # Use vectorized pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    pair_min = dist.min() / 2.0
    return min(wall_min, pair_min)


def _hexagonal_seed(spacing, v_offset, h_offset):
    """
    Create a hexagonal lattice packing of N circles.
    spacing: horizontal spacing between centers in same row
    v_offset: vertical spacing between rows
    h_offset: horizontal offset of alternating rows
    Returns centers as (N, 2) array, or None if invalid.
    """
    # Hexagonal packing: rows alternate offset
    # Row 0: x = -spacing*(m-1)/2 + i*spacing
    # Row 1: x = -spacing*(m-1)/2 + h_offset + i*spacing
    # Try different row counts
    best_centers = None
    best_score = -1.0

    # Determine number of rows
    # For hex packing, alternate row lengths
    # Total N circles, distribute among rows
    # Try row counts from 5 to 10
    for n_rows in range(4, 12):
        # Distribute N circles among n_rows
        base = N // n_rows
        extra = N % n_rows
        row_lengths = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        # Hexagonal: alternate rows have offset
        # Row i has length row_lengths[i]
        # Horizontal position: centered
        # Vertical position: i * v_offset, centered vertically
        
        centers = []
        for i in range(n_rows):
            n_in_row = row_lengths[i]
            if n_in_row == 0:
                continue
            # Horizontal positions, centered
            if n_in_row == 1:
                xs = np.array([0.0])
            else:
                xs = np.linspace(-spacing * (n_in_row - 1) / 2, spacing * (n_in_row - 1) / 2, n_in_row)
            # Offset for odd rows
            if i % 2 == 1:
                xs = xs + h_offset
            ys = np.full(n_in_row, (i - (n_rows - 1) / 2.0) * v_offset)
            for x, y in zip(xs, ys):
                centers.append((x, y))
        
        if len(centers) != N:
            continue
        centers = np.array(centers)
        
        # Check bounds
        if np.any(np.abs(centers) > 0.5):
            continue
        
        s = _score(centers)
        if s > best_score:
            best_score = s
            best_centers = centers

    return best_centers


def _neg_score(params):
    """Negative score for minimization."""
    spacing, v_offset, h_offset = params
    if spacing <= 0 or v_offset <= 0:
        return 1.0
    centers = _hexagonal_seed(spacing, v_offset, h_offset)
    if centers is None:
        return 1.0
    return -_score(centers)


def _optimize_hex():
    """Optimize hexagonal lattice parameters."""
    # Initial guess
    # For N=50, hex packing ~ 7-8 rows
    # Spacing ~ 1/8 = 0.125 for square, hex can be tighter
    x0 = [0.12, 0.10, 0.06]
    
    result = minimize(_neg_score, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
    
    if result.success or result.fun < 0:
        spacing, v_offset, h_offset = result.x
        centers = _hexagonal_seed(spacing, v_offset, h_offset)
        if centers is not None:
            return centers, _score(centers)
    
    return None, -1.0


def _local_refine(centers, max_iters=2000):
    """Local refinement using gradient-free optimization on individual positions."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Use Nelder-Mead on flattened positions
    def _neg_score_flat(params):
        c = params.reshape(N, 2)
        if np.any(np.abs(c) > 0.5):
            return 1.0
        return -_score(c)
    
    x0 = centers.flatten()
    result = minimize(_neg_score_flat, x0, method='Nelder-Mead',
                      options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12,
                               'initial_simplex': 1e-4})
    
    if result.fun < -best_score:
        new_centers = result.x.reshape(N, 2)
        if np.all(np.abs(new_centers) <= 0.5):
            return new_centers, _score(new_centers)
    
    return best_centers, best_score


def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Step 1: Find good hexagonal lattice parameters
    hex_centers, hex_score = _optimize_hex()
    
    if hex_centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    best_centers = hex_centers
    best_score = hex_score
    
    # Step 2: Local refinement
    if time.time() - START_TIME < DEADLINE - 30:
        refined, refined_score = _local_refine(hex_centers, max_iters=3000)
        if refined_score > best_score:
            best_centers = refined
            best_score = refined_score
    
    # Step 3: Try more local refinement with smaller steps
    if time.time() - START_TIME < DEADLINE - 10:
        def _neg_score_flat(params):
            c = params.reshape(N, 2)
            if np.any(np.abs(c) > 0.5):
                return 1.0
            return -_score(c)
        
        x0 = best_centers.flatten()
        result = minimize(_neg_score_flat, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'initial_simplex': 1e-5})
        if result.fun < -best_score:
            new_centers = result.x.reshape(N, 2)
            if np.all(np.abs(new_centers) <= 0.5):
                best_centers = new_centers
                best_score = _score(new_centers)
    
    # Ensure validity
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
