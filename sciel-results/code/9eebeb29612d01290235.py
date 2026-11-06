"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _score_from_centers(centers):
    """Compute the common radius for a set of centers inside [-0.5, 0.5]^2."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check boundaries
    dist_to_bound = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ], axis=1), axis=1)
    min_boundary = np.min(dist_to_bound)
    if min_boundary < -1e-12:
        return 0.0
    # Check pairwise distances
    # Use broadcasting for N=50 (2500 pairs, fine)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    if min_pair < -1e-12:
        return 0.0
    return min(min_boundary, min_pair)


def _generate_hex_packing(params, row_lengths):
    """
    Generate hexagonal lattice packing.
    params: [sx, sy, x_shift_odd, x_global, y_global]
    row_lengths: list of integers for each row's circle count.
    """
    sx, sy, x_shift_odd, x_global, y_global = params
    n_rows = len(row_lengths)
    
    centers = []
    for i, length in enumerate(row_lengths):
        if length == 0:
            continue
        # Vertical position: centered around y_global
        y = y_global + (i - (n_rows - 1) / 2.0) * sy
        
        # Horizontal position
        # Center the row horizontally around x_global
        # If odd number of rows, the middle row is at i=(n_rows-1)/2
        # Standard hex shift: odd rows (0-indexed) shifted by x_shift_odd
        shift = x_shift_odd if i % 2 != 0 else 0.0
        
        x_start = x_global + shift - sx * (length - 1) / 2.0
        for j in range(length):
            x = x_start + j * sx
            centers.append([x, y])
            
    centers = np.array(centers)
    # Ensure we have exactly N centers. 
    # The patterns provided should sum to N.
    if len(centers) != N:
        # Fallback: pad or truncate? Better to return invalid shape to fail score.
        # But we control the patterns, so this shouldn't happen.
        pass
    return centers


def construct():
    """
    Construct 50 circle centers using hexagonal lattice packing with
    various row length patterns, optimized to maximize the common radius.
    """
    
    # Define several candidate row patterns summing to 50
    patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4], # Current
        [8, 7, 7, 7, 7, 7, 7, 0], # 7 rows, one 8, six 7s? 8+42=50.
        [7, 7, 7, 7, 7, 7, 8, 0], # Reversed
        [6, 7, 7, 7, 7, 7, 7, 2], # 6+42+2=50
        [5, 7, 7, 7, 7, 7, 7, 3], # 5+42+3=50
        [4, 7, 7, 7, 7, 7, 7, 4], # 4+42+4=50
        [6, 6, 7, 7, 7, 7, 7, 3], # 12+35+3=50
        [5, 6, 7, 7, 7, 7, 7, 4], # 11+35+4=50
        [7, 7, 7, 7, 7, 7, 7, 1], # 49+1=50
        [8, 6, 8, 6, 8, 6, 8, 0], # 32+24=56 -> Invalid, skip
        [6, 8, 6, 8, 6, 8, 8, 0], # 18+24=42 -> Invalid
        [7, 8, 7, 8, 7, 8, 5, 0], # 7+8+7+8+7+8+5 = 50
        [8, 8, 7, 7, 7, 7, 6, 0], # 16+28+6=50
        [6, 7, 8, 7, 7, 7, 8, 0], # 6+7+8+7+7+7+8 = 50
        [5, 8, 7, 7, 7, 7, 9, 0], # 5+8+28+9 = 50
    ]
    
    # Filter patterns that sum to N
    valid_patterns = [p for p in patterns if sum(p) == N and len(p) > 0]
    
    best_score = -np.inf
    best_centers = None
    
    # Initial guesses for parameters: sx, sy, x_shift_odd, x_global, y_global
    # Ideal hex: sy = sx * sqrt(3)/2
    # If sx ~ 1/6, sy ~ 0.144
    # x_shift_odd ~ sx/2
    
    for pattern in valid_patterns:
        n_rows = len(pattern)
        # Estimate initial sx based on max row length
        max_len = max(pattern)
        sx_init = 1.0 / (max_len + 0.5) # Rough estimate to fit in width
        sy_init = sx_init * np.sqrt(3) / 2.0
        
        # If vertical span is too large, reduce sy
        total_height = (n_rows - 1) * sy_init
        if total_height > 0.9:
            sy_init = 0.9 / (n_rows - 1) if n_rows > 1 else 0.1
            # Adjust sx to maintain hex ratio? Or just let optimizer figure it out.
            # Let's try to keep sx reasonable.
            
        starts = [
            [sx_init, sy_init, sx_init / 2.0, 0.0, 0.0],
            [sx_init, sy_init, -sx_init / 2.0, 0.0, 0.0],
            [sx_init * 0.9, sy_init * 1.1, 0.0, 0.0, 0.0],
            [sx_init * 1.1, sy_init * 0.9, 0.0, 0.0, 0.0],
        ]
        
        def _neg_score(params):
            centers = _generate_hex_packing(params, pattern)
            if centers.shape != (N, 2):
                return 0.0
            return -_score_from_centers(centers)
        
        bounds = [
            (0.01, 0.5),   # sx
            (0.01, 0.5),   # sy
            (-0.5, 0.5),   # x_shift_odd
            (-0.5, 0.5),   # x_global
            (-0.5, 0.5)    # y_global
        ]
        
        for x0 in starts:
            try:
                result = minimize(_neg_score, x0, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_centers = _generate_hex_packing(result.x, pattern)
            except Exception:
                pass
    
    if best_centers is not None and best_score > 0:
        return best_centers
    
    # Fallback: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
# EVOLVE-BLOCK-END
