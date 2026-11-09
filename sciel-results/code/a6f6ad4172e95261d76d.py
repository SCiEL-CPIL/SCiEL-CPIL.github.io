"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing outer side length.
    All inner hexagons have the same orientation (0 degrees).
    """
    
    def compute_enclosing_hexagon_side_length(centers):
        """
        Compute the side length of the minimal enclosing regular hexagon (same orientation)
        for a set of unit hexagons with given centers.
        """
        centers = np.array(centers)
        directions = [0, np.pi/3, 2*np.pi/3]
        
        max_side = 0
        for d in directions:
            projections = centers[:, 0] * np.cos(d) + centers[:, 1] * np.sin(d)
            extent = projections.max() - projections.min()
            side_length = extent / np.sqrt(3) + 1
            max_side = max(max_side, side_length)
        
        return max_side
    
    s = np.sqrt(3)
    candidates = []
    
    # Generate various row-based arrangements
    row_patterns = [
        [4, 3, 4],
        [3, 4, 3, 1],
        [1, 4, 4, 2],
        [2, 4, 4, 1],
        [2, 3, 4, 2],
        [1, 3, 5, 2],
        [2, 3, 3, 3],
        [3, 3, 3, 2],
        [4, 4, 3],
        [3, 4, 4],
        [1, 2, 3, 4, 1],
        [2, 2, 3, 2, 2],
        [1, 3, 3, 3, 1],
        [2, 3, 3, 2, 1],
        [1, 2, 4, 3, 1],
    ]
    
    for pattern in row_patterns:
        if sum(pattern) != N:
            continue
        n_rows = len(pattern)
        centers = []
        for j, count in enumerate(pattern):
            for i in range(count):
                x = (i - (count - 1) / 2.0) * s
                y = (j - (n_rows - 1) / 2.0) * s * np.sqrt(3) / 2
                centers.append([x, y])
        candidates.append(centers)
    
    # Also try a more compact "flower" arrangement: center + 6 ring + 4 extra
    # Center at (0,0), ring at distance s in 6 directions
    centers_flower = [[0.0, 0.0]]
    for k in range(6):
        angle = k * np.pi / 3
        centers_flower.append([s * np.cos(angle), s * np.sin(angle)])
    # Add 4 more in the gaps (at distance s*sqrt(3) from center, at angles 30, 90, 150, 210, 270, 330)
    # Actually, the gaps between ring hexagons are at angles 30, 90, etc.
    for k in range(4):
        angle = (2 * k + 1) * np.pi / 6  # 30, 90, 150, 210
        # Distance to fit in the gap
        centers_flower.append([s * np.sqrt(3) * np.cos(angle), s * np.sqrt(3) * np.sin(angle)])
    candidates.append(centers_flower)
    
    best_side = float('inf')
    best_centers = None
    
    for centers in candidates:
        if len(centers) != N:
            continue
        side_len = compute_enclosing_hexagon_side_length(centers)
        if side_len < best_side:
            best_side = side_len
            best_centers = centers
    
    if best_centers is None:
        best_centers = [[k * s, 0] for k in range(N)]
        best_side = compute_enclosing_hexagon_side_length(best_centers)
    
    # Center the configuration
    cx = np.mean([c[0] for c in best_centers])
    cy = np.mean([c[1] for c in best_centers])
    centered = [[c[0] - cx, c[1] - cy] for c in best_centers]
    
    inner = [[c[0], c[1], 0.0] for c in centered]
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END

