"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import combinations
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """Construct 11 unit hexagons inside a minimal outer hexagon."""
    sqrt3 = math.sqrt(3)
    
    def axial_to_cartesian(q, r):
        return (sqrt3 * (q + r / 2.0), 1.5 * r)
    
    # Base hexagonal cluster of 7
    base = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Generate more candidates
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if (q, r) not in base:
                candidates.append((q, r))
    
    # Six half-plane normals for flat-top hexagon
    normals = []
    for i in range(6):
        angle = math.radians(60 * i + 30)
        normals.append((math.cos(angle), math.sin(angle)))
    
    def min_enclosing_side_centered(centers):
        """Assume outer hexagon at origin, find min side to enclose all unit hexagons at centers."""
        max_apothem = 0.0
        for (cx, cy) in centers:
            for j in range(6):
                angle = math.radians(60 * j)
                vx = cx + math.cos(angle)
                vy = cy + math.sin(angle)
                for (nx, ny) in normals:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_apothem:
                        max_apothem = proj
        return max_apothem / (sqrt3 / 2.0)
    
    best_side = float('inf')
    best_config = None
    
    # Try combinations of 4 from candidates
    # To limit computation, only consider candidates near the base
    near_candidates = [c for c in candidates if max(abs(c[0]), abs(c[1]), abs(c[0]+c[1])) <= 2]
    
    for combo in combinations(range(len(near_candidates)), 4):
        config_axial = base + [near_candidates[i] for i in combo]
        centers = [axial_to_cartesian(q, r) for (q, r) in config_axial]
        
        # Center the configuration
        mean_x = sum(c[0] for c in centers) / len(centers)
        mean_y = sum(c[1] for c in centers) / len(centers)
        centered = [(c[0] - mean_x, c[1] - mean_y) for c in centers]
        
        side = min_enclosing_side_centered(centered)
        if side < best_side:
            best_side = side
            best_config = config_axial
    
    if best_config is None:
        centers = [(sqrt3 * k, 0.0) for k in range(-(N//2), N - N//2)]
        return [[c[0], c[1], 0.0] for c in centers], (0.0, 0.0), 40.0, 0.0
    
    centers = [axial_to_cartesian(q, r) for (q, r) in best_config]
    mean_x = sum(c[0] for c in centers) / len(centers)
    mean_y = sum(c[1] for c in centers) / len(centers)
    centered = [(c[0] - mean_x, c[1] - mean_y) for c in centers]
    
    inner = [[c[0], c[1], 0.0] for c in centered]
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END

