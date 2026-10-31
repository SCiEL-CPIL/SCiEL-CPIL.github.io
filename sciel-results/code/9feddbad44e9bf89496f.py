"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Try to import relax from the provided environment
try:
    from science_chemistry import relax
    HAS_RELAX = True
except ImportError:
    HAS_RELAX = False
    def relax(positions):
        # Fallback: simple gradient descent or return input if relax not available
        # But the task says it is available.
        return positions

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    # 1. Generate FCC lattice points
    # Lattice constant a such that nearest neighbor distance r_min = 2^(1/6)
    # In FCC, nearest neighbor distance is a / sqrt(2)
    # So a = 2^(1/6) * sqrt(2)
    a = (2 ** (1/6)) * np.sqrt(2)
    
    # Points in FCC: (x, y, z) = (a/2 * u, a/2 * v, a/2 * w)
    # where u, v, w are integers and u+v+w is even.
    
    points = []
    # Search range: we need 38 points. The radius will be small.
    # Max u^2+v^2+w^2 for 38 points?
    # Shell 1 (r^2=2): 12 pts. Cumulative 13.
    # Shell 2 (r^2=4): 6 pts. Cumulative 19.
    # Shell 3 (r^2=6): 24 pts. Cumulative 43.
    # So we need points up to r^2=6.
    # u^2 <= 6 => u in [-2, 2].
    
    max_u = 3
    for u in range(-max_u, max_u + 1):
        for v in range(-max_u, max_u + 1):
            for w in range(-max_u, max_u + 1):
                if (u + v + w) % 2 == 0:
                    x = a * 0.5 * u
                    y = a * 0.5 * v
                    z = a * 0.5 * w
                    points.append([x, y, z])
    
    points = np.array(points)
    
    # Calculate distances from origin
    dists = np.sum(points**2, axis=1)
    
    # Sort by distance
    sorted_indices = np.argsort(dists)
    
    # Take the first 38 unique points
    # Note: The origin (0,0,0) is included.
    selected_points = points[sorted_indices[:N]]
    
    # Initial positions
    positions = selected_points.copy()
    
    # 2. Relax the positions
    if HAS_RELAX:
        # relax function signature: relax(positions)
        # It likely minimizes the LJ energy.
        # We need to ensure we don't exceed time.
        # We can't easily pass a deadline to relax, so we hope it's fast.
        # L-BFGS is usually fast for 38 atoms.
        try:
            relaxed_positions = relax(positions)
            # Check if relaxed positions are valid (finite, etc.)
            if np.all(np.isfinite(relaxed_positions)):
                positions = relaxed_positions
        except Exception:
            pass # Keep original if relax fails
    
    # 3. Final check and return
    # Ensure atoms are at least 0.5 apart?
    # The LJ minimum is at 1.12, so they should be far apart.
    # Relaxation should maintain this.
    
    return positions

# EVOLVE-BLOCK-END
