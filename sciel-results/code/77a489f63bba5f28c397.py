<<<<<<< SEARCH
import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0

def hex_support(alpha, direction_angle):
    """
    Compute the support function of a unit regular hexagon with rotation alpha
    in the given direction.
    For a regular hexagon of side 1 (circumradius 1), vertices are at angles
    alpha + k*pi/3 for k=0..5.
    Support in direction theta = max_k cos(theta - alpha - k*pi/3)
    """
    diff = (direction_angle - alpha) % (math.pi / 3.0)
    # The max of cos over the 6 directions
    # cos is symmetric, so we can reduce to [0, pi/6] range
    d = min(diff, math.pi / 3.0 - diff)
    return math.cos(d)

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Generate 11 points on a triangular lattice
    # Axial coordinates (q, r)
    # This is a compact "flower" shape: center + 6 neighbors + 4 from the next ring
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Use a very small expansion to ensure strict disjointness (no touching)
    # while keeping the size minimal.
    SCALE = 1.0 + 1e-6
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts)
    n_pts = len(pts)
=======
import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0

def hex_support(alpha, direction_angle):
    """
    Compute the support function of a unit regular hexagon with rotation alpha
    in the given direction.
    For a regular hexagon of side 1 (circumradius 1), vertices are at angles
    alpha + k*pi/3 for k=0..5.
    Support in direction theta = max_k cos(theta - alpha - k*pi/3)
    """
    diff = (direction_angle - alpha) % (math.pi / 3.0)
    d = min(diff, math.pi / 3.0 - diff)
    return math.cos(d)

def generate_lattice_config(axial_coords, scale=1.0 + 1e-6):
    """Generate Cartesian points from axial coordinates with given scale."""
    pts = []
    for q, r in axial_coords:
        x = math.sqrt(3) * (q + r / 2.0) * scale
        y = 1.5 * r * scale
        pts.append((x, y))
    return np.array(pts)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Multiple candidate configurations on triangular lattice
    # Config 1: Standard flower (center + 6 neighbors + 4 from ring 2)
    axial1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Config 2: Alternate ring-2 selection for better symmetry
    axial2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-2, 1), (-1, -2), (2, -1)
    ]
    
    # Config 3: Slightly different ring-2 placement
    axial3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (0, 2), (-2, 2), (-2, 0), (2, -2)
    ]
    
    configs = [axial1, axial2, axial3]
    all_candidates = []
    
    for axial in configs:
        pts = generate_lattice_config(axial)
        all_candidates.append(pts)
        
    n_pts = 11
>>>>>>> REPLACE

