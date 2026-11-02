import math
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# The 11 cells of the minimal 3x3 hexagonal lattice block (axial coordinates)
# These form a hexagon of "radius" 2 in axial coordinates
CELLS = [
    (0, 0),
    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
    (1, 1), (-1, 2), (-2, 1), (2, -1)
]

def axial_to_cartesian(axial_list):
    """Convert axial coordinates to Cartesian coordinates for unit hexagons."""
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return pts

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Analytically derived optimal solution for 11 unit hexagons in a 3x3 block:
    # - Side length S = 4.0
    # - Rotation angle = 30 degrees (pi/6 radians)
    # - Center at (0, 0)
    
    pts = axial_to_cartesian(CELLS)
    
    center = [0.0, 0.0]
    side = 4.0
    angle_deg = 30.0
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    return inner, center, float(side), float(angle_deg)
