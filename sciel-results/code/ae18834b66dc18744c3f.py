"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from collections import deque

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N=11 unit regular hexagons inside a regular hexagon.
    Returns: inner (list of [x, y, angle_degrees]), center (x, y), side, angle
    """
    
    def min_enclosing_hexagon_side(points):
        """
        Find the minimum side length of a regular hexagon (any orientation) 
        that contains all given points. Returns (side, center_x, center_y, angle_deg).
        """
        points = np.array(points)
        best = (np.inf, 0.0, 0.0, 0.0)
        
        # Try different orientations (hexagon is symmetric every 60 degrees)
        for angle_deg in np.linspace(0, 60, 61):
            angle_rad = np.radians(angle_deg)
            
            # 6 face normals for the regular hexagon
            normals = np.array([[np.cos(angle_rad + k * np.pi / 3), 
                                  np.sin(angle_rad + k * np.pi / 3)] for k in range(6)])
            
            # Project points onto each normal direction
            projections = points @ normals.T  # shape (n_points, 6)
            
            # For each face, the range of projections
            proj_max = np.max(projections, axis=0)
            proj_min = np.min(projections, axis=0)
            ranges = proj_max - proj_min
            
            # Minimum inradius r for this orientation
            r_min = np.max(ranges) / 2.0
            # Side length s = 2*r / sqrt(3)
            s_min = r_min * 2.0 / np.sqrt(3)
            
            if s_min < best[0]:
                # Find center: c satisfies normals[k].c = (proj_max[k] + proj_min[k])/2
                midpoints = (proj_max + proj_min) / 2.0
                c, _, _, _ = np.linalg.lstsq(normals.T, midpoints, rcond=None)
                best = (s_min, c[0], c[1], angle_deg)
        
        return best
    
    # Flat-top hexagon lattice: axial (q, r) to cartesian
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    # Axial neighbor directions for hex grid
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    # BFS from origin to collect lattice points
    visited = set()
    queue = deque([(0, 0)])
    visited.add((0, 0))
    lattice_points = []
    
    while len(lattice_points) < N + 10:
        if not queue:
            break
        q, r = queue.popleft()
        lattice_points.append((q, r))
        for dq, dr in directions:
            nq, nr = q + dq, r + dr
            if (nq, nr) not in visited:
                visited.add((nq, nr))
                queue.append((nq, nr))
    
    # Convert to cartesian and sort by distance from origin
    cartesian_points = []
    for q, r in lattice_points:
        x, y = axial_to_cartesian(q, r)
        dist = np.sqrt(x * x + y * y)
        cartesian_points.append((dist, x, y))
    
    cartesian_points.sort()
    
    # Take the N closest to origin
    selected = [(x, y) for _, x, y in cartesian_points[:N]]
    
    # Compute minimal enclosing hexagon
    side, cx, cy, angle = min_enclosing_hexagon_side(selected)
    
    # Build inner hexagon list (all with 0-degree rotation, same orientation as lattice)
    inner = [[x, y, 0.0] for x, y in selected]
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END

