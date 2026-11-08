import numpy as np
import time
from itertools import product

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        ang = a + i * np.pi / 3
        verts.append([cx + side * np.cos(ang), cy + side * np.sin(ang)])
    return np.array(verts)

def _all_inner_vertices(inner):
    """Return all vertices of all inner hexagons."""
    all_verts = []
    for (x, y, ang) in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    return np.vstack(all_verts)

def _min_enclosing_hex_side(vertices, center, angle_deg):
    """Given vertices, center, and outer hex rotation, find minimum side length."""
    cx, cy = center
    a = np.radians(angle_deg)
    # 6 edge normals at angles a + 30 + i*60 degrees
    min_side = 0.0
    for i in range(6):
        ang = a + np.pi/6 + i * np.pi/3
        nx, ny = np.cos(ang), np.sin(ang)
        # Project all vertices onto this normal
        projs = (vertices[:, 0] - cx) * nx + (vertices[:, 1] - cy) * ny
        max_proj = np.max(projs)
        # Side length needed: max_proj / (sqrt(3)/2)
        side_needed = max_proj / (np.sqrt(3) / 2)
        if side_needed > min_side:
            min_side = side_needed
    return min_side

def _best_side_for_arrangement(inner):
    """Find best center and rotation for given inner arrangement."""
    vertices = _all_inner_vertices(inner)
    # Try various centers around centroid and various rotations
    centroid = np.mean(vertices, axis=0)
    
    best_side = float('inf')
    best_center = (float(centroid[0]), float(centroid[1]))
    best_angle = 0.0
    
    # Try rotations: 0 to 60 degrees in small steps
    for angle_deg in np.arange(0, 60, 1.0):
        # Try centers: grid around centroid
        cx_range = np.linspace(centroid[0] - 2, centroid[0] + 2, 15)
        cy_range = np.linspace(centroid[1] - 2, centroid[1] + 2, 15)
        for cx in cx_range:
            for cy in cy_range:
                side = _min_enclosing_hex_side(vertices, (cx, cy), angle_deg)
                if side < best_side:
                    best_side = side
                    best_center = (float(cx), float(cy))
                    best_angle = float(angle_deg)
    
    return best_side, best_center, best_angle

def _honeycomb_positions(rows):
    """Generate honeycomb positions for given row pattern."""
    positions = []
    dx = np.sqrt(3)  # horizontal spacing
    dy = 1.5  # vertical spacing (3/2 for unit hexagons)
    
    total_rows = len(rows)
    for r, count in enumerate(rows):
        row_y = (r - (total_rows - 1) / 2) * dy
        # Offset alternating rows
        if r % 2 == 1:
            x_offset = dx / 2
        else:
            x_offset = 0
        for c in range(count):
            x = (c - (count - 1) / 2) * dx + x_offset
            positions.append([x, row_y, 0.0])
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try various compact arrangements for 11 hexagons
    arrangements = []
    
    # Row patterns that sum to 11
    row_patterns = [
        [1, 2, 3, 2, 1, 2],  # 11
        [2, 3, 2, 3, 1],     # 11
        [1, 3, 3, 3, 1],     # 11
        [2, 2, 3, 2, 2],     # 11
        [3, 3, 2, 3],         # 11
        [1, 2, 3, 3, 2],      # 11
        [2, 3, 3, 2, 1],      # 11
        [3, 2, 3, 2, 1],      # 11
        [1, 3, 2, 3, 2],      # 11
        [2, 1, 3, 3, 2],      # 11
        [3, 1, 3, 2, 2],      # 11
        [1, 1, 3, 3, 3],      # 11
        [2, 2, 1, 3, 3],      # 11
        [3, 3, 3, 2],          # 11
        [1, 4, 3, 3],          # 11
        [2, 4, 3, 2],          # 11
        [3, 4, 2, 2],          # 11
        [4, 3, 2, 2],          # 11
        [5, 3, 3],             # 11
        [6, 3, 2],             # 11
        [7, 2, 2],             # 11
        [8, 2, 1],             # 11
        [9, 1, 1],             # 11
        [10, 1],               # 11
        [11],                  # 11
        [2, 3, 6],             # 11
        [1, 5, 5],             # 11
        [3, 5, 3],             # 11
        [1, 2, 4, 4],          # 11
        [2, 2, 4, 3],          # 11
        [1, 3, 4, 3],          # 11
        [2, 3, 4, 2],          # 11
        [3, 3, 4, 1],          # 11
        [1, 1, 4, 5],          # 11
        [2, 1, 4, 4],          # 11
        [1, 2, 5, 3],          # 11
        [2, 2, 5, 2],          # 11
        [1, 3, 5, 2],          # 11
        [2, 3, 5, 1],          # 11
        [3, 3, 5],             # 11
        [1, 4, 6],             # 11
        [2, 4, 5],             # 11
        [3, 4, 4],             # 11
        [1, 5, 5],             # 11
        [2, 5, 4],             # 11
        [3, 5, 3],             # 11
        [4, 5, 2],             # 11
        [5, 5, 1],             # 11
        [1, 1, 1, 8],          # 11
        [2, 1, 1, 7],          # 11
        [1, 2, 1, 7],          # 11
        [3, 1, 1, 6],          # 11
        [1, 3, 1, 6],          # 11
        [2, 3, 1, 5],          # 11
        [1, 2, 3, 5],          # 11
        [2, 2, 3, 4],          # 11
        [1, 3, 3, 4],          # 11
        [3, 3, 3, 2],          # 11
        [1, 1, 3, 6],          # 11
        [2, 1, 3, 5],          # 11
        [1, 2, 3, 5],          # 11
        [2, 2, 3, 4],          # 11
        [1, 3, 3, 4],          # 11
        [3, 3, 3, 2],          # 11
    ]
    
    # Also try the "flower" arrangement: center + 6 ring + 4 extra
    # And other symmetric patterns
    
    best_overall = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for pattern in row_patterns:
        if sum(pattern) != N:
            continue
        if time.time() > deadline:
            break
        
        inner = _honeycomb_positions(pattern)
        
        # Also try with some hexagons rotated 30 degrees (flat-top vs pointy-top)
        # First try all with 0 degree rotation
        side, center, angle = _best_side_for_arrangement(inner)
        if side < best_overall:
            best_overall = side
            best_inner = [list(x) for x in inner]
            best_center = center
            best_angle = angle
        
        # Try with alternating rows rotated 30 degrees
        inner_rot = []
        for r, count in enumerate(pattern):
            for c in range(count):
                # Get position
                idx = len(inner_rot)
                if idx < len(inner):
                    x, y, _ = inner[idx]
                    # Rotate every other row by 30 degrees
                    ang = 30.0 if r % 2 == 1 else 0.0
                    inner_rot.append([x, y, ang])
        
        if len(inner_rot) == N:
            side, center, angle = _best_side_for_arrangement(inner_rot)
            if side < best_overall:
                best_overall = side
                best_inner = inner_rot
                best_center = center
                best_angle = angle
    
    # Also try the compact "hexagonal cluster" arrangement
    # Center + 6 neighbors + 4 more in a compact way
    if time.time() < deadline:
        # Hexagonal cluster: center at origin, 6 at distance sqrt(3), 4 more
        dx = np.sqrt(3)
        dy = 1.5
        cluster = [[0, 0, 0.0]]
        # 6 neighbors
        for i in range(6):
            ang = i * np.pi / 3
            x = dx * np.cos(ang)
            y = dx * np.sin(ang)
            cluster.append([x, y, 0.0])
        # 4 more: add at positions that fill gaps
        # Positions at distance 2*dx in certain directions, or at (dx*1.5, dy*1.5) etc.
        extra_positions = [
            [2*dx, 0, 0.0],
            [dx, dy*2, 0.0],
            [-dx, dy*2, 0.0],
            [0, 2*dy, 0.0],
        ]
        cluster.extend(extra_positions)
        
        if len(cluster) == N:
            side, center, angle = _best_side_for_arrangement(cluster)
            if side < best_overall:
                best_overall = side
                best_inner = [list(x) for x in cluster]
                best_center = center
                best_angle = angle
    
    # Try another compact arrangement: 3-4-4 pattern (rows of 3, 4, 4)
    if time.time() < deadline:
        for pattern in [[3, 4, 4], [4, 3, 4], [4, 4, 3], [2, 4, 5], [5, 4, 2], [3, 5, 3], [1, 5, 5], [5, 5, 1]]:
            if sum(pattern) != N:
                continue
            inner = _honeycomb_positions(pattern)
            side, center, angle = _best_side_for_arrangement(inner)
            if side < best_overall:
                best_overall = side
                best_inner = [list(x) for x in inner]
                best_center = center
                best_angle = angle
    
    if best_inner is None:
        # Fallback: linear arrangement
        inner = [[dx * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_overall, best_angle
# EVOLVE-BLOCK-END
