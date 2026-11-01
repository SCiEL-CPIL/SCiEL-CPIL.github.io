import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    points = []
    points.append((0.0, 0.0))
    
    remaining = N - 1
    ring = 1
    while remaining > 0:
        count_in_ring = 6 * ring
        if count_in_ring > remaining:
            count_in_ring = remaining
            
        ring_points = []
        corner_points = []
        
        for q in range(-ring, ring + 1):
            for r in range(-ring, ring + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) == ring:
                    x = np.sqrt(3) * (q + r / 2.0)
                    y = 1.5 * r
                    
                    # Check if corner
                    # Corners are (ring, -ring), (ring, 0), (0, ring), (-ring, ring), (-ring, 0), (0, -ring)
                    is_corner = False
                    if (q == ring and r == -ring) or \
                       (q == ring and r == 0) or \
                       (q == 0 and r == ring) or \
                       (q == -ring and r == ring) or \
                       (q == -ring and r == 0) or \
                       (q == 0 and r == -ring):
                        is_corner = True
                        
                    if is_corner:
                        corner_points.append((x, y))
                    else:
                        ring_points.append((x, y))
                        
        # We want to pick 'count_in_ring' points.
        # Prefer non-corners to minimize bounding box.
        # If we need more than non-corners, we take corners.
        
        # Sort non-corners by angle for determinism
        ring_points.sort(key=lambda p: np.arctan2(p[1], p[0]))
        corner_points.sort(key=lambda p: np.arctan2(p[1], p[0]))
        
        selected = []
        if len(ring_points) >= count_in_ring:
            selected = ring_points[:count_in_ring]
        else:
            selected = ring_points[:]
            need = count_in_ring - len(selected)
            selected.extend(corner_points[:need])
            
        points.extend(selected)
        remaining -= count_in_ring
        ring += 1
        
    # Calculate minimal S_out
    normals = [
        (0.0, 1.0),
        (np.sqrt(3)/2, 0.5),
        (np.sqrt(3)/2, -0.5),
        (0.0, -1.0),
        (-np.sqrt(3)/2, -0.5),
        (-np.sqrt(3)/2, 0.5)
    ]
    
    max_proj = 0.0
    for nx, ny in normals:
        proj = 0.0
        for x, y in points:
            p = x * nx + y * ny
            if p > proj:
                proj = p
        if proj > max_proj:
            max_proj = proj
            
    S_out = max_proj * (2.0 / np.sqrt(3)) + 1.0
    
    inner = [[x, y, 0.0] for x, y in points]
    
    return inner, (0.0, 0.0), S_out, 0.0
# EVOLVE-BLOCK-END
