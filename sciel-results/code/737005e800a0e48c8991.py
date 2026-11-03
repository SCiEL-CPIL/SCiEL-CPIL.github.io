import math
import time
import numpy as np
# from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a

        r = side
        inradius = r * math.sqrt(3) / 2.0

        # Three pairs of parallel supporting lines
        for k in range(3):
            a = k * math.pi / 3.0
            dist = abs(lx * math.cos(a) + ly * math.sin(a))
            if dist > inradius + 1e-9:
                return False
        return True

    def minimal_enclosing_hexagon_side(points, angle_deg):
        """
        Calculate the minimum side length of a regular hexagon centered at the centroid
        of the points, with the given rotation angle, that contains all points.
        """
        if not points:
            return 0.0
        
        pts = np.array(points)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        
        # Translate to center
        pts_c = pts - np.array([cx, cy])
        
        # Rotate points by -angle_deg to align hexagon with axes
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated_pts = np.zeros_like(pts_c)
        rotated_pts[:, 0] = pts_c[:, 0] * cos_a + pts_c[:, 1] * sin_a
        rotated_pts[:, 1] = -pts_c[:, 0] * sin_a + pts_c[:, 1] * cos_a
        
        # The hexagon is defined by |x| <= inradius in 3 directions 60 deg apart.
        # Directions: 0, 60, 120 degrees.
        # For a point (x, y) in the rotated frame, the constraint is:
        # |x * cos(k*60) + y * sin(k*60)| <= inradius
        # So inradius >= max_k |x * cos(k*60) + y * sin(k*60)|
        
        max_dist = 0.0
        for p in rotated_pts:
            x, y = p
            for k in range(3):
                a = k * math.pi / 3.0
                dist = abs(x * math.cos(a) + y * math.sin(a))
                if dist > max_dist:
                    max_dist = dist
        
        # inradius = side * sqrt(3) / 2  =>  side = inradius * 2 / sqrt(3)
        side = max_dist * 2.0 / math.sqrt(3)
        return side

    # Topologies for 12 unit hexagons
    # Lattice vectors for hexagonal packing (side=1):
    # Distance between centers of touching hexagons is sqrt(3).
    # Basis vectors: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    
    d = math.sqrt(3)
    
    def lattice_to_cart(i, j):
        return (d * (i + j / 2.0), 3.0 * j / 2.0)

    # Topology 1: Hexagonal cluster (Center + 6 neighbors + 5 from next ring)
    # This is a "hexagon of hexagons" shape but missing 7 from the full 19-cluster.
    # Let's use a more compact shape.
    # A 4x3 rectangle in the hex lattice is often good.
    # Or a "flower" shape.
    
    topologies = []
    
    # T1: 4x3 rectangular grid in hex lattice
    # Rows: 4, Cols: 3
    # (i, j) where i in 0..2, j in 0..3
    # But hex lattice is staggered.
    # Let's define explicit sets of (i, j) coordinates in the triangular lattice.
    
    # T1: 3x4 block
    t1 = []
    for j in range(4):
        for i in range(3):
            t1.append((i, j))
    topologies.append(t1)
    
    # T2: Hexagonal cluster (1 center, 6 neighbors, 5 more)
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (partial): (2,0), (1,1), (0,2), (-1,2), (-2,1)
    t2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    topologies.append(t2)
    
    # T3: 2x6 strip
    t3 = []
    for j in range(6):
        for i in range(2):
            t3.append((i, j))
    topologies.append(t3)
    
    # T4: 5x3 block (slightly wider)
    t4 = []
    for j in range(3):
        for i in range(5):
            t4.append((i, j))
    topologies.append(t4)
    
    # T5: "Pentagon" like shape?
    # Let's try a shape that minimizes the max distance from centroid.
    # A perfect hexagon of 19 is best, but we have 12.
    # Maybe a 1-2-3-4-3-1? No, that's 14.
    # 1-2-3-3-2-1 = 12.
    t5 = []
    # Row -2: 1 hex at (0, -2) -> (0, -2)
    # Row -1: 2 hexes at (-0.5, -1), (0.5, -1) -> (-1, -1), (0, -1) in lattice?
    # Let's stick to lattice coordinates.
    # Row j=0: 4 hexes (i=0,1,2,3)
    # Row j=1: 4 hexes (i=0,1,2,3)
    # Row j=2: 4 hexes (i=0,1,2,3)
    # That's T1.
    
    # Let's try a more circular distribution.
    # Center (0,0)
    # 6 neighbors
    # 5 from ring 2 chosen to be symmetric.
    # Ring 2 coords: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-1)
    # To be symmetric, pick (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-1) is 11.
    # We need 5.
    # Maybe (2,0), (0,2), (-2,2), (-2,0), (0,-2)?
    t5 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2)
    ]
    topologies.append(t5)
    
    best_side = float('inf')
    best_config = None
    
    for topo in topologies:
        if time_left() < 5.0:
            break
            
        # Generate Cartesian positions
        positions = []
        for i, j in topo:
            x, y = lattice_to_cart(i, j)
            positions.append((x, y))
        
        positions = np.array(positions)
        
        # Center of mass
        cx = np.mean(positions[:, 0])
        cy = np.mean(positions[:, 1])
        
        # Translate to origin for calculation
        positions_c = positions - np.array([cx, cy])
        
        # Try different rotations of the outer hexagon
        # The inner hexagons are fixed at angle 0 (flat top).
        # We can rotate the outer hexagon to minimize its size.
        # Due to 6-fold symmetry of the outer hexagon, we only need to check 0 to 60 degrees.
        # However, the cluster shape might not be symmetric.
        
        for angle_deg in range(0, 61, 5):
            if time_left() < 5.0:
                break
                
            side = minimal_enclosing_hexagon_side(positions_c, angle_deg)
            
            if side < best_side:
                best_side = side
                best_config = {
                    'positions': positions,
                    'cx': cx,
                    'cy': cy,
                    'angle': angle_deg
                }
        
        # Also try rotating the inner cluster?
        # If we rotate the cluster, the relative positions change.
        # Since we can rotate each inner hexagon independently, and they are identical,
        # we can just rotate the whole cluster and keep inner angles at 0 relative to cluster?
        # No, the task allows individual rotation.
        # However, for a lattice packing, keeping them aligned is optimal for non-overlap.
        # If we rotate the cluster, we should rotate the inner hexagons too to maintain alignment.
        # But the "angle" in the output is per hexagon.
        # Let's just stick to the cluster at angle 0 and rotate the outer boundary.
        # This is sufficient for lattice topologies.
        
        # If we want to be thorough, we could also rotate the cluster itself.
        # But that's equivalent to rotating the outer hexagon in the other direction.
        # Since we're sweeping the outer hexagon angle 0-60, it covers all relative orientations.

    if best_config is None:
        # Fallback: simple grid
        positions = []
        for j in range(4):
            for i in range(3):
                x, y = lattice_to_cart(i, j)
                positions.append((x, y))
        positions = np.array(positions)
        cx = np.mean(positions[:, 0])
        cy = np.mean(positions[:, 1])
        best_config = {
            'positions': positions,
            'cx': cx,
            'cy': cy,
            'angle': 0.0
        }
        best_side = minimal_enclosing_hexagon_side(positions - np.array([cx, cy]), 0.0)

    # Construct the result
    inner = []
    for pos in best_config['positions']:
        # Inner hexagons are at angle 0 (flat top)
        inner.append([float(pos[0]), float(pos[1]), 0.0])
    
    return inner, (float(best_config['cx']), float(best_config['cy'])), float(best_side), float(best_config['angle'])

# EVOLVE-BLOCK-END
