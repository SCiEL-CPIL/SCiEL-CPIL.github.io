import math
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Define a set of axial coordinates for N=11 unit hexagons
    # using a hexagonal lattice pattern.
    # Axial coordinates (q, r)
    # Cartesian conversion:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # We generate coordinates in order of distance from origin
    coords = []
    # Add center
    coords.append((0, 0))
    
    # Add neighbors (distance 1)
    # Directions: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    dirs = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    for d in dirs:
        if len(coords) < N:
            coords.append(d)
            
    # Add next ring (distance 2)
    # Directions from origin: (2,0), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,0), (-1,1), (0,2), (1,2), (2,1)
    # Actually, let's just iterate a bounding box of q,r in -3..3
    # and pick the closest ones not already added
    added = set(coords)
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if (q, r) not in added:
                # Calculate squared distance for sorting (approximate Euclidean)
                x = math.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                dist_sq = x*x + y*y
                candidates.append((dist_sq, q, r))
    
    candidates.sort()
    for dist_sq, q, r in candidates:
        if len(coords) >= N:
            break
        if (q, r) not in added:
            coords.append((q, r))
            added.add((q, r))

    # Convert to Cartesian centers
    centers = []
    for q, r in coords:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append((x, y))

    # Calculate the required outer side length S
    # The outer hexagon is centered at (0,0).
    # We need to contain all vertices of the unit hexagons at these centers.
    # A unit hexagon at center (cx, cy) has vertices at distance 1 from (cx, cy).
    # The furthest point from origin in direction theta is max(cx*cos(t), cy*sin(t)) + 1?
    # Actually, we can just compute the max distance from origin to any vertex of any inner hex.
    # Then find the minimal S of a regular hexagon that contains all these points.
    
    # For a regular hexagon with side S centered at origin, the distance to the boundary
    # in direction phi is R(phi) = (S * sqrt(3) / 2) / cos(phi_mod), where phi_mod is the
    # angle relative to the nearest side normal.
    # Side normals are at 30, 90, 150, 210, 270, 330 degrees.
    
    # We assume the container is aligned such that side normals are at 30, 90, etc.
    # (This corresponds to a "flat-top" or "pointy-top" orientation? 
    #  If normals are at 30, 90... the vertices are at 0, 60... 
    #  This is a pointy-top hexagon if y is up? 
    #  Let's just use the formula.)
    
    max_s_required = 0.0
    
    # For each inner hex center, check its 6 vertices
    for cx, cy in centers:
        for k in range(6):
            angle = math.radians(60 * k)
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            
            # Calculate distance from origin
            r = math.sqrt(vx*vx + vy*vy)
            if r == 0:
                continue
            
            # Angle of the point
            theta = math.atan2(vy, vx)
            
            # Find the angle to the nearest side normal of the container
            # Container side normals are at 30 + 60*i degrees.
            # Let's assume container is rotated by 0.
            # Normals at: pi/6, pi/6+pi/3, ...
            # We want to minimize the required S.
            # S >= r * cos(delta) / (sqrt(3)/2) ? No.
            # r <= (S * sqrt(3) / 2) / cos(delta)
            # S >= 2 * r * cos(delta) / sqrt(3)
            
            # delta is the angular difference between theta and the nearest normal.
            # Normalize theta to [0, 2pi)
            theta_mod = theta % (2 * math.pi)
            
            # Nearest normal angle
            # Normals are at pi/6 + i*pi/3
            # i = round((theta_mod - pi/6) / (pi/3))
            i = int(round((theta_mod - math.pi/6) / (math.pi/3)))
            normal_angle = math.pi/6 + i * math.pi/3
            
            delta = abs(theta_mod - normal_angle)
            # Ensure delta is the smallest angle difference (should be <= pi/6)
            if delta > math.pi/2:
                 delta = 2*math.pi - delta
            # Actually, due to symmetry, delta is always in [0, pi/6] if we pick nearest.
            
            s_req = (2.0 * r * math.cos(delta)) / math.sqrt(3)
            if s_req > max_s_required:
                max_s_required = s_req

    # Add a small epsilon for safety
    S = max_s_required + 0.01
    
    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
    
    center = (0.0, 0.0)
    side = S
    angle = 0.0
    
    return inner, center, side, angle
