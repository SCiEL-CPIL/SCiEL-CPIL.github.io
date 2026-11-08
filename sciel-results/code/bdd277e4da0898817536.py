import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Helper: Generate vertices of a regular hexagon
    def hex_verts(cx, cy, s, rot_deg):
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    # Strategy: 
    # 1. Define a specific "cluster" shape for 12 hexagons that is known to pack well.
    #    A common optimal packing for small N in a hexagon is a "flower" or "hexagonal number" shape.
    #    For N=12, a 2-3-2-3-2 or similar arrangement isn't a perfect hexagon.
    #    However, the "Honeycomb" lattice is the densest packing in the plane.
    #    If we take a subset of the honeycomb lattice, the minimal bounding hexagon 
    #    will be very efficient.
    #    
    #    Let's generate a set of 12 positions from the hexagonal lattice.
    #    Lattice vectors for hexagons of side 1:
    #    Distance between centers of touching hexagons is sqrt(3).
    #    v1 = (sqrt(3), 0)
    #    v2 = (sqrt(3)/2, 1.5)
    #    
    #    We need to select 12 points from this lattice such that their convex hull 
    #    is as "hexagonal" as possible to minimize the enclosing regular hexagon.
    #    
    #    Candidate configurations for 12 hexagons:
    #    A. 3x4 rectangle in lattice coords (distorted hexagon)
    #    B. "Hexagonal" shape: 1 center, 6 around it, 5 more? No, 1+6=7. Next is 19.
    #       So 12 is not a magic hexagonal number.
    #    C. Two rows of 6? No, that's long and thin.
    #    D. A "chunky" shape.
    #    
    #    Let's try to optimize the rotation and translation of a fixed lattice-based set.
    #    We will define a specific set of 12 lattice points, then find the best 
    #    rotation/translation to minimize the bounding regular hexagon size.
    #    
    #    Actually, a simpler and often optimal strategy for "fitting N hexagons in a hexagon"
    #    is to assume the inner hexagons are aligned with the outer one (angle 0).
    #    The current program failed to find a good S with Nelder-Mead.
    #    Why? Because the search space is non-convex and the penalty landscape is tricky.
    #    
    #    NEW STRATEGY:
    #    Use a "Greedy Placement" or "Simulated Annealing" on the positions, 
    #    but with a MUCH better initial guess and a more robust objective.
    #    Or, even better: **Direct Construction via Symmetry**.
    #    
    #    Let's assume the 12 hexagons form a symmetric pattern.
    #    Common patterns for 12:
    #    - 2 rows of 3, 2 rows of 3? (2x3 grid of "super-hexagons"?)
    #    - Let's try a 4x3 grid of hexagons in the lattice.
    #    
    #    Lattice coordinates (i, j) -> Cartesian:
    #    x = sqrt(3) * (i + j/2)
    #    y = 1.5 * j
    #    
    #    Let's select 12 points:
    #    i in [0, 1, 2, 3], j in [0, 1, 2, 3] is 16 points.
    #    We need 12.
    #    Let's take a 3x4 block: i in [0,1,2], j in [0,1,2,3].
    #    Points:
    #    (0,0), (1,0), (2,0)
    #    (0,1), (1,1), (2,1)
    #    (0,2), (1,2), (2,2)
    #    (0,3), (1,3), (2,3)
    #    
    #    This forms a parallelogram.
    #    Let's calculate the bounding box of these centers.
    #    Then we add the "radius" of the hexagons (1.0) to get the required outer size.
    #    
    #    But we can rotate the whole assembly!
    #    We will search for the optimal rotation angle theta of this fixed lattice pattern.
    #    For each theta, we calculate the minimal regular hexagon side length S that contains
    #    all 12 hexagons (centers + their extent).
    #    We pick the theta that minimizes S.
    #    
    #    This reduces the problem from 24 dimensions (12 x, 12 y) to 1 dimension (theta),
    #    plus potentially translation (which can be set to center the pattern at origin).
    #    This is a massive structural change: from local search in high-D to global 1-D search 
    #    over a fixed geometric pattern.

    # 1. Generate the base lattice points for a 3x4 block
    sqrt3 = np.sqrt(3)
    base_points = []
    for i in range(3):
        for j in range(4):
            x = sqrt3 * (i + 0.5 * j)
            y = 1.5 * j
            base_points.append((x, y))
    base_points = np.array(base_points)
    
    # Center the pattern at origin
    center = np.mean(base_points, axis=0)
    base_points -= center

    # 2. Function to calculate the minimal enclosing regular hexagon side length
    #    for a set of hexagons (centers + orientation) given an outer rotation.
    #    For simplicity, we assume inner hexagons are aligned with the outer hexagon 
    #    (relative rotation 0). This is a strong hypothesis for optimality in symmetric packs.
    #    If we rotate the *pattern* by theta, the inner hexagons are also rotated by theta.
    #    The outer hexagon is fixed at angle 0.
    #    Wait, the problem allows rotating the outer hexagon too. 
    #    But rotating the outer is equivalent to rotating the inner pattern relative to it.
    #    So we can fix the outer hexagon at angle 0 and search for the best rotation 
    #    of the inner pattern.
    
    def calc_min_outer_side(pattern_rotation_deg):
        """
        Given a rotation of the inner pattern, find the minimal side length S
        of a regular hexagon (centered at 0,0, angle 0) that contains all 12 hexagons.
        """
        theta = np.deg2rad(pattern_rotation_deg)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        
        # Rotate centers
        rotated_centers = (rot_mat @ base_points.T).T
        
        # For each inner hexagon, we need to check its vertices.
        # An inner hexagon of side 1, rotated by theta, has vertices at:
        # center + (cos(theta + k*60), sin(theta + k*60)) for k=0..5
        
        # To find the minimal S for the outer hexagon (angle 0), we need to satisfy:
        # For all vertices v of all inner hexagons:
        # max_k (v . n_k) <= S * sqrt(3) / 2
        # where n_k are the normals of the outer hexagon faces.
        # Outer hexagon normals (angle 0, pointy right? No, standard definition:
        # Vertices at 0, 60, ... -> Normals at 30, 90, ...
        # Let's stick to the definition:
        # Outer hexagon side S, center (0,0), angle 0.
        # Inradius r = S * sqrt(3) / 2.
        # The condition for a point (x,y) to be inside is:
        # |x cos(30) + y sin(30)| <= r
        # |x cos(90) + y sin(90)| <= r
        # |x cos(150) + y sin(150)| <= r
        # and their negatives.
        # Effectively: max( |dot(v, n1)|, |dot(v, n2)|, |dot(v, n3)| ) <= r
        # where n1=(cos30, sin30), n2=(cos90, sin90), n3=(cos150, sin150)
        
        n1 = np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
        n2 = np.array([np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))])
        n3 = np.array([np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))])
        normals = np.array([n1, n2, n3])
        
        max_proj = 0.0
        
        for cx, cy in rotated_centers:
            # Vertices of this inner hexagon
            for k in range(6):
                angle = theta + np.deg2rad(k * 60)
                vx = cx + np.cos(angle)
                vy = cy + np.sin(angle)
                v = np.array([vx, vy])
                
                # Project onto normals
                projs = np.abs(normals @ v)
                local_max = np.max(projs)
                if local_max > max_proj:
                    max_proj = local_max
        
        # max_proj is the required inradius r
        # S = 2 * r / sqrt(3)
        S = 2.0 * max_proj / sqrt3
        return S

    # 3. Search for optimal rotation
    # We can use a simple 1-D search (Golden Section or just fine grid + refine)
    # Since the pattern is symmetric (3x4 lattice is not fully symmetric, but close),
    # we expect a specific angle to be optimal.
    
    best_S = float('inf')
    best_theta = 0.0
    
    # Coarse search
    for theta in np.linspace(0, 60, 61):
        S = calc_min_outer_side(theta)
        if S < best_S:
            best_S = S
            best_theta = theta
            
    # Fine search around best
    if best_theta > 0 and best_theta < 60:
        for theta in np.linspace(best_theta - 1, best_theta + 1, 101):
            S = calc_min_outer_side(theta)
            if S < best_S:
                best_S = S
                best_theta = theta
    else:
        # Handle boundary
        for theta in np.linspace(0, 2, 101):
            S = calc_min_outer_side(theta)
            if S < best_S:
                best_S = S
                best_theta = theta
        for theta in np.linspace(58, 60, 101):
            S = calc_min_outer_side(theta)
            if S < best_S:
                best_S = S
                best_theta = theta

    # 4. Construct the final answer
    # We have the optimal rotation best_theta.
    # We need to output the inner hexagons (x, y, angle) and outer (center, side, angle).
    # Outer center is (0,0). Outer angle is 0.
    # Inner hexagons are at rotated_centers, with angle best_theta.
    
    theta_rad = np.deg2rad(best_theta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    rot_mat = np.array([[c, -s], [s, c]])
    final_centers = (rot_mat @ base_points.T).T
    
    inner = []
    for cx, cy in final_centers:
        inner.append([cx, cy, best_theta])
        
    # Return format:
    # inner: list of [x, y, angle_degrees]
    # center: [x, y]
    # side: S
    # angle: outer angle (0.0)
    
    return inner, [0.0, 0.0], best_S, 0.0

if __name__ == "__main__":
    construct()
