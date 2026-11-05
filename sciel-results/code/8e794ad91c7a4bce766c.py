import numpy as np
import time
import itertools
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    eps = 1e-12
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    # Iterate all 15 combinations of 4 outer hexagons from 6 possible positions
    all_angles = [0, 60, 120, 180, 240, 300]
    combinations = list(itertools.combinations(all_angles, 4))
    
    best_score = float('inf')
    best_theta_rel = 0.0
    best_centers = None
    best_config = None

    for combo in combinations:
        # Build centers
        centers = np.array([[0.0, 0.0]])
        
        # Inner ring (6) at 30, 90, ...
        # Distance sqrt(3) is the distance between centers of two touching hexagons
        # if they are aligned with flat sides touching?
        # Let's verify:
        # Hexagon side 1. Apothem sqrt(3)/2.
        # If two hexagons touch flat sides, distance between centers = 2 * apothem = sqrt(3).
        # The normal to the flat side is at 30, 90, ... degrees.
        # So placing centers at distance sqrt(3) along these normals works.
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        
        # Outer ring (4) at chosen angles
        # The gaps between inner ring hexagons are at 0, 60, 120, ...
        # Distance from center to the "corner" of the cluster.
        # The inner ring hexagon at 30 deg has vertices.
        # The outer hexagon should touch the inner ring.
        # Distance 3.0 is a standard lattice distance for next shell in hex packing?
        # In a perfect hex lattice, distance from center to 2nd shell is 2 * (distance to 1st shell)?
        # No, 1st shell is sqrt(3). 2nd shell in a hex lattice is 2*sqrt(3) ~ 3.46?
        # Or is it 3.0?
        # Let's check the distance between center of inner ring hex (at r1, 30deg) and outer ring hex (at r2, 0deg).
        # They should touch.
        # Distance squared = r1^2 + r2^2 - 2 r1 r2 cos(30).
        # If they touch, distance = sqrt(3).
        # 3 + r2^2 - 2 * sqrt(3) * r2 * (sqrt(3)/2) = 3
        # 3 + r2^2 - 3 r2 = 3
        # r2^2 - 3 r2 = 0 => r2 = 3.
        # Yes, r2 = 3.0 is correct for touching the inner ring hexagons.
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
        n_hex = len(centers)
        
        def compute_side_rigid(theta_rel):
            """
            Compute minimum outer hexagon side length for a given relative rotation.
            theta_rel is the rotation of the outer hexagon relative to the inner cluster.
            """
            # Outer hexagon face normals
            # For a regular hexagon with side S, the apothem is S*sqrt(3)/2.
            # The normals point to the midpoints of the sides.
            # If the hexagon is rotated by theta_out, the normals are at 30 + k*60 + theta_out.
            angles = np.deg2rad(30 + np.arange(6) * 60) + theta_rel
            normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
            
            max_proj = 0.0
            # Inner hexagons are all at rotation 0.
            # So their vertices are just base_vertices + centers[i]
            for i in range(n_hex):
                verts = base_vertices + centers[i]  # (6, 2)
                dots = verts @ normals.T  # (6, 6)
                # We need the maximum projection of any vertex onto any normal direction.
                # Since the hexagon is centered at origin, we check abs(dots) to cover all directions?
                # No, the outer hexagon contains the cluster. The cluster is near origin.
                # The projection of a vertex v onto normal n is v . n.
                # The constraint is that for all vertices v and all normals n,
                # the distance from origin to the face perpendicular to n must be >= v . n.
                # Actually, the outer hexagon is defined by |x . n_k| <= apothem.
                # So we need max_i,v | (c_i + v) . n_k | <= apothem.
                max_proj = max(max_proj, np.max(np.abs(dots)))
            
            # apothem = S * sqrt(3) / 2  => S = 2 * apothem / sqrt(3)
            return 2.0 * max_proj / np.sqrt(3)

        # Coarse grid search
        # The function is periodic with period pi/3 (60 deg) due to hexagon symmetry?
        # Actually, the cluster is not 6-fold symmetric if we pick 4 outer hexagons.
        # But the outer hexagon is 6-fold symmetric.
        # So we only need to search theta in [0, pi/6] (30 deg) because of the symmetry of the outer hexagon normals?
        # Wait, if we rotate the outer hexagon by 60 deg, it looks the same.
        # So theta_rel is defined mod pi/3.
        # Also, theta_rel = 0 and theta_rel = pi/6 might be symmetric depending on the cluster.
        # Let's search [0, pi/6].
        
        # Check time
        if time.time() > deadline - 10:
            break

        thetas = np.linspace(0, np.pi/6, 200)
        local_best_score = float('inf')
        local_best_theta = 0.0
        
        for t in thetas:
            s = compute_side_rigid(t)
            if s < local_best_score:
                local_best_score = s
                local_best_theta = t
        
        # Refine using minimize_scalar
        if time.time() < deadline - 5:
            step = (np.pi/6) / 200
            # Search in a small window around the best grid point
            bounds = (max(0, local_best_theta - 5*step), min(np.pi/6, local_best_theta + 5*step))
            try:
                res = minimize_scalar(compute_side_rigid, bounds=bounds, method='bounded', 
                                      options={'xatol': 1e-12, 'maxiter': 100})
                if res.fun < local_best_score:
                    local_best_score = res.fun
                    local_best_theta = res.x
            except Exception:
                pass

        if local_best_score < best_score:
            best_score = local_best_score
            best_theta_rel = local_best_theta
            best_centers = centers
            best_config = combo

    # Fallback if best_centers is still None
    if best_centers is None:
        combo = combinations[0]
        centers = np.array([[0.0, 0.0]])
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
        def compute_side_rigid(theta_rel):
            angles = np.deg2rad(30 + np.arange(6) * 60) + theta_rel
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            max_proj = 0.0
            for i in range(len(centers)):
                verts = base_vertices + centers[i]
                dots = verts @ normals.T
                max_proj = max(max_proj, np.max(np.abs(dots)))
            return 2.0 * max_proj / np.sqrt(3)
            
        best_score = compute_side_rigid(0.0)
        best_theta_rel = 0.0
        best_centers = centers
        best_config = combo

    # Add a tiny margin to the side length to ensure validity
    # The evaluator might check strict inequality or have floating point issues.
    # 1e-9 is negligible for the score but ensures safety.
    side = best_score + 1e-9

    inner = []
    for i in range(len(best_centers)):
        inner.append([best_centers[i][0], best_centers[i][1], 0.0])

    theta_out_deg = np.rad2deg(best_theta_rel)

    return inner, (0.0, 0.0), side, theta_out_deg

if __name__ == "__main__":
    construct()
