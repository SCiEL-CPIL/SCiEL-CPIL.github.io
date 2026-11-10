import numpy as np
from scipy.optimize import minimize_scalar
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    def get_hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
        """
        Returns vertices of a regular hexagon with center (cx, cy), side 'side', 
        and rotation 'angle_deg'.
        Vertices are at angles angle + 60*i.
        """
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.deg2rad(np.arange(6) * 60)
        xs = cx + side * np.cos(angles)
        ys = cy + side * np.sin(angles)
        return np.column_stack((xs, ys))

    def generate_topology(name):
        """
        Generates a list of (x, y) centers for 12 unit hexagons.
        Uses axial coordinates converted to Cartesian.
        d = sqrt(3) is the distance between centers of edge-touching hexagons.
        """
        d = np.sqrt(3)
        centers = []
        
        def axial_to_cartesian(q, r):
            x = d * (q + r / 2.0)
            y = d * (np.sqrt(3) / 2.0 * r)
            return (x, y)

        if name == "T1_4x3":
            # 4 columns, 3 rows
            for r in range(3):
                for q in range(4):
                    centers.append(axial_to_cartesian(q, r))
        elif name == "T2_3x4":
            # 3 columns, 4 rows
            for r in range(4):
                for q in range(3):
                    centers.append(axial_to_cartesian(q, r))
        elif name == "T3_3432":
            # Row 0: 3, Row 1: 4, Row 2: 3, Row 3: 2
            rows = [3, 4, 3, 2]
            for r, count in enumerate(rows):
                for q in range(count):
                    centers.append(axial_to_cartesian(q, r))
        elif name == "T4_2343":
            # Row 0: 2, Row 1: 3, Row 2: 4, Row 3: 3
            rows = [2, 3, 4, 3]
            for r, count in enumerate(rows):
                for q in range(count):
                    centers.append(axial_to_cartesian(q, r))
        elif name == "T5_1344":
            # Row 0: 1, Row 1: 3, Row 2: 4, Row 3: 4
            rows = [1, 3, 4, 4]
            for r, count in enumerate(rows):
                for q in range(count):
                    centers.append(axial_to_cartesian(q, r))
        else:
            # Default fallback: 4x3
            for r in range(3):
                for q in range(4):
                    centers.append(axial_to_cartesian(q, r))
            
        return np.array(centers)

    def min_enclosing_hex_side(centers, theta_deg):
        """
        Calculates the minimum side length s of a regular hexagon with orientation 
        theta_deg that contains all inner hexagons centered at 'centers'.
        """
        # Get all vertices of all inner hexagons
        all_vertices = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices) # Shape (72, 2)
        
        # Normals for the 6 edges of the outer hexagon
        # Edge normals are perpendicular to edges.
        # If outer hexagon has angle theta, its vertices are at theta + 60*i.
        # The edges are between vertices.
        # The normal to the edge between vertex i and i+1 is at angle theta + 60*i + 30.
        # Let's verify.
        # Vertex 0: (1, 0) rotated by theta.
        # Vertex 1: (1/2, sqrt(3)/2) rotated by theta.
        # Midpoint of edge 0-1: ...
        # Normal direction is theta + 30 + 60*i.
        
        theta_rad = np.deg2rad(theta_deg)
        normal_angles = theta_rad + np.deg2rad(30 + np.arange(6) * 60)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles))) # (6, 2)
        
        # Project all vertices onto each normal
        # projections shape: (72, 6)
        projections = all_vertices @ normals.T
        
        # Width in each normal direction
        widths = np.max(projections, axis=0) - np.min(projections, axis=0)
        
        # The apothem of the outer hexagon is s * sqrt(3) / 2
        # The width of the cluster must be <= 2 * apothem = s * sqrt(3)
        # So s >= width / sqrt(3)
        # We need s to be large enough for ALL 6 directions
        max_width = np.max(widths)
        s = max_width / np.sqrt(3)
        
        return s

    def find_optimal_for_topology(centers):
        """
        Finds the optimal orientation theta and corresponding s, center for a given topology.
        """
        # Optimize theta in [0, 60)
        result = minimize_scalar(min_enclosing_hex_side, args=(centers,), bounds=(0, 60), method='bounded', options={'xatol': 1e-6})
        
        if not result.success:
            # Fallback to grid search if optimization fails
            thetas = np.linspace(0, 60, 100)
            s_vals = [min_enclosing_hex_side(centers, t) for t in thetas]
            best_idx = np.argmin(s_vals)
            best_theta = thetas[best_idx]
            best_s = s_vals[best_idx]
        else:
            best_theta = result.x
            best_s = result.fun
            
        # Calculate the optimal center for this theta and s
        # We need C such that for all k, C . n_k is within [min_v + s*sqrt(3)/2, max_v - s*sqrt(3)/2]
        # Ideally, C . n_k = (min_v + max_v) / 2
        
        all_vertices = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        theta_rad = np.deg2rad(best_theta)
        normal_angles = theta_rad + np.deg2rad(30 + np.arange(6) * 60)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        projections = all_vertices @ normals.T
        mins = np.min(projections, axis=0)
        maxs = np.max(projections, axis=0)
        centers_proj = (mins + maxs) / 2.0
        
        # Solve for C = (Cx, Cy)
        # We have 6 equations C . n_k = centers_proj[k]
        # Use least squares on the first 3 independent ones (or all 6)
        A = normals[:3] # (3, 2)
        b = centers_proj[:3]
        C, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
        
        return best_theta, best_s, C

    # List of topologies to test
    topologies = ["T1_4x3", "T2_3x4", "T3_3432", "T4_2343", "T5_1344"]
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    best_theta = None
    
    for t_name in topologies:
        if time.time() > deadline - safety_margin:
            break
            
        centers = generate_topology(t_name)
        
        try:
            theta, s, C = find_optimal_for_topology(centers)
            
            if s < best_s:
                best_s = s
                best_theta = theta
                best_C = C
                best_centers = centers
        except Exception as e:
            pass # Skip failed topology

    if best_inner is None and best_s < float('inf'):
        # Construct the final result
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i, 0] = best_centers[i, 0]
            inner_hex_data[i, 1] = best_centers[i, 1]
            inner_hex_data[i, 2] = 0.0 # Angle
            
        outer_hex_data = np.array([best_C[0], best_C[1], best_theta])
        
        return inner_hex_data, outer_hex_data, best_s

    # Fallback if all failed (should not happen)
    inner_hex_data = np.array([[0,0,0]] * 12)
    outer_hex_data = np.array([0, 0, 0])
    return inner_hex_data, outer_hex_data, 10.0

# EVOLVE-BLOCK-END
