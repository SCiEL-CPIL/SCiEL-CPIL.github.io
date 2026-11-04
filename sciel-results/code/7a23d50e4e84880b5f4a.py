import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Define 11 hexagon centers in local coordinates
    # Using a triangular lattice with spacing sqrt(3)
    d = np.sqrt(3.0)
    
    # 7 central hexagons
    centers = [
        (0.0, 0.0),
        (d, 0.0),
        (-d, 0.0),
        (d/2.0, 1.5),
        (-d/2.0, 1.5),
        (d/2.0, -1.5),
        (-d/2.0, -1.5)
    ]
    
    # 4 additional hexagons to make 11
    # Added to make a more compact, roughly hexagonal shape
    centers.append((1.5*d, 1.5))
    centers.append((-1.5*d, -1.5))
    centers.append((1.5*d, -1.5))
    centers.append((-1.5*d, 1.5))
    
    centers = np.array(centers)
    
    # Normals for the 3 independent directions of the outer hexagon
    # Angles 0, 60, 120
    angles = np.array([0.0, np.pi/3.0, 2.0*np.pi/3.0])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    def objective(theta):
        # Rotate centers
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated = centers @ R.T
        
        # Project onto normals
        # projections shape: (11, 3)
        projections = rotated @ normals.T
        
        # Spans for each of the 3 directions
        spans = projections.max(axis=0) - projections.min(axis=0)
        
        return np.max(spans)
    
    # Optimize theta in [0, pi/3)
    # Use a simple grid search first to get a good start, then refine?
    # Or just use DE.
    
    best_theta = 0.0
    best_val = objective(0.0)
    
    # Simple grid search for robustness and speed
    # 360 steps is fast
    thetas = np.linspace(0, np.pi/3.0, 361, endpoint=False)
    for th in thetas:
        val = objective(th)
        if val < best_val:
            best_val = val
            best_theta = th
            
    # Refine with a few local steps around best_theta
    # Using a small Nelder-Mead or just finer grid
    if time.time() < deadline - 5.0:
        fine_thetas = np.linspace(best_theta - 0.01, best_theta + 0.01, 100)
        for th in fine_thetas:
            val = objective(th)
            if val < best_val:
                best_val = val
                best_theta = th
                
    # Calculate L and T
    c = np.cos(best_theta)
    s = np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    rotated = centers @ R.T
    
    projections = rotated @ normals.T
    max_proj = projections.max(axis=0)
    min_proj = projections.min(axis=0)
    spans = max_proj - min_proj
    
    L = np.max(spans) / np.sqrt(3.0)
    
    # Find T
    # Constraints:
    # T . n_k in [ -L*sqrt(3)/2 - min_proj[k], L*sqrt(3)/2 - max_proj[k] ]
    
    # Let's find the intersection of the 3 strips.
    # We can solve this by checking the vertices of the potential polygon.
    # Or simpler: The optimal T is the center of the bounding box of the cluster in the hexagonal metric.
    # A robust heuristic for T:
    # The center of the feasible region.
    # We can compute the 6 lines:
    # x . n_k = -L*sqrt(3)/2 - min_proj[k]  (lower bound)
    # x . n_k = L*sqrt(3)/2 - max_proj[k]   (upper bound)
    
    # Let's just use the centroid of the 6 intersection points of the pairs of opposite lines?
    # No, the feasible region is the intersection of the 6 half-planes.
    # We can find the vertices of this polygon.
    
    # Lines:
    # n_0 . x = b_0_low, n_0 . x = b_0_high
    # n_1 . x = b_1_low, n_1 . x = b_1_high
    # n_2 . x = b_2_low, n_2 . x = b_2_high
    
    b_lows = -L * np.sqrt(3.0) / 2.0 - min_proj
    b_highs = L * np.sqrt(3.0) / 2.0 - max_proj
    
    # Find vertices of the intersection
    # There are at most 6 vertices.
    # Each vertex is the intersection of two lines.
    # We test all pairs of lines (12 lines total, 66 pairs).
    vertices = []
    lines = []
    for k in range(3):
        lines.append((normals[k], b_lows[k]))
        lines.append((normals[k], b_highs[k]))
        
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            n1, b1 = lines[i]
            n2, b2 = lines[j]
            # Solve n1 . x = b1, n2 . x = b2
            # [n1_x n1_y; n2_x n2_y] [x; y] = [b1; b2]
            A = np.array([n1, n2])
            b = np.array([b1, b2])
            det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
            if abs(det) > 1e-10:
                x = (b[0]*A[1,1] - b[1]*A[0,1]) / det
                y = (A[0,0]*b[1] - A[1,0]*b[0]) / det
                pt = np.array([x, y])
                # Check if pt satisfies all constraints
                valid = True
                for k in range(3):
                    val = pt @ normals[k]
                    if val < b_lows[k] - 1e-9 or val > b_highs[k] + 1e-9:
                        valid = False
                        break
                if valid:
                    vertices.append(pt)
                    
    if vertices:
        T = np.mean(vertices, axis=0)
    else:
        # Fallback to centroid of rotated centers
        T = np.mean(rotated, axis=0)
        
    # Construct inner list
    # inner rows: (x, y, angle_degrees)
    # The angle of each inner hexagon is the global rotation best_theta
    angle_deg = np.degrees(best_theta)
    
    inner = []
    for i in range(N):
        # The center of the i-th hexagon in the global frame
        # rotated[i] is the local center rotated.
        # We need to add T to get the global center?
        # Wait, the objective function assumed the outer hexagon is centered at 0.
        # The condition was: max_i ((rotated_i + T) . n_k) <= ...
        # So the global center of the i-th hexagon is rotated_i + T.
        gx = rotated[i, 0] + T[0]
        gy = rotated[i, 1] + T[1]
        inner.append([gx, gy, angle_deg])
        
    # Outer hexagon center is T?
    # No, the outer hexagon is centered at 0 in our derivation?
    # Let's re-read:
    # "outer hexagon center (x, y)"
    # In my derivation, I assumed the outer hexagon is at 0, and moved the inner ones.
    # So the outer hexagon center is (0,0) and the inner ones are at rotated_i + T.
    # OR, I can say the outer hexagon is at T, and the inner ones are at rotated_i.
    # It doesn't matter for the score, as long as they are disjoint and inside.
    # Let's return outer center as (0,0) and inner as calculated.
    # Actually, returning outer center as T and inner as rotated_i is also valid.
    # Let's stick to outer center (0,0) for simplicity in the return, 
    # but wait, the problem asks for "outer hexagon center (x, y)".
    # If I return (0,0), the inner hexagons are at rotated_i + T.
    # This is a valid configuration.
    
    return inner, (0.0, 0.0), L, 0.0

# EVOLVE-BLOCK-END
