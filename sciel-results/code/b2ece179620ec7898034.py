"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill

    # Unit regular hexagon vertices (flat-top orientation, angle=0)
    # For a regular hexagon with side length 1, the vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # are at distance 1 from center.
    # Vertex i: (cos(i*pi/3), sin(i*pi/3)) for i=0..5
    hex_vertices = np.array([[np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)] for i in range(6)])

    def rotate_vertices(angle_deg):
        """Rotate hexagon vertices by given angle in degrees."""
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        return hex_vertices @ R.T

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px, py) is inside regular hexagon centered at (cx, cy) with given side and angle."""
        # Transform point to hexagon's local frame
        dx = px - cx
        dy = py - cy
        rad = np.radians(-angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        lx = c * dx + s * dy
        ly = -s * dx + c * dy

        # For a regular hexagon with side length 'side', the apothem is side * sqrt(3)/2
        # The hexagon can be described by 6 half-planes.
        # In local frame, the constraints are:
        # |lx| <= side
        # |ly| <= side * sqrt(3)/2
        # |lx * sqrt(3)/2 + ly / 2| <= side * sqrt(3)/2  (for the other two pairs)
        # Actually, let's use the standard form. A regular hexagon centered at origin with side s
        # has vertices at (s, 0), (s/2, s*sqrt(3)/2), etc.
        # The six half-plane constraints (normal vectors pointing inward) are:
        # 1. x <= s
        # 2. -x <= s
        # 3. sqrt(3)/2 * x + 1/2 * y <= s * sqrt(3)/2
        # 4. -sqrt(3)/2 * x + 1/2 * y <= s * sqrt(3)/2
        # 5. -sqrt(3)/2 * x - 1/2 * y <= s * sqrt(3)/2
        # 6. sqrt(3)/2 * x - 1/2 * y <= s * sqrt(3)/2

        s3 = np.sqrt(3)
        a = side
        # Compute the maximum violation
        v1 = lx / a
        v2 = -lx / a
        v3 = (s3/2 * lx + 0.5 * ly) / (a * s3 / 2)
        v4 = (-s3/2 * lx + 0.5 * ly) / (a * s3 / 2)
        v5 = (-s3/2 * lx - 0.5 * ly) / (a * s3 / 2)
        v6 = (s3/2 * lx - 0.5 * ly) / (a * s3 / 2)

        # If all <= 1 (with small tolerance), point is inside
        return max(v1, v2, v3, v4, v5, v6) <= 1.0 + 1e-9

    def outer_side_length(inner_positions, inner_angles, outer_center, outer_side, outer_angle):
        """
        Compute the minimum outer side length needed to contain all inner hexagons.
        inner_positions: (N, 2) array
        inner_angles: (N,) array in degrees
        outer_center: (2,)
        outer_side: scalar (not used directly, we compute it)
        outer_angle: scalar in degrees
        Returns: the required outer side length.
        """
        # For each inner hexagon, find the maximum distance from outer_center to any of its vertices,
        # but we need the side length of the containing hexagon.
        
        # A regular hexagon with side S and center C contains point P iff P is within distance
        # such that... Actually, the condition for a point to be inside a regular hexagon of side S
        # centered at C with rotation theta is more complex.
        
        # Alternative approach: For a given outer hexagon (center, side, angle), check if all inner
        # hexagon vertices are inside. We want to minimize the outer side.
        
        # The outer hexagon has 6 sides. Each side is a line. A point is inside if it's on the
        # correct side of all 6 lines.
        
        # Let's compute the required side length by checking each inner vertex against the
        # outer hexagon's half-planes.
        
        # Outer hexagon vertices (for reference):
        # The outer hexagon with side S, center (cx, cy), angle theta has vertices at:
        # (cx + S*cos(theta + i*pi/3), cy + S*sin(theta + i*pi/3)) for i=0..5
        
        # The 6 half-plane constraints for the outer hexagon (in local frame, then rotated):
        # In local frame (unrotated), the constraints are:
        # x <= S
        # -x <= S
        # sqrt(3)/2 * x + 1/2 * y <= S * sqrt(3)/2
        # -sqrt(3)/2 * x + 1/2 * y <= S * sqrt(3)/2
        # -sqrt(3)/2 * x - 1/2 * y <= S * sqrt(3)/2
        # sqrt(3)/2 * x - 1/2 * y <= S * sqrt(3)/2
        
        # For a point P, transform to outer hexagon's local frame:
        # dx = Px - cx, dy = Py - cy
        # Rotate by -theta to get local coords
        # Then the required S must satisfy all 6 inequalities.
        
        # From each inequality, we get a lower bound on S.
        # x_local <= S  =>  S >= x_local
        # -x_local <= S  =>  S >= -x_local
        # sqrt(3)/2 * x_local + 1/2 * y_local <= S * sqrt(3)/2  =>  S >= (sqrt(3)/2 * x_local + 1/2 * y_local) / (sqrt(3)/2)
        # etc.
        
        theta_rad = np.radians(outer_angle)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        
        # Transform all points to outer hexagon's local frame
        # We'll compute the required S for each point and take the max.
        
        # For efficiency, let's vectorize.
        # Get all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            cx, cy = inner_positions[i]
            angle = inner_angles[i]
            verts = rotate_vertices(angle) + np.array([cx, cy])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)  # (6N, 2)
        
        # Transform to outer local frame
        dx = all_verts[:, 0] - outer_center[0]
        dy = all_verts[:, 1] - outer_center[1]
        
        # Rotate by -theta
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        
        # Compute the 6 constraints
        s3 = np.sqrt(3)
        bound1 = lx
        bound2 = -lx
        bound3 = (s3/2 * lx + 0.5 * ly) / (s3/2)
        bound4 = (-s3/2 * lx + 0.5 * ly) / (s3/2)
        bound5 = (-s3/2 * lx - 0.5 * ly) / (s3/2)
        bound6 = (s3/2 * lx - 0.5 * ly) / (s3/2)
        
        # Required S is the max of all these bounds
        required_S = max(
            np.max(bound1), np.max(bound2),
            np.max(bound3), np.max(bound4),
            np.max(bound5), np.max(bound6)
        )
        
        # S must be non-negative
        return max(0.0, required_S)

    # Objective function: outer side length
    def objective(params):
        """
        params: array of length 3*N + 2
        First N*3: x, y, angle for each inner hexagon
        Last 2: outer center x, y
        Outer angle is fixed at 0 for simplicity (or we can include it)
        """
        inner_positions = params[:N*3].reshape(N, 3)[:, :2]
        inner_angles = params[:N*3].reshape(N, 3)[:, 2]
        outer_center = params[N*3:]
        outer_angle = 0.0  # Fix outer angle to 0
        
        return outer_side_length(inner_positions, inner_angles, outer_center, 0, outer_angle)

    # Initial guess: place hexagons in a honeycomb-like pattern
    def initial_guess():
        """Generate an initial configuration."""
        params = np.zeros(N * 3 + 2)
        # Place in a rough grid
        # For 12 hexagons, maybe a 3x4 or similar arrangement
        # Honeycomb packing: each hex has 6 neighbors
        # Let's try a simple linear arrangement first, then optimize
        for i in range(N):
            # Rough placement
            params[i*3] = i * 2.0  # x
            params[i*3+1] = 0.0    # y
            params[i*3+2] = 0.0    # angle
        # Outer center
        params[N*3] = (N-1)  # center of the line
        params[N*3+1] = 0.0
        return params

    # Run optimization
    best_params = None
    best_score = np.inf

    # Try multiple initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 10:
            break
            
        if restart == 0:
            params0 = initial_guess()
        else:
            # Perturb the initial guess
            params0 = initial_guess() + np.random.randn(N * 3 + 2) * 1.0
            params0[N*3+2] = 0.0  # Keep outer angle fixed
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception as e:
            pass

    if best_params is None:
        # Fallback: return the initial guess
        best_params = initial_guess()
        best_score = objective(best_params)

    # Extract results
    inner_positions = best_params[:N*3].reshape(N, 3)[:, :2]
    inner_angles = best_params[:N*3].reshape(N, 3)[:, 2]
    outer_center = best_params[N*3:]
    outer_side = best_score
    outer_angle = 0.0

    # Format inner as list of [x, y, angle_degrees]
    inner = [[float(inner_positions[i][0]), float(inner_positions[i][1]), float(inner_angles[i])] for i in range(N)]

    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
# EVOLVE-BLOCK-END
