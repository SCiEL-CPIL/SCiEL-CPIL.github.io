import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    sqrt3 = np.sqrt(3)
    apothem = sqrt3 / 2.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def support_function(phi, hex_angle):
        """
        Computes the support function of a unit regular hexagon in direction phi.
        phi: angle of the direction vector.
        hex_angle: rotation angle of the hexagon.
        """
        # Relative angle between direction and hexagon normal
        # Hexagon normals are at hex_angle + k*pi/3
        # The support function is periodic with pi/3
        rel_angle = (phi - hex_angle) % (np.pi / 3)
        # Map to [-pi/6, pi/6]
        if rel_angle > np.pi / 6:
            rel_angle = rel_angle - np.pi / 3
        # The distance from center to edge is apothem / cos(rel_angle)
        # But wait, the normal to the edge is at hex_angle + k*pi/3.
        # The angle between phi and the nearest normal is delta.
        # delta = rel_angle mapped to [-pi/6, pi/6].
        # h(phi) = apothem / cos(delta)
        return apothem / np.cos(rel_angle)

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def objective(params):
        """Objective: minimize outer side length + collision penalty."""
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Compute collision penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ai = inner[i]
                xj, yj, aj = inner[j]
                
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Angle from i to j
                phi_ij = np.arctan2(dy, dx)
                
                # Support function of hex i in direction phi_ij
                # Hex i angle is ai (radians)
                h_i = support_function(phi_ij, np.radians(ai))
                
                # Support function of hex j in direction phi_ij + pi (from j to i)
                h_j = support_function(phi_ij + np.pi, np.radians(aj))
                
                d_req = h_i + h_j
                
                if dist < d_req:
                    penalty += (d_req - dist)**2
        
        if penalty > 1e-6:
            # If there is significant overlap, return a large value to push away
            # We add the penalty to the side length
            pass
        
        # Compute side
        side = compute_min_side_fast(inner, center, angle)
        
        # Return side length plus a weighted penalty
        # The penalty should be significant enough to prevent overlap
        return side + 100.0 * penalty

    def initial_guess():
        params = np.zeros(3*N + 3)
        positions = []
        # Hexagonal packing for 12 hexagons: 1 center, 6 ring 1, 5 ring 2
        positions.append((0, 0, 0))
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    best_params = None
    best_score = float('inf')
    
    # Run multiple restarts
    num_restarts = 50
    restarts_done = 0
    
    while time.time() < deadline - 10.0 and restarts_done < num_restarts:
        restarts_done += 1
        # Create a random initial guess
        params = initial_guess()
        # Add some random noise
        for i in range(len(params)):
            params[i] += random.gauss(0, 0.5)
        
        # Optimize
        try:
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
            if res.success:
                score = res.fun
                # Check validity
                # Extract inner
                inner = []
                for i in range(N):
                    inner.append((res.x[3*i], res.x[3*i+1], res.x[3*i+2]))
                center = (res.x[3*N], res.x[3*N+1])
                angle = res.x[3*N+2]
                
                # Verify disjointness strictly
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        xi, yi, ai = inner[i]
                        xj, yj, aj = inner[j]
                        dx = xj - xi
                        dy = yj - yi
                        dist = np.sqrt(dx*dx + dy*dy)
                        phi_ij = np.arctan2(dy, dx)
                        h_i = support_function(phi_ij, np.radians(ai))
                        h_j = support_function(phi_ij + np.pi, np.radians(aj))
                        d_req = h_i + h_j
                        if dist < d_req - 1e-9:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    side = compute_min_side_fast(inner, center, angle)
                    if side < best_score:
                        best_score = side
                        best_params = res.x.copy()
        except Exception:
            pass

    # If no valid solution found, use the initial guess
    if best_params is None:
        best_params = initial_guess()
        best_score = objective(best_params)

    # Final polish with very tight tolerance
    if time.time() < deadline - 5.0 and best_params is not None:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            if res.success:
                # Verify
                inner = []
                for i in range(N):
                    inner.append((res.x[3*i], res.x[3*i+1], res.x[3*i+2]))
                center = (res.x[3*N], res.x[3*N+1])
                angle = res.x[3*N+2]
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        xi, yi, ai = inner[i]
                        xj, yj, aj = inner[j]
                        dx = xj - xi
                        dy = yj - yi
                        dist = np.sqrt(dx*dx + dy*dy)
                        phi_ij = np.arctan2(dy, dx)
                        h_i = support_function(phi_ij, np.radians(ai))
                        h_j = support_function(phi_ij + np.pi, np.radians(aj))
                        d_req = h_i + h_j
                        if dist < d_req - 1e-9:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    side = compute_min_side_fast(inner, center, angle)
                    if side < best_score:
                        best_score = side
                        best_params = res.x.copy()
        except Exception:
            pass

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
