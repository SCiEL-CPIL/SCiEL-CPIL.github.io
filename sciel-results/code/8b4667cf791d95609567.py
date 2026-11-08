import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Helper: Generate vertices of a regular hexagon
    def hex_verts(cx, cy, s, rot_deg):
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    # Helper: Check if point is inside a convex polygon (using cross products)
    def point_in_poly(pt, poly):
        # poly is (6, 2)
        sign = 0
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
            if cross > 1e-9:
                if sign < 0:
                    return False
                sign = 1
            elif cross < -1e-9:
                if sign > 0:
                    return False
                sign = -1
        return True

    # Helper: Minimum distance between two convex polygons
    # Returns 0 if they intersect, else minimum distance between boundaries
    def min_dist_polygons(v1, v2):
        # First check for intersection
        # If any vertex of v1 is inside v2, they intersect
        for v in v1:
            if point_in_poly(v, v2):
                return 0.0
        for v in v2:
            if point_in_poly(v, v1):
                return 0.0
                
        # Check segment intersections
        def segments_intersect(p1, p2, p3, p4):
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j + 1) % 6]
                if segments_intersect(p1, p2, p3, p4):
                    return 0.0

        # If no intersection, calculate min distance
        min_d = float('inf')
        # Vertex of v1 to edges of v2
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j + 1) % 6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1 / c2) * v)
                if d < min_d:
                    min_d = d
        # Vertex of v2 to edges of v1
        for i in range(6):
            p1 = v2[i]
            for j in range(6):
                p2 = v1[j]
                p3 = v1[(j + 1) % 6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1 / c2) * v)
                if d < min_d:
                    min_d = d
        return min_d

    def get_energy_and_forces(params, outer_s):
        """
        params: [x1, y1, a1, ..., x12, y12, a12]
        Returns (energy, forces)
        Forces correspond to parameters: [fx1, fy1, fa1, ...]
        """
        energy = 0.0
        forces = np.zeros(36)
        
        # Extract positions and angles
        xs = params[0::3]
        ys = params[1::3]
        as_ = params[2::3]
        
        # Precompute vertices
        verts = [hex_verts(xs[i], ys[i], 1.0, as_[i]) for i in range(N)]
        
        # Outer hexagon containment
        # Outer hexagon: center (0,0), side outer_s, angle 0
        # Inradius r = outer_s * sqrt(3) / 2
        r = outer_s * np.sqrt(3) / 2.0
        # Normals for outer hexagon (perpendicular to edges)
        # Edges are at 30, 90, 150, 210, 270, 330 degrees from center?
        # Vertices at 0, 60, 120...
        # Edge 1 connects v0 and v1. Midpoint angle 30. Normal is 30.
        normals = np.deg2rad(30 + np.arange(6) * 60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        for i in range(N):
            v = verts[i]
            # Distance of each vertex from origin along normals
            dists = np.dot(v, n_vecs.T)
            # If dist > r, it's outside
            out = np.maximum(0, dists - r)
            if np.any(out > 0):
                energy += np.sum(out ** 2) * 10.0 # Strong penalty for containment
                # Force: push back inside
                # Gradient of out^2 is 2*out * n
                # Sum over vertices and normals
                # This is complex to distribute to x, y, a.
                # Approximation: Force is towards center for vertices that are outside
                # Or simply: if center is too close to edge, push center.
                # For simplicity and stability in physics sim, let's compute force on center
                # based on the most violated normal.
                max_out_idx = np.argmax(out)
                if out[max_out_idx] > 0:
                    n = n_vecs[max_out_idx]
                    # Push center in direction -n
                    # Scale by violation
                    f_mag = out[max_out_idx] * 10.0
                    forces[3*i] -= n[0] * f_mag
                    forces[3*i+1] -= n[1] * f_mag
                    # Rotation force is harder to derive analytically here without full gradient
                    # We can add a small random rotation force or ignore it for containment
                    # Actually, rotating might help fit corners. 
                    # But for now, center translation is the main fix.
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d = min_dist_polygons(verts[i], verts[j])
                if d < 0.5: # Interaction range
                    # Force magnitude increases as distance decreases
                    # F = k * (1/d - 1/d0) if d < d0
                    d0 = 0.5
                    k = 10.0
                    f_mag = k * (1.0 / max(d, 1e-6) - 1.0 / d0)
                    if f_mag < 0:
                        f_mag = 0
                    
                    # Direction: vector from center i to center j
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist_centers = math.sqrt(dx*dx + dy*dy)
                    if dist_centers < 1e-6:
                        # Random direction if coincident
                        ang = np.random.uniform(0, 2*np.pi)
                        dx, dy = np.cos(ang), np.sin(ang)
                        dist_centers = 1.0
                    else:
                        dx /= dist_centers
                        dy /= dist_centers
                    
                    # Apply force: i pushed away from j, j pushed away from i
                    forces[3*i] -= dx * f_mag
                    forces[3*i+1] -= dy * f_mag
                    forces[3*j] += dx * f_mag
                    forces[3*j+1] += dy * f_mag
                    
                    # Rotation force: try to untangle
                    # Simple heuristic: if they are close, add small random torque
                    # Or estimate based on relative orientation
                    # For now, small random torque to escape local minima
                    forces[3*i+2] += np.random.normal(0, 0.1) * f_mag
                    forces[3*j+2] += np.random.normal(0, 0.1) * f_mag
                    
                    # Energy contribution
                    energy += 0.5 * k * (1.0 / max(d, 1e-6) - 1.0 / d0) ** 2

        return energy, forces

    def solve_for_s(outer_s):
        """
        Run physics simulation to find valid configuration for given outer_s.
        Returns (success, best_params)
        """
        best_params = None
        best_energy = float('inf')
        
        # Initial random positions within a small circle
        # Avoid placing them too close to edges initially
        init_radius = outer_s * 0.5
        x0 = np.zeros(36)
        for i in range(N):
            ang = np.random.uniform(0, 2*np.pi)
            rad = np.random.uniform(0, init_radius)
            x0[3*i] = rad * np.cos(ang)
            x0[3*i+1] = rad * np.sin(ang)
            x0[3*i+2] = np.random.uniform(0, 60)
            
        params = x0.copy()
        
        # Simulated Annealing / Physics Relaxation
        T = 1.0
        T_min = 0.01
        cooling = 0.95
        max_iter = 5000
        
        current_energy, _ = get_energy_and_forces(params, outer_s)
        
        for it in range(max_iter):
            if time.time() > deadline - 5: # Check time
                break
                
            # Compute forces
            _, forces = get_energy_and_forces(params, outer_s)
            
            # Update velocities (implicit in this simple scheme, or we can keep vel)
            # Let's use a simple gradient descent with noise (Langevin dynamics)
            # params_new = params + forces * dt + noise
            dt = 0.05
            noise_scale = T * 0.1
            new_params = params + forces * dt + np.random.normal(0, noise_scale, 36)
            
            # Keep angles bounded? Not strictly necessary, but can wrap
            # Wrap angles to [0, 360)
            new_params[2::3] = new_params[2::3] % 360
            
            new_energy, _ = get_energy_and_forces(new_params, outer_s)
            
            # Metropolis acceptance
            if new_energy < current_energy:
                params = new_params
                current_energy = new_energy
                if current_energy < 1e-6:
                    # Converged to a very low energy state
                    # Verify validity
                    # (Energy < 1e-6 implies containment and no overlap with our penalties)
                    return True, params
            else:
                # Accept with probability exp(-(E_new - E_old) / T)
                if np.random.rand() < np.exp(-(new_energy - current_energy) / T):
                    params = new_params
                    current_energy = new_energy
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        # Check if final state is valid
        if current_energy < 1e-6:
            return True, params
            
        return False, params

    # Binary search on outer side length
    LB = 3.0
    UB = 15.0
    
    best_solution = None
    best_s = float('inf')
    
    # We have ~140s.
    # Each solve_for_s might take 1-5s.
    # Let's do ~20 binary search steps.
    
    for step in range(25):
        if time.time() > deadline:
            break
            
        mid_s = (LB + UB) / 2.0
        
        # Try to find a valid configuration for this S
        # Use a few restarts if first fails? 
        # Physics sim is stochastic, so one run is decent. 
        # But for robustness, maybe 2 runs if time permits.
        found_valid = False
        best_params = None
        
        for attempt in range(2):
            if time.time() > deadline:
                break
            success, params = solve_for_s(mid_s)
            if success:
                found_valid = True
                best_params = params
                break
        
        if found_valid:
            UB = mid_s
            best_solution = best_params
            best_s = mid_s
        else:
            LB = mid_s

    if best_solution is None:
        # Fallback: Return a large valid box
        # Just place them in a grid with large spacing
        inner = []
        for i in range(12):
            x = (i % 4) * 2.5 - 3.75
            y = (i // 4) * 2.2 - 2.2
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 15.0, 0.0

    # Extract best solution
    xs = best_solution[0::3]
    ys = best_solution[1::3]
    as_ = best_solution[2::3]
    
    inner = []
    for i in range(N):
        inner.append([xs[i], ys[i], as_[i]])
        
    return inner, (0.0, 0.0), best_s, 0.0

