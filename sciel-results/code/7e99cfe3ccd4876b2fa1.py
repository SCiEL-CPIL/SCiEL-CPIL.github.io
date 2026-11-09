import numpy as np
import math
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    sqrt3 = math.sqrt(3)
    
    # --- Geometry Helpers ---
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = np.zeros((6, 2))
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts[k, 0] = cx + side * math.cos(theta)
            verts[k, 1] = cy + side * math.sin(theta)
        return verts

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point is inside hexagon."""
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        apothem = side * sqrt3 / 2
        for k in range(6):
            theta = math.radians(30 + k * 60)
            nx = math.cos(theta)
            ny = math.sin(theta)
            if rx * nx + ry * ny > apothem + 1e-9:
                return False
        return True

    def get_axes(verts):
        """Get edge normals for SAT."""
        axes = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            # Normal is perpendicular to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal /= norm
            axes.append(normal)
        return axes

    def project(verts, axis):
        """Project vertices onto axis."""
        proj = np.dot(verts, axis)
        return proj.min(), proj.max()

    def sat_overlap(verts1, verts2):
        """
        Check if two convex polygons overlap using SAT.
        Returns (overlap, axis) where overlap is the minimum penetration depth
        and axis is the separating axis (normal).
        Returns (0, None) if no overlap.
        """
        min_overlap = float('inf')
        min_axis = None
        
        axes1 = get_axes(verts1)
        axes2 = get_axes(verts2)
        all_axes = axes1 + axes2
        
        for axis in all_axes:
            min1, max1 = project(verts1, axis)
            min2, max2 = project(verts2, axis)
            
            if max1 < min2 or max2 < min1:
                return 0.0, None # Separated
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
                min_axis = axis
        
        # Ensure axis points from 1 to 2
        c1 = np.mean(verts1, axis=0)
        c2 = np.mean(verts2, axis=0)
        diff = c2 - c1
        if np.dot(diff, min_axis) < 0:
            min_axis = -min_axis
            
        return min_overlap, min_axis

    def find_contact_points(verts1, verts2, axis):
        """
        Find contact points for impulse resolution.
        Simplified: uses the deepest vertex of one polygon against the face of the other.
        """
        # Project all vertices of both polygons onto the contact axis
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        
        # Find vertices closest to the contact zone
        # The contact zone is the intersection of the projections
        min_proj = max(proj1.min(), proj2.min())
        max_proj = min(proj1.max(), proj2.max())
        
        # Get vertices within the contact zone (with some tolerance)
        tol = 1e-6
        contacts = []
        
        # Check vertices of 1 against faces of 2
        # And vertices of 2 against faces of 1
        
        # Simplified approach for hexagons: just take the average of the "deepest" points
        # For stability, we'll identify the vertices that define the overlap interval
        
        # Vertices of 1 that are in the overlap range
        v1_idx = np.where((proj1 >= min_proj - tol) & (proj1 <= max_proj + tol))[0]
        v2_idx = np.where((proj2 >= min_proj - tol) & (proj2 <= max_proj + tol))[0]
        
        # If we have vertices from both, we might have edge-edge contact.
        # For a basic impulse solver, using the center of mass of the contact vertices is often sufficient.
        
        # Let's pick the vertex from each polygon that is most "deep" into the other
        # Actually, for impulse resolution, we need the contact point.
        # A common approximation is the midpoint between the closest points.
        
        # Let's find the closest pair of points between the convex hulls.
        # Since they are small (6 verts), we can just brute force closest points.
        min_dist = float('inf')
        closest_p1 = None
        closest_p2 = None
        for i in range(len(verts1)):
            for j in range(len(verts2)):
                d = np.linalg.norm(verts1[i] - verts2[j])
                if d < min_dist:
                    min_dist = d
                    closest_p1 = verts1[i]
                    closest_p2 = verts2[j]
        
        contact_point = (closest_p1 + closest_p2) / 2.0
        return contact_point

    # --- Simulation ---
    
    def simulate(inner_init, outer_center, outer_side, outer_angle, dt=0.01, steps=200):
        """
        Run rigid body simulation.
        inner_init: list of [x, y, angle_deg]
        Returns: list of [x, y, angle_deg]
        """
        n = len(inner_init)
        # State: x, y, angle, vx, vy, w (angular vel)
        state = np.zeros((n, 6))
        for i, (x, y, a) in enumerate(inner_init):
            state[i, 0] = x
            state[i, 1] = y
            state[i, 2] = math.radians(a)
            
        # Constants
        mass = 1.0
        inertia = 0.5 * mass * (1.0**2) # Approximation for hexagon moment of inertia
        gravity_strength = 0.5 # Pull towards outer center
        
        # Precompute outer hexagon vertices
        outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        outer_axes = get_axes(outer_verts)
        
        # Precompute inner hexagon vertices for current state (will update)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # 1. Apply Forces
            # Gravity towards center
            for i in range(n):
                cx, cy = state[i, 0], state[i, 1]
                dx = outer_center[0] - cx
                dy = outer_center[1] - cy
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 1e-6:
                    fx = gravity_strength * dx / dist
                    fy = gravity_strength * dy / dist
                    state[i, 3] += fx * dt / mass
                    state[i, 4] += fy * dt / mass
            
            # Damping
            for i in range(n):
                state[i, 3] *= 0.98
                state[i, 4] *= 0.98
                state[i, 5] *= 0.95 # Angular damping
            
            # 2. Update Positions (Velocity Verlet simplified: Euler)
            for i in range(n):
                state[i, 0] += state[i, 3] * dt
                state[i, 1] += state[i, 4] * dt
                state[i, 2] += state[i, 5] * dt
            
            # 3. Collision Detection and Resolution
            # We need to iterate a few times to resolve simultaneous collisions
            for iter in range(3):
                # Inner-Outer Collisions
                for i in range(n):
                    cx, cy, ang = state[i, 0], state[i, 1], state[i, 2]
                    inner_verts = hexagon_vertices(cx, cy, 1.0, math.degrees(ang))
                    
                    overlap, axis = sat_overlap(inner_verts, outer_verts)
                    if overlap > 0:
                        # Push inner out
                        # Impulse along axis
                        j = -overlap / dt # Approximate
                        # Apply to velocity
                        state[i, 3] -= axis[0] * j * 0.5
                        state[i, 4] -= axis[1] * j * 0.5
                        
                        # Positional correction
                        state[i, 0] -= axis[0] * overlap * 0.5
                        state[i, 1] -= axis[1] * overlap * 0.5
                        
                        # Torque
                        contact_point = find_contact_points(inner_verts, outer_verts, axis)
                        r = np.array([contact_point[0] - cx, contact_point[1] - cy])
                        torque = r[0] * (-axis[1] * j) - r[1] * (-axis[0] * j)
                        state[i, 5] += torque * dt / inertia
                        
                # Inner-Inner Collisions
                for i in range(n):
                    for j in range(i + 1, n):
                        cx1, cy1, ang1 = state[i, 0], state[i, 1], state[i, 2]
                        cx2, cy2, ang2 = state[j, 0], state[j, 1], state[j, 2]
                        
                        verts1 = hexagon_vertices(cx1, cy1, 1.0, math.degrees(ang1))
                        verts2 = hexagon_vertices(cx2, cy2, 1.0, math.degrees(ang2))
                        
                        overlap, axis = sat_overlap(verts1, verts2)
                        if overlap > 0:
                            # Equal mass, so split impulse
                            j_imp = -overlap / dt * 0.5
                            
                            # Velocity update
                            state[i, 3] -= axis[0] * j_imp
                            state[i, 4] -= axis[1] * j_imp
                            state[j, 3] += axis[0] * j_imp
                            state[j, 4] += axis[1] * j_imp
                            
                            # Positional correction
                            state[i, 0] -= axis[0] * overlap * 0.5
                            state[i, 1] -= axis[1] * overlap * 0.5
                            state[j, 0] += axis[0] * overlap * 0.5
                            state[j, 1] += axis[1] * overlap * 0.5
                            
                            # Torques
                            cp = find_contact_points(verts1, verts2, axis)
                            r1 = np.array([cp[0] - cx1, cp[1] - cy1])
                            r2 = np.array([cp[0] - cx2, cp[1] - cy2])
                            
                            t1 = r1[0] * (-axis[1] * j_imp) - r1[1] * (-axis[0] * j_imp)
                            t2 = r2[0] * (axis[1] * j_imp) - r2[1] * (axis[0] * j_imp)
                            
                            state[i, 5] += t1 * dt / inertia
                            state[j, 5] += t2 * dt / inertia

        # Return final positions
        result = []
        for i in range(n):
            result.append([state[i, 0], state[i, 1], math.degrees(state[i, 2])])
        return result

    def min_outer_side(inner_list, outer_center, outer_angle):
        """Binary search for minimum outer side containing all inner hexagons."""
        cx, cy = outer_center
        lo = 1.0
        hi = 25.0
        
        def fits(side):
            for ix, iy, iang in inner_list:
                verts = hexagon_vertices(ix, iy, 1.0, iang)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, cx, cy, side, outer_angle):
                        return False
            return True
        
        if not fits(hi):
            return None
        if fits(lo):
            return lo
        
        for _ in range(30):
            mid = (lo + hi) / 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        return hi + 1e-10

    def check_disjoint(inner_list, threshold=1e-6):
        """Check all pairs are disjoint using SAT."""
        n = len(inner_list)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, a1 = inner_list[i]
                x2, y2, a2 = inner_list[j]
                v1 = hexagon_vertices(x1, y1, 1.0, a1)
                v2 = hexagon_vertices(x2, y2, 1.0, a2)
                overlap, _ = sat_overlap(v1, v2)
                if overlap > threshold:
                    return False
        return True

    def evaluate(inner_list, outer_center, outer_angle):
        """Evaluate packing: return (side, valid)."""
        if not check_disjoint(inner_list):
            return None, False
        side = min_outer_side(inner_list, outer_center, outer_angle)
        if side is None:
            return None, False
        return side, True

    # --- Initial Configuration Generation ---
    
    def generate_hex_ring(n):
        positions = [(0, 0)]
        ring = 1
        while len(positions) < n:
            q, r = ring, 0
            for side_idx in range(6):
                for step in range(ring):
                    positions.append((q, r))
                    dq = [1, 0, -1, -1, 0, 1][side_idx]
                    dr = [0, -1, -1, 0, 1, 1][side_idx]
                    q += dq
                    r += dr
            ring += 1
        return positions[:n]

    def axial_to_cartesian(q, r, spacing):
        x = spacing * (2.0 * q + r)
        y = spacing * sqrt3 * r
        return x, y

    axial_pos = generate_hex_ring(N)
    
    best_side = float('inf')
    best_result = None
    
    # --- Strategy 1: Physics Simulation with Hexagonal Lattice Start ---
    for spacing in [1.7, 1.6, 1.5]:
        if time.time() > deadline - 30:
            break
        for outer_angle in [0.0, 30.0]:
            if time.time() > deadline - 25:
                break
            
            init_inner = []
            for q, r in axial_pos:
                x, y = axial_to_cartesian(q, r, spacing)
                init_inner.append([x, y, 0.0])
            
            # Center cluster
            cx_mean = sum(p[0] for p in init_inner) / len(init_inner)
            cy_mean = sum(p[1] for p in init_inner) / len(init_inner)
            centered_init = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in init_inner]
            
            # Estimate initial outer side
            max_r = max(math.sqrt(p[0]**2 + p[1]**2) for p in centered_init)
            est_side = max_r * 1.5 + 2.0
            
            # Run simulation
            sim_result = simulate(centered_init, (0.0, 0.0), est_side, outer_angle, dt=0.02, steps=300)
            
            # Re-center result
            cx_mean = sum(p[0] for p in sim_result) / len(sim_result)
            cy_mean = sum(p[1] for p in sim_result) / len(sim_result)
            centered_sim = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in sim_result]
            
            side, valid = evaluate(centered_sim, (0.0, 0.0), outer_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (centered_sim, (0.0, 0.0), side, outer_angle)
    
    # --- Strategy 2: Random Starts ---
    if time.time() < deadline - 40:
        rng = np.random.default_rng(42)
        for trial in range(10):
            if time.time() > deadline - 20:
                break
            
            init_inner = []
            for _ in range(N):
                x = rng.uniform(-3, 3)
                y = rng.uniform(-3, 3)
                a = rng.uniform(0, 360)
                init_inner.append([x, y, a])
            
            cx_mean = sum(p[0] for p in init_inner) / len(init_inner)
            cy_mean = sum(p[1] for p in init_inner) / len(init_inner)
            centered_init = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in init_inner]
            
            max_r = max(math.sqrt(p[0]**2 + p[1]**2) for p in centered_init)
            est_side = max_r * 1.5 + 2.0
            
            sim_result = simulate(centered_init, (0.0, 0.0), est_side, 0.0, dt=0.02, steps=300)
            
            cx_mean = sum(p[0] for p in sim_result) / len(sim_result)
            cy_mean = sum(p[1] for p in sim_result) / len(sim_result)
            centered_sim = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in sim_result]
            
            side, valid = evaluate(centered_sim, (0.0, 0.0), 0.0)
            if valid and side < best_side:
                best_side = side
                best_result = (centered_sim, (0.0, 0.0), side, 0.0)

    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-END
