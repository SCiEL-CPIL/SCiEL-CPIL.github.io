import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute hex vertices for side 1, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        verts = []
        for vx, vy in base_verts:
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            verts.append((cx + S * rx, cy + S * ry))
        return verts

    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def point_in_polygon(p, polygon):
        x, y = p
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(verts_a, verts_b):
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        for i in range(6):
            a1, a2 = verts_a[i], verts_a[(i + 1) % 6]
            b1, b2 = verts_b[i], verts_b[(i + 1) % 6]
            if segments_intersect(a1, a2, b1, b2):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def all_hexagons_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            iv = transform_verts(ix, iy, 1.0, iangle)
            inner_verts.append(iv)
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        for i in range(len(inner_verts)):
            for j in range(i + 1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def force_directed_optimize(initial_positions, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        
        if time.time() > deadline:
            return

        rng = random.Random(seed)
        
        # Define a plausible contact graph (adjacency)
        # 0 is center, 1-5 are neighbors, 6-10 are second ring
        # This is a heuristic graph. The force model will relax it.
        adj_matrix = np.zeros((N, N), dtype=int)
        adj_matrix[0, 1:6] = 1
        adj_matrix[1:6, 0] = 1
        # Second ring connections
        adj_matrix[1, 6] = 1
        adj_matrix[2, 7] = 1
        adj_matrix[3, 8] = 1
        adj_matrix[4, 9] = 1
        adj_matrix[5, 10] = 1
        adj_matrix[6, 1] = 1
        adj_matrix[7, 2] = 1
        adj_matrix[8, 3] = 1
        adj_matrix[9, 4] = 1
        adj_matrix[10, 5] = 1
        
        # Also connect neighbors in the first ring to each other? 
        # In a hexagonal packing, 1 and 2 are NOT touching. 1 and 6 are touching.
        # Let's stick to the direct neighbor model for the springs.
        # Actually, for a compact blob, we might want to encourage 1-2, 2-3 etc? No, they are far.
        # The springs should only exist for pairs that *should* be touching in a tight packing.
        
        # Force parameters
        k_spring = 10.0
        k_repel = 100.0
        d_min = math.sqrt(3) # Minimum distance for side 1 hexagons to touch
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            # Unpack
            pos = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            
            # Symmetry breaking: fix hex 0 at origin
            pos[0] = [0.0, 0.0]
            angles[0] = 0.0
            
            # Compute forces
            forces = np.zeros_like(pos)
            torques = np.zeros(N)
            
            # 1. Spring forces (attractive if too far, repulsive if too close, but mainly attractive to hold structure)
            # We want them to be at distance d_min if connected.
            for i in range(N):
                for j in range(i+1, N):
                    if adj_matrix[i, j] == 1:
                        dx = pos[j][0] - pos[i][0]
                        dy = pos[j][1] - pos[i][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-6:
                            continue
                        # Spring force magnitude
                        # Hooke's law: F = k * (dist - d_min)
                        # If dist > d_min, pull together. If dist < d_min, push apart (though repulsion handles this)
                        f_mag = k_spring * (dist - d_min)
                        fx = f_mag * dx / dist
                        fy = f_mag * dy / dist
                        
                        forces[i][0] += fx
                        forces[i][1] += fy
                        forces[j][0] -= fx
                        forces[j][1] -= fy
                        
                        # Torque: align edges
                        # Ideal relative angle is determined by geometry.
                        # For simplicity, we just apply a small torque to align them to 0 or 60 degrees relative to each other?
                        # Actually, just minimizing the angle difference to the line connecting centers might help.
                        # Let's skip complex torque for now, just position forces.
                        # Or a simple torque to keep them flat.
                        # torque[i] += 0.1 * (angles[i] - angles[j])

            # 2. Repulsive forces (Lennard-Jones-like for ALL pairs)
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[j][0] - pos[i][0]
                    dy = pos[j][1] - pos[i][1]
                    dist_sq = dx*dx + dy*dy
                    dist = math.sqrt(dist_sq)
                    if dist < 1e-6:
                        continue
                    # Soft repulsion to prevent overlap
                    # F_repel = k_repel / dist^2 (simplified)
                    # Use a smoother potential
                    if dist < d_min * 1.1:
                         # Strong repulsion
                         f_mag = -k_repel * (d_min - dist) / dist
                    else:
                         f_mag = 0.0
                    # Always a small repulsion to keep them apart if not connected?
                    # No, only if they are close.
                    
                    fx = f_mag * dx / dist
                    fy = f_mag * dy / dist
                    
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
            
            # 3. Centering force
            # Pull all towards center of mass (which is 0 if hex 0 is fixed and symmetric)
            # Or just a weak spring to origin for all
            for i in range(1, N):
                forces[i][0] -= 0.1 * pos[i][0]
                forces[i][1] -= 0.1 * pos[i][1]

            # Compute objective: Sum of squared forces + Torques
            obj = np.sum(forces**2) + np.sum(torques**2)
            
            # Add a penalty if any pair is overlapping (dist < d_min)
            # This ensures we don't converge to an invalid state
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[j][0] - pos[i][0]
                    dy = pos[j][1] - pos[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < d_min:
                        obj += 1000.0 * (d_min - dist)**2
            
            return obj

        # Initial guess
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[2*i] = initial_positions[i][0]
            x0[2*i+1] = initial_positions[i][1]
            x0[2*N+i] = initial_positions[i][2]

        # Optimize
        try:
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-5})
            
            if res.success or res.fun < 1e-3:
                # Extract results
                pos = res.x[:2*N].reshape(N, 2)
                angles = res.x[2*N:3*N]
                
                inner = []
                for i in range(N):
                    inner.append([pos[i][0], pos[i][1], angles[i] * 180.0 / math.pi])
                
                # Center the configuration
                cx = sum(p[0] for p in inner) / N
                cy = sum(p[1] for p in inner) / N
                for i in range(N):
                    inner[i][0] -= cx
                    inner[i][1] -= cy
                
                # Try a few outer angles
                for oangle in [0, 30, 60]:
                    S = compute_outer_hex(inner, 0.0, 0.0, oangle)
                    if all_hexagons_fit(inner, 0.0, 0.0, S, oangle):
                        if S < best_S:
                            best_S = S
                            best_inner = [list(x) for x in inner]
                            best_center = (0.0, 0.0)
                            best_angle = oangle
                            best_valid = True
        except Exception:
            pass

    # Generate initial configurations for the force-directed optimizer
    # 1. Hexagonal lattice blob
    sp = math.sqrt(3)
    hex_config = [[0.0, 0.0, 0.0]]
    for i in range(6):
        theta = math.radians(60 * i)
        hex_config.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    for i in [0, 2, 4]:
        theta = math.radians(60 * i + 30)
        hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
    for i in [1, 3, 5]:
        theta = math.radians(60 * i)
        hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
    while len(hex_config) < N:
        hex_config.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
    
    initial_configs = [hex_config[:N]]
    
    # 2. Random perturbation of hexagonal lattice
    for _ in range(3):
        cfg = [list(x) for x in hex_config[:N]]
        for i in range(N):
            cfg[i][0] += random.gauss(0, 0.5)
            cfg[i][1] += random.gauss(0, 0.5)
        initial_configs.append(cfg)
        
    # 3. Random cluster
    for _ in range(3):
        cfg = [[random.uniform(-2, 2), random.uniform(-2, 2), 0.0] for _ in range(N)]
        initial_configs.append(cfg)

    seed_counter = 100
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        force_directed_optimize(cfg, seed_counter)
        seed_counter += 1

    # Fallback if force-directed didn't find a valid solution
    if not best_valid or best_inner is None:
        sp = math.sqrt(3)
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        for _ in range(100):
            if all_hexagons_fit(fallback_inner, 0.0, 0.0, S, 0.0):
                return fallback_inner, (0.0, 0.0), S, 0.0
            for k in range(N):
                fallback_inner[k][0] *= 1.05
            S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
