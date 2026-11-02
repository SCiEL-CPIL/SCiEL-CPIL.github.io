"""
Strategy: Grid-Based Local Search (Hill Climbing) with Multi-Start.
Instead of Simulated Annealing's random perturbations and probabilistic acceptance,
this strategy uses a deterministic, discrete search over a grid of positions and angles.
It iteratively moves each hexagon to the best neighboring grid cell that improves the
objective (outer side length) while maintaining disjointness. Multiple random starts
are used to mitigate local minima. This approach is often more stable and precise
for low-dimensional packing problems when the grid is fine enough.
"""
import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        Returns True if disjoint, False if overlapping.
        """
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def min_outer_side(inner_positions, outer_angle_deg):
        """
        Calculate the minimum side length of a regular hexagon with given angle
        and center (0,0) that contains all inner hexagons.
        """
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
        
        all_verts = np.vstack(all_verts)
        
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        max_proj = 0.0
        for na in normals:
            nx = np.cos(na)
            ny = np.sin(na)
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        return max_proj / (np.sqrt(3) / 2)

    def get_min_outer_config(inner_positions):
        """
        Find the best outer hexagon (center, side, angle) for given inner positions.
        """
        best_s = float('inf')
        best_theta = 0.0
        
        # Scan theta
        for theta in np.arange(0, 60, 1.0):
            s = min_outer_side(inner_positions, theta)
            if s < best_s:
                best_s = s
                best_theta = theta
        
        # Refine theta around best
        for theta in np.arange(best_theta - 1.0, best_theta + 1.0, 0.1):
            s = min_outer_side(inner_positions, theta)
            if s < best_s:
                best_s = s
                best_theta = theta
                
        return best_s, best_theta

    def objective(params):
        """
        params: 3*N values for inner hexagons (x, y, angle).
        Outer hexagon is assumed centered at (0,0).
        Returns the side length of the minimal enclosing outer hexagon + penalty for overlap.
        """
        inner_positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    penalty += 10.0
        
        s, theta = get_min_outer_config(inner_positions)
        
        return s + penalty

    def local_search(params, step_size, angle_step, max_iters=500):
        """
        Perform local search by trying to move each hexagon to a better neighbor.
        """
        current_params = params.copy()
        current_score = objective(current_params)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                # Try 8 spatial neighbors + 3 angle neighbors + original
                # Spatial: N, S, E, W, NE, NW, SE, SW
                # Angle: +angle_step, -angle_step
                
                best_neighbor_params = current_params
                best_neighbor_score = current_score
                
                # Spatial moves
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        trial = current_params.copy()
                        trial[3*i] += dx
                        trial[3*i+1] += dy
                        score = objective(trial)
                        if score < best_neighbor_score:
                            best_neighbor_score = score
                            best_neighbor_params = trial
                            
                # Angle moves
                for da in [-angle_step, angle_step]:
                    trial = current_params.copy()
                    trial[3*i+2] += da
                    score = objective(trial)
                    if score < best_neighbor_score:
                        best_neighbor_score = score
                        best_neighbor_params = trial
                        
                if best_neighbor_score < current_score:
                    current_params = best_neighbor_params
                    current_score = best_neighbor_score
                    improved = True
                    
            if not improved:
                break
                
        return current_params, current_score

    # Initial configuration: Hexagonal lattice
    def generate_initial_config(seed):
        np.random.seed(seed)
        init_params = np.zeros(3 * N)
        
        # Center
        init_params[0:3] = [0.0, 0.0, 0.0]
        
        # Ring 1
        for k in range(6):
            angle = np.radians(60 * k)
            idx = (k + 1) * 3
            init_params[idx:idx+3] = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0]
            
        # Ring 2 (5 hexagons)
        for k in range(5):
            angle = np.radians(60 * k + 30)
            idx = (k + 7) * 3
            init_params[idx:idx+3] = [3.5 * np.cos(angle), 3.5 * np.sin(angle), 0.0]
            
        # Add some random perturbation to break symmetry
        for i in range(1, N):
            init_params[3*i] += np.random.uniform(-0.5, 0.5)
            init_params[3*i+1] += np.random.uniform(-0.5, 0.5)
            init_params[3*i+2] += np.random.uniform(-10, 10)
            
        return init_params

    best_params = None
    best_score = float('inf')
    
    # Multi-start Local Search
    # We will run several local searches with different starting points and step sizes.
    # Start with coarser steps to find a good region, then refine.
    
    starts = [
        (0, 0.5, 10.0),
        (1, 0.5, 10.0),
        (2, 0.5, 10.0),
        (3, 0.5, 10.0),
        (4, 0.5, 10.0),
    ]
    
    for seed, step, angle_step in starts:
        if time.time() > deadline:
            break
            
        init_params = generate_initial_config(seed)
        
        # Coarse search
        p1, s1 = local_search(init_params, step, angle_step, max_iters=100)
        
        # Medium search
        p2, s2 = local_search(p1, step * 0.5, angle_step * 0.5, max_iters=100)
        
        # Fine search
        p3, s3 = local_search(p2, step * 0.25, angle_step * 0.25, max_iters=100)
        
        if s3 < best_score:
            best_score = s3
            best_params = p3
            
    if best_params is None:
        # Fallback to initial
        best_params = generate_initial_config(0)
        best_score = objective(best_params)
        
    # Final refinement with very small steps
    if time.time() < deadline - 10:
        best_params, best_score = local_search(best_params, 0.05, 1.0, max_iters=50)
        
    # Extract best inner positions
    inner = []
    for i in range(N):
        inner.append((best_params[3*i], best_params[3*i+1], best_params[3*i+2]))
        
    # Compute final outer side and angle
    s, theta = get_min_outer_config(inner)
    
    # Center is (0,0)
    center = (0.0, 0.0)
    side = s
    angle = theta
    
    # Validation check
    valid = True
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: expand side slightly
        side *= 1.01
        
    return inner, center, side, angle

# EVOLVE-BLOCK-END
