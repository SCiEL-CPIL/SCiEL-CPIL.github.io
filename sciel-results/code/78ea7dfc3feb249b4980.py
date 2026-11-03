import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def hexagons_disjoint(inner, eps=1e-6):
        """Check if all hexagons are disjoint using SAT."""
        n = len(inner)
        for i in range(n):
            for j in range(i+1, n):
                if not sat_disjoint(inner[i], inner[j], eps):
                    return False
        return True

    def sat_disjoint(hex1, hex2, eps=1e-6):
        """Check if two hexagons are disjoint using Separating Axis Theorem."""
        x1, y1, a1 = hex1
        x2, y2, a2 = hex2
        verts1 = get_hex_vertices(x1, y1, 1.0, a1)
        verts2 = get_hex_vertices(x2, y2, 1.0, a2)
        
        # Get all edge normals from both hexagons
        axes = []
        for verts in [verts1, verts2]:
            for i in range(6):
                x_a, y_a = verts[i]
                x_b, y_b = verts[(i+1)%6]
                dx = x_b - x_a
                dy = y_b - y_a
                length = math.sqrt(dx*dx + dy*dy)
                if length > 1e-10:
                    # Normal perpendicular to edge
                    nx = -dy / length
                    ny = dx / length
                    axes.append((nx, ny))
        
        # Check each axis
        for nx, ny in axes:
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            for px, py in verts1:
                proj = px * nx + py * ny
                min1 = min(min1, proj)
                max1 = max(max1, proj)
            for px, py in verts2:
                proj = px * nx + py * ny
                min2 = min(min2, proj)
                max2 = max(max2, proj)
            # If there's a gap, they are disjoint
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        return False

    def compute_min_outer_side(inner, cx, cy, theta_deg):
        """Compute minimum outer hexagon side length for given center and angle."""
        theta = math.radians(theta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for x, y, a in inner:
                verts = get_hex_vertices(x, y, 1.0, a)
                for px, py in verts:
                    proj = (px - cx) * nx + (py - cy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def get_hex_vertices_array(cx, cy, side, angle_rad):
        """Vectorized hex vertices."""
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def sat_disjoint_fast(hex1, hex2, eps=1e-6):
        """Fast SAT check for two hexagons."""
        x1, y1, a1 = hex1
        x2, y2, a2 = hex2
        verts1 = get_hex_vertices_array(x1, y1, 1.0, math.radians(a1))
        verts2 = get_hex_vertices_array(x2, y2, 1.0, math.radians(a2))
        
        # Get all edge normals from both hexagons
        for verts in [verts1, verts2]:
            for i in range(6):
                x_a, y_a = verts[i]
                x_b, y_b = verts[(i+1)%6]
                dx = x_b - x_a
                dy = y_b - y_a
                length = math.sqrt(dx*dx + dy*dy)
                if length > 1e-10:
                    nx = -dy / length
                    ny = dx / length
                    
                    proj1 = verts1 @ np.array([nx, ny])
                    proj2 = verts2 @ np.array([nx, ny])
                    
                    if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                        return True
        return False

    def hexagons_disjoint_fast(inner, eps=1e-6):
        """Check if all hexagons are disjoint using fast SAT."""
        n = len(inner)
        for i in range(n):
            for j in range(i+1, n):
                if not sat_disjoint_fast(inner[i], inner[j], eps):
                    return False
        return True

    def compute_min_outer_side_fast(inner, cx, cy, theta_deg):
        """Fast computation of minimum outer hexagon side length using vectorized ops."""
        theta = math.radians(theta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for x, y, a in inner:
                verts = get_hex_vertices_array(x, y, 1.0, math.radians(a))
                projs = (verts[:, 0] - cx) * nx + (verts[:, 1] - cy) * ny
                if projs.max() > max_proj:
                    max_proj = projs.max()
        return 2.0 * max_proj / math.sqrt(3)

    def structured_local_search(inner, cx, cy, theta_deg, max_iter=300, step_size=0.05):
        """Structured local search with coordinate descent and line search."""
        best_side = compute_min_outer_side_fast(inner, cx, cy, theta_deg)
        best_inner = [list(h) for h in inner]
        best_cx, best_cy, best_theta = cx, cy, theta_deg
        
        current_inner = [list(h) for h in inner]
        current_cx, current_cy, current_theta = cx, cy, theta_deg
        current_side = best_side
        
        # Precompute directions for each hexagon
        directions = []
        for _ in range(N):
            dirs = []
            for ang in np.linspace(0, 2*np.pi, 12, endpoint=False):
                dirs.append((math.cos(ang), math.sin(ang)))
            directions.append(dirs)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            step = step_size
            
            # Try each hexagon in turn
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                
                # Try each direction for this hexagon
                for dx_dir, dy_dir in directions[i]:
                    new_inner = [list(h) for h in current_inner]
                    new_inner[i][0] += step * dx_dir
                    new_inner[i][1] += step * dy_dir
                    
                    if not hexagons_disjoint_fast(new_inner):
                        continue
                    
                    new_side = compute_min_outer_side_fast(new_inner, current_cx, current_cy, current_theta)
                    
                    if new_side < current_side - 1e-10:
                        # Accept and try to extend in this direction
                        current_inner = new_inner
                        current_side = new_side
                        improved = True
                        
                        # Line search: try to go further in same direction
                        for _ in range(5):
                            new_inner2 = [list(h) for h in current_inner]
                            new_inner2[i][0] += step * dx_dir * 0.5
                            new_inner2[i][1] += step * dy_dir * 0.5
                            
                            if not hexagons_disjoint_fast(new_inner2):
                                break
                            
                            new_side2 = compute_min_outer_side_fast(new_inner2, current_cx, current_cy, current_theta)
                            if new_side2 < current_side - 1e-10:
                                current_inner = new_inner2
                                current_side = new_side2
                            else:
                                break
            
            # Try outer center perturbation
            for _ in range(6):
                dx = random.uniform(-step, step)
                dy = random.uniform(-step, step)
                new_cx, new_cy = current_cx + dx, current_cy + dy
                new_side = compute_min_outer_side_fast(current_inner, new_cx, new_cy, current_theta)
                if new_side < current_side - 1e-10:
                    current_cx, current_cy = new_cx, new_cy
                    current_side = new_side
                    improved = True
            
            # Try outer angle perturbation
            for dtheta in [-1.0, -0.5, 0.5, 1.0]:
                new_theta = current_theta + dtheta
                new_side = compute_min_outer_side_fast(current_inner, current_cx, current_cy, new_theta)
                if new_side < current_side - 1e-10:
                    current_theta = new_theta
                    current_side = new_side
                    improved = True
            
            if not improved:
                step *= 0.7
                if step < 0.001:
                    break
            else:
                step *= 0.95
            
            if current_side < best_side - 1e-12:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_cx, best_cy, best_theta = current_cx, current_cy, current_theta
        
        return best_inner, best_cx, best_cy, best_theta, best_side

    def high_precision_polish(inner, cx, cy, theta_deg, max_iter=1000, initial_step=0.005):
        """High-precision polish with very small steps and many iterations."""
        best_side = compute_min_outer_side_fast(inner, cx, cy, theta_deg)
        best_inner = [list(h) for h in inner]
        best_cx, best_cy, best_theta = cx, cy, theta_deg
        
        current_inner = [list(h) for h in inner]
        current_cx, current_cy, current_theta = cx, cy, theta_deg
        current_side = best_side
        step = initial_step
        
        for iteration in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            
            # Fine-grained position adjustments
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_inner = [list(h) for h in current_inner]
                        new_inner[i][0] += dx
                        new_inner[i][1] += dy
                        
                        if not hexagons_disjoint_fast(new_inner):
                            continue
                        
                        new_side = compute_min_outer_side_fast(new_inner, current_cx, current_cy, current_theta)
                        if new_side < current_side - 1e-12:
                            current_inner = new_inner
                            current_side = new_side
                            improved = True
                            break
                    if improved:
                        break
            
            # Fine-grained outer center
            for dx in [-step, step]:
                for dy in [-step, step]:
                    new_cx, new_cy = current_cx + dx, current_cy + dy
                    new_side = compute_min_outer_side_fast(current_inner, new_cx, new_cy, current_theta)
                    if new_side < current_side - 1e-12:
                        current_cx, current_cy = new_cx, new_cy
                        current_side = new_side
                        improved = True
                        break
                if improved:
                    break
            
            # Fine-grained outer angle
            for dtheta in [-0.1, 0.1]:
                new_theta = current_theta + dtheta
                new_side = compute_min_outer_side_fast(current_inner, current_cx, current_cy, new_theta)
                if new_side < current_side - 1e-12:
                    current_theta = new_theta
                    current_side = new_side
                    improved = True
                    break
            
            if not improved:
                step *= 0.8
                if step < 1e-5:
                    break
            else:
                step *= 0.98
            
            if current_side < best_side - 1e-14:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_cx, best_cy, best_theta = current_cx, current_cy, current_theta
        
        return best_inner, best_cx, best_cy, best_theta, best_side

    def generate_ring12_topology():
        """Generate 1+6+5 ring topology."""
        eps = 0.01
        d = math.sqrt(3) + eps
        
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = [np.array([0.0, 0.0])]
        
        ring1_vectors = [
            u, v, v - u, -u, -v, u - v
        ]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
        ring2_candidates = []
        basis = [u, v, v-u, -u, -v, u-v]
        
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
                        
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
            
        return [[c[0], c[1], 30.0] for c in centers]

    def generate_hex6plus6_topology():
        """Generate two clusters of 6 hexagons side by side."""
        eps = 0.01
        d = math.sqrt(3) + eps
        
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = []
        
        # First cluster: center + 5 neighbors (one side open)
        centers.append(np.array([0.0, 0.0]))
        ring1 = [u, v, v-u, -u, -v]  # 5 neighbors, leaving u-v side open
        for vec in ring1:
            centers.append(vec.copy())
        
        # Second cluster: offset by 2*d in x direction
        offset = np.array([2.0*d, 0.0])
        centers.append(offset + np.array([0.0, 0.0]))
        ring1_2 = [u, v, v-u, -u, -v]
        for vec in ring1_2:
            centers.append(offset + vec.copy())
            
        return [[c[0], c[1], 30.0] for c in centers]

    def generate_rect3x4_topology():
        """Generate a 3x4 rectangular grid of hexagons."""
        eps = 0.01
        d = math.sqrt(3) + eps
        
        # For a grid, we need to stagger rows
        # Row spacing: d*sqrt(3)/2 * 2 = d*sqrt(3) for touching
        row_spacing = d * math.sqrt(3)
        col_spacing = d
        
        centers = []
        for row in range(3):
            for col in range(4):
                x = col * col_spacing
                y = row * row_spacing
                if row % 2 == 1:
                    x += col_spacing / 2.0
                centers.append(np.array([x, y]))
                
        return [[c[0], c[1], 30.0] for c in centers]

    def generate_compact_cluster_topology():
        """Generate a compact 3x4 arrangement with tighter spacing."""
        eps = 0.005
        d = math.sqrt(3) + eps
        
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = []
        
        # 4 rows of 3 hexagons each, staggered
        for row in range(4):
            for col in range(3):
                x = col * u[0]
                y = row * v[1]
                if row % 2 == 1:
                    x += u[0] / 2.0
                centers.append(np.array([x, y]))
        
        return [[c[0], c[1], 30.0] for c in centers]

    def evaluate_topology(inner):
        """Evaluate a topology and return best configuration."""
        all_verts = []
        for x, y, a in inner:
            verts = get_hex_vertices(x, y, 1.0, a)
            all_verts.extend(verts)
            
        cx = sum(p[0] for p in all_verts) / len(all_verts)
        cy = sum(p[1] for p in all_verts) / len(all_verts)
        
        best_side = float('inf')
        best_angle = 0.0
        
        # Scan outer hexagon angle with finer resolution
        for theta_deg in np.linspace(0, 60, 121):
            s = compute_min_outer_side_fast(inner, cx, cy, theta_deg)
            if s < best_side:
                best_side = s
                best_angle = theta_deg
                
        return inner, cx, cy, best_angle, best_side

    # Generate multiple topologies and find the best
    topologies = [
        ("ring12", generate_ring12_topology()),
        ("hex6plus6", generate_hex6plus6_topology()),
        ("rect3x4", generate_rect3x4_topology()),
        ("compact", generate_compact_cluster_topology()),
    ]
    
    best_overall = None
    best_overall_side = float('inf')
    
    for name, inner in topologies:
        if time.time() > deadline - 30.0:
            break
            
        # Check if topology is valid (disjoint)
        if not hexagons_disjoint_fast(inner):
            continue
            
        inner_eval, cx_eval, cy_eval, angle_eval, side_eval = evaluate_topology(inner)
        
        if side_eval < best_overall_side:
            best_overall = (inner_eval, cx_eval, cy_eval, angle_eval, side_eval)
            best_overall_side = side_eval
    
    if best_overall is None:
        # Fallback: use ring12 with larger epsilon
        eps = 0.05
        d = math.sqrt(3) + eps
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
        ring2_candidates = []
        basis = [u, v, v-u, -u, -v, u-v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
                        
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
            
        inner = [[c[0], c[1], 30.0] for c in centers]
        best_overall = evaluate_topology(inner)
        best_overall_side = best_overall[4]
    
    inner, cx, cy, best_angle, best_side = best_overall
    
    # Structured local search
    if time.time() < deadline - 30.0:
        inner_opt, cx_opt, cy_opt, theta_opt, side_opt = structured_local_search(
            inner, cx, cy, best_angle, max_iter=200, step_size=0.03
        )
        if side_opt < best_side:
            inner = inner_opt
            cx, cy = cx_opt, cy_opt
            best_angle = theta_opt
            best_side = side_opt
    
    # High-precision polish
    if time.time() < deadline - 5.0:
        inner_polish, cx_polish, cy_polish, theta_polish, side_polish = high_precision_polish(
            inner, cx, cy, best_angle, max_iter=500, initial_step=0.005
        )
        if side_polish < best_side:
            inner = inner_polish
            cx, cy = cx_polish, cy_polish
            best_angle = theta_polish
            best_side = side_polish
    
    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
