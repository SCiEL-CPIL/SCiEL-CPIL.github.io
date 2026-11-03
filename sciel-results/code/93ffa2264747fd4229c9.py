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

    def smooth_outer_side(inner, cx, cy, theta_deg, beta=50.0):
        """Smooth approximation of outer side using log-sum-exp for max projection."""
        theta = math.radians(theta_deg)
        # Collect all projections
        projs = []
        for k in range(6):
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for x, y, a in inner:
                verts = get_hex_vertices(x, y, 1.0, a)
                for px, py in verts:
                    proj = (px - cx) * nx + (py - cy) * ny
                    projs.append(proj)
        projs = np.array(projs)
        # Soft max via log-sum-exp
        m = np.max(projs)
        if m > 1e-10:
            log_sum_exp = m + np.log(np.sum(np.exp(projs - m)) / len(projs))
        else:
            log_sum_exp = 0.0
        return 2.0 * log_sum_exp / math.sqrt(3)

    def smooth_disjointness_penalty(inner, eps=0.01, beta=20.0):
        """
        Smooth penalty for hexagon overlap using minimum distance between edges.
        For each pair, compute the minimum gap (negative if overlapping).
        Use soft-min to get a smooth surrogate for the minimum gap.
        """
        n = len(inner)
        total_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, a1 = inner[i]
                x2, y2, a2 = inner[j]
                verts1 = get_hex_vertices(x1, y1, 1.0, a1)
                verts2 = get_hex_vertices(x2, y2, 1.0, a2)
                
                # Compute all edge-to-edge distances using SAT-like approach
                # Get minimum separation along all axes
                min_gap = float('inf')
                axes = []
                for verts in [verts1, verts2]:
                    for idx in range(6):
                        x_a, y_a = verts[idx]
                        x_b, y_b = verts[(idx+1)%6]
                        dx = x_b - x_a
                        dy = y_b - y_a
                        length = math.sqrt(dx*dx + dy*dy)
                        if length > 1e-10:
                            nx = -dy / length
                            ny = dx / length
                            axes.append((nx, ny))
                
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
                    # Gap along this axis (positive means separated)
                    gap = min(max1 - min2, max2 - min1)
                    # Actually we want the minimum separation
                    # If max1 < min2: gap = min2 - max1
                    # If max2 < min1: gap = min1 - max2
                    # Otherwise: gap = 0 (overlapping on this axis)
                    if max1 < min2:
                        gap = min2 - max1
                    elif max2 < min1:
                        gap = min1 - max2
                    else:
                        gap = 0.0
                    if gap < min_gap:
                        min_gap = gap
                
                # min_gap is the minimum separation (>=0 if disjoint, <0 if overlapping)
                # We want to penalize when min_gap < eps
                # Use smooth hinge: soft penalty when gap < eps
                deficit = eps - min_gap
                if deficit > 0:
                    # Smooth quadratic penalty
                    total_penalty += deficit * deficit / (2.0 * beta)
        
        return total_penalty

    def optimize_configuration(inner, cx, cy, theta_deg, max_iter=200, step_size=0.05):
        """Local optimization to reduce outer side length while maintaining disjointness."""
        best_side = compute_min_outer_side(inner, cx, cy, theta_deg)
        best_inner = [list(h) for h in inner]
        best_cx, best_cy, best_theta = cx, cy, theta_deg
        
        current_inner = [list(h) for h in inner]
        current_cx, current_cy, current_theta = cx, cy, theta_deg
        current_side = best_side
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try small perturbations
            for _ in range(20):
                new_inner = [list(h) for h in current_inner]
                new_cx, new_cy = current_cx, current_cy
                new_theta = current_theta
                
                # Randomly perturb one parameter
                r = random.random()
                if r < 0.7:
                    # Perturb a hexagon position
                    idx = random.randint(0, len(new_inner)-1)
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_inner[idx][0] += dx
                    new_inner[idx][1] += dy
                elif r < 0.85:
                    # Perturb outer center
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_cx += dx
                    new_cy += dy
                else:
                    # Perturb outer angle
                    dtheta = random.uniform(-2.0, 2.0)
                    new_theta += dtheta
                
                # Check disjointness
                if not hexagons_disjoint(new_inner):
                    continue
                
                # Compute new side length
                new_side = compute_min_outer_side(new_inner, new_cx, new_cy, new_theta)
                
                if new_side < current_side - 1e-8:
                    current_inner = new_inner
                    current_cx, current_cy, current_theta = new_cx, new_cy, new_theta
                    current_side = new_side
                    improved = True
                    step_size *= 0.95
                    break
            
            if not improved:
                step_size *= 0.8
                if step_size < 0.001:
                    break
            
            # Update best
            if current_side < best_side - 1e-10:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_cx, best_cy, best_theta = current_cx, current_cy, current_theta
        
        return best_inner, best_cx, best_cy, best_theta, best_side

    def polish_configuration(inner, cx, cy, theta_deg, max_iter=300, step_size=0.01):
        """High-precision polish pass with smaller steps and tighter tolerance."""
        best_side = compute_min_outer_side(inner, cx, cy, theta_deg)
        best_inner = [list(h) for h in inner]
        best_cx, best_cy, best_theta = cx, cy, theta_deg
        
        current_inner = [list(h) for h in inner]
        current_cx, current_cy, current_theta = cx, cy, theta_deg
        current_side = best_side
        
        for iteration in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            # Try very small perturbations in a structured way
            for _ in range(40):
                new_inner = [list(h) for h in current_inner]
                new_cx, new_cy = current_cx, current_cy
                new_theta = current_theta
                
                # Randomly perturb one parameter with smaller step
                r = random.random()
                if r < 0.6:
                    # Perturb a hexagon position
                    idx = random.randint(0, len(new_inner)-1)
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_inner[idx][0] += dx
                    new_inner[idx][1] += dy
                elif r < 0.8:
                    # Perturb outer center
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_cx += dx
                    new_cy += dy
                elif r < 0.9:
                    # Perturb outer angle
                    dtheta = random.uniform(-1.0, 1.0)
                    new_theta += dtheta
                else:
                    # Perturb a hexagon angle slightly
                    idx = random.randint(0, len(new_inner)-1)
                    dangle = random.uniform(-2.0, 2.0)
                    new_inner[idx][2] += dangle
                
                # Check disjointness
                if not hexagons_disjoint(new_inner):
                    continue
                
                # Compute new side length
                new_side = compute_min_outer_side(new_inner, new_cx, new_cy, new_theta)
                
                if new_side < current_side - 1e-9:
                    current_inner = new_inner
                    current_cx, current_cy, current_theta = new_cx, new_cy, new_theta
                    current_side = new_side
                    improved = True
                    step_size *= 0.97
                    break
            
            if not improved:
                step_size *= 0.85
                if step_size < 0.0005:
                    break
            
            # Update best
            if current_side < best_side - 1e-11:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_cx, best_cy, best_theta = current_cx, current_cy, current_theta
        
        return best_inner, best_cx, best_cy, best_theta, best_side

    # Lattice vectors for hexagonal packing
    # Distance between centers of touching unit hexagons is sqrt(3)
    # We add a small epsilon to ensure strict disjointness
    eps = 0.01
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    # Ring 0
    centers = [np.array([0.0, 0.0])]
    
    # Ring 1
    # Neighbors of center
    ring1_vectors = [
        u,
        v,
        v - u,
        -u,
        -v,
        u - v
    ]
    for vec in ring1_vectors:
        centers.append(vec.copy())
        
    # Ring 2 Candidates
    # Neighbors of Ring 1 that are not in Ring 0 or 1
    ring2_candidates = []
    # For each ring1 center, find its 6 neighbors
    # A neighbor of P is P +/- u, P +/- v, P +/- (v-u)
    basis = [u, v, v-u, -u, -v, u-v]
    
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            # Check if p2 is in Ring 0 or Ring 1
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                # Avoid duplicates in candidates list
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    # Sort Ring 2 candidates by distance to origin
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    # Pick 5 closest
    for i in range(5):
        centers.append(ring2_candidates[i].copy())
        
    # Now we have 12 centers
    # Verify count
    assert len(centers) == 12
    
    # Convert to list of [x, y, angle]
    # Rotate hexagons by 30 degrees so they are 'flat-top' relative to the x-axis,
    # allowing them to touch at distance sqrt(3) along the x-axis.
    inner = [[c[0], c[1], 30.0] for c in centers]
    
    # Find initial outer hexagon parameters
    all_verts = []
    for x, y, a in inner:
        verts = get_hex_vertices(x, y, 1.0, a)
        all_verts.extend(verts)
        
    cx = sum(p[0] for p in all_verts) / len(all_verts)
    cy = sum(p[1] for p in all_verts) / len(all_verts)
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Scan outer hexagon angle
    for theta_deg in np.linspace(0, 60, 61):
        s = compute_min_outer_side(inner, cx, cy, theta_deg)
        if s < best_side:
            best_side = s
            best_angle = theta_deg
            
    # Local optimization
    if time.time() < deadline - 15.0:
        inner_opt, cx_opt, cy_opt, theta_opt, side_opt = optimize_configuration(
            inner, cx, cy, best_angle, max_iter=200, step_size=0.03
        )
        if side_opt < best_side:
            inner = inner_opt
            cx, cy = cx_opt, cy_opt
            best_angle = theta_opt
            best_side = side_opt
    
    # High-precision polish pass
    if time.time() < deadline - 5.0:
        inner_pol, cx_pol, cy_pol, theta_pol, side_pol = polish_configuration(
            inner, cx, cy, best_angle, max_iter=400, step_size=0.005
        )
        if side_pol < best_side:
            inner = inner_pol
            cx, cy = cx_pol, cy_pol
            best_angle = theta_pol
            best_side = side_pol
    
    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
