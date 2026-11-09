"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Structural change: Custom gradient descent with analytical overlap gradients,
hexagonal lattice initial configurations with diverse subset selections,
and joint optimization of inner positions, rotations, and outer hexagon angle.
"""
import numpy as np
import time
import random
import math

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _hex_edge_normals(verts):
    """Return unit normals for each edge of a convex polygon (vertices in CCW order)."""
    n = len(verts)
    normals = []
    for i in range(n):
        j = (i + 1) % n
        edge = verts[j] - verts[i]
        length = np.linalg.norm(edge)
        if length < 1e-12:
            normals.append(np.array([1.0, 0.0]))
        else:
            nx = -edge[1] / length
            ny = edge[0] / length
            normals.append(np.array([nx, ny]))
    return np.array(normals)

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0
    angle_rad = np.radians(outer_angle)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    
    # Precompute face normal angles: for a hexagon with vertices at angles 0,60,...,
    # the face normals are at 30, 90, 150, ... (offset by 30 degrees)
    face_normal_angles = [np.radians(30 + k * 60) for k in range(6)]
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            r = np.sqrt(lx * lx + ly * ly)
            if r < 1e-12:
                continue
            
            theta = np.arctan2(ly, lx)
            best_cos = 0.0
            for a in face_normal_angles:
                d = abs(((theta - a) % (2 * np.pi) + np.pi) % (2 * np.pi))
                d = min(d, 2 * np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c
            
            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r * best_cos / np.sqrt(3)
                best_side = max(best_side, required_side)
    
    return best_side

def _compute_overlap_penalty_and_grad(verts_list):
    """Compute smooth overlap penalty and its gradient w.r.t. all vertex positions.
    Returns (penalty, grad) where grad is shape (N, 6, 2)."""
    N_hex = len(verts_list)
    penalty = 0.0
    grad = np.zeros((N_hex, 6, 2))
    
    for i in range(N_hex):
        for j in range(i + 1, N_hex):
            v1 = verts_list[i]
            v2 = verts_list[j]
            normals1 = _hex_edge_normals(v1)
            normals2 = _hex_edge_normals(v2)
            all_normals = np.vstack([normals1, normals2])
            
            pair_penalty = 0.0
            for nx, ny in all_normals:
                n_vec = np.array([nx, ny])
                proj1 = v1 @ n_vec
                proj2 = v2 @ n_vec
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 <= min2 or max2 <= min1:
                    break  # Separated
                
                overlap = min(max1, max2) - max(min1, min2)
                pair_penalty += overlap * overlap
                
                # Gradient contribution: d(overlap^2)/d(pos) = 2*overlap * d(overlap)/d(pos)
                # d(overlap)/d(pos_k) depends on which vertex determines min/max
                # For simplicity, use a soft approximation
                # The overlap decreases when we move v1 in -n direction and v2 in +n direction
                grad[i] += 2.0 * overlap * (-n_vec) / 6.0  # distribute to all 6 vertices
                grad[j] += 2.0 * overlap * (n_vec) / 6.0
    
    return penalty, grad

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons are disjoint using SAT."""
    normals1 = _hex_edge_normals(verts1)
    normals2 = _hex_edge_normals(verts2)
    all_normals = np.vstack([normals1, normals2])
    
    for nx, ny in all_normals:
        n_vec = np.array([nx, ny])
        proj1 = verts1 @ n_vec
        proj2 = verts2 @ n_vec
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    
    return False

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """Check if configuration is valid."""
    cx, cy = outer_center
    angle_rad = np.radians(outer_angle)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    inradius = outer_side * np.sqrt(3) / 2
    
    # Check all inner hexagons are inside outer
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            for k in range(6):
                a = np.radians(30 + k * 60)
                nx = np.cos(a)
                ny = np.sin(a)
                if lx * nx + ly * ny > inradius + tol:
                    return False
    
    # Check no overlaps between inner hexagons
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i + 1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol):
                return False
    
    return True

def _hex_lattice_positions(subset_indices=None):
    """Generate hexagonal lattice positions for 11 hexagons.
    Uses flat-top hex grid: x = 1.5*q, y = sqrt(3)*(r + 0.5*q)
    Adjacent centers are sqrt(3) apart."""
    sqrt3 = np.sqrt(3)
    # Full hexagonal cluster of radius 2 has 19 positions
    # We need 11: center + 6 neighbors + 4 from the next ring
    all_axial = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                all_axial.append((q, r))
    
    # Sort by distance from center
    all_axial.sort(key=lambda x: abs(x[0]) + abs(x[1]) + abs(x[0] + x[1]))
    
    if subset_indices is not None:
        selected = [all_axial[i] for i in subset_indices if i < len(all_axial)]
    else:
        selected = all_axial[:11]
    
    centers = []
    for q, r in selected:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y, 0.0))
    
    return centers

def _diverse_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Standard hexagonal lattice (first 11 positions)
    configs.append(_hex_lattice_positions())
    
    # Different subset selections from the 19-position cluster
    # Center + 6 neighbors (7) + 4 specific from ring 2
    sqrt3 = np.sqrt(3)
    ring2 = [(2, 0), (2, -1), (1, -2), (0, -2), (-1, -1), (-2, 0),
             (-2, 1), (-1, 2), (0, 2), (1, 1), (2, -2), (2, 2)]
    
    for subset in [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14],
                   [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13],
                   [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                   [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]:
        if len(subset) == 11:
            cfg = _hex_lattice_positions(subset)
            if cfg and len(cfg) == 11:
                configs.append(cfg)
    
    # Linear chain with zigzag
    dist = 2.0 * np.sqrt(3)
    linear = [[0.0, 0.0, 0.0]]
    for i in range(1, N):
        x = i * dist * 0.85
        y = (i % 2) * dist * 0.4
        linear.append([x, y, 0.0])
    configs.append(linear)
    
    # Circular arrangement
    circular = [[0.0, 0.0, 0.0]]
    for i in range(1, N):
        a = 2 * np.pi * i / (N - 1)
        r = 3.2
        circular.append([r * np.cos(a), r * np.sin(a), 0.0])
    configs.append(circular)
    
    # Random configurations
    for seed in [42, 123, 456, 789, 1011]:
        rng = random.Random(seed)
        rand_cfg = [[0.0, 0.0, 0.0]]
        for i in range(1, N):
            r = rng.uniform(1.5, 4.5)
            a = rng.uniform(0, 2 * np.pi)
            angle = rng.uniform(0, 60)  # Keep rotations small
            rand_cfg.append([r * np.cos(a), r * np.sin(a), angle])
        configs.append(rand_cfg)
    
    return configs

def _gradient_descent(positions, outer_angle, max_iters=500, lr=0.05, penalty_weight=50.0):
    """Custom gradient descent to minimize outer side + overlap penalty."""
    pos = np.array(positions, dtype=float)
    best_pos = pos.copy()
    best_score = float('inf')
    
    for it in range(max_iters):
        if _remaining_time() < 5.0:
            break
        
        # Compute current objective
        verts_list = [_hex_vertices(pos[i, 0], pos[i, 1], 1.0, pos[i, 2]) for i in range(N)]
        side = _min_enclosing_side([tuple(p) for p in pos], (0.0, 0.0), outer_angle)
        overlap_penalty, overlap_grad = _compute_overlap_penalty_and_grad(verts_list)
        
        total = side + penalty_weight * overlap_penalty
        
        if total < best_score:
            best_score = total
            best_pos = pos.copy()
        
        # Compute gradient numerically (central differences) for positions
        grad = np.zeros_like(pos)
        eps = 1e-4
        
        for i in range(N):
            for d in range(3):  # x, y, angle
                pos_plus = pos.copy()
                pos_minus = pos.copy()
                pos_plus[i, d] += eps
                pos_minus[i, d] -= eps
                
                # Compute objective for plus
                verts_plus = [_hex_vertices(pos_plus[j, 0], pos_plus[j, 1], 1.0, pos_plus[j, 2]) for j in range(N)]
                side_plus = _min_enclosing_side([tuple(p) for p in pos_plus], (0.0, 0.0), outer_angle)
                op_plus, _ = _compute_overlap_penalty_and_grad(verts_plus)
                f_plus = side_plus + penalty_weight * op_plus
                
                # Compute objective for minus
                verts_minus = [_hex_vertices(pos_minus[j, 0], pos_minus[j, 1], 1.0, pos_minus[j, 2]) for j in range(N)]
                side_minus = _min_enclosing_side([tuple(p) for p in pos_minus], (0.0, 0.0), outer_angle)
                op_minus, _ = _compute_overlap_penalty_and_grad(verts_minus)
                f_minus = side_minus + penalty_weight * op_minus
                
                grad[i, d] = (f_plus - f_minus) / (2 * eps)
        
        # Update positions
        pos -= lr * grad
        
        # Keep angles in [0, 360)
        pos[:, 2] = pos[:, 2] % 360
        
        # Decay learning rate
        lr *= 0.995
    
    return [tuple(p) for p in best_pos], best_score

def _optimize_with_outer_angle(positions, outer_angle):
    """Optimize inner positions for a given outer angle, then try to improve outer angle."""
    if _remaining_time() < 10.0:
        return None, float('inf')
    
    # Gradient descent on inner positions
    opt_positions, opt_value = _gradient_descent(positions, outer_angle, max_iters=300)
    
    # Compute actual side length
    side = _min_enclosing_side(opt_positions, (0.0, 0.0), outer_angle)
    
    # Try to improve outer angle
    best_side = side
    best_angle = outer_angle
    for angle_offset in [-15, -10, -5, 5, 10, 15, -30, 30]:
        new_angle = (outer_angle + angle_offset) % 60
        new_side = _min_enclosing_side(opt_positions, (0.0, 0.0), new_angle)
        if new_side < best_side:
            best_side = new_side
            best_angle = new_angle
    
    # Check validity
    if _check_validity(opt_positions, (0.0, 0.0), best_side, best_angle, tol=1e-5):
        return opt_positions, best_side
    
    # If invalid, try with more separation
    for scale in [1.01, 1.02, 1.05, 1.1]:
        scaled_positions = [[p[0] * scale, p[1] * scale, p[2]] for p in opt_positions]
        side_s = _min_enclosing_side(scaled_positions, (0.0, 0.0), best_angle)
        if _check_validity(scaled_positions, (0.0, 0.0), side_s, best_angle, tol=1e-5):
            return scaled_positions, side_s
    
    return None, float('inf')

def construct():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    initial_configs = _diverse_initial_configs()
    outer_angles = [0.0, 15.0, 30.0, 45.0]
    
    # First pass: quick evaluation of all configs with all angles
    for oa in outer_angles:
        if _remaining_time() < 30.0:
            break
        
        for init_pos in initial_configs:
            if _remaining_time() < 15.0:
                break
            
            positions = [list(p) for p in init_pos]
            if len(positions) != N:
                continue
            
            result, side = _optimize_with_outer_angle(positions, oa)
            
            if result is not None and side < best_score:
                best_score = side
                best_result = (result, (0.0, 0.0), side, oa)
    
    # Second pass: refine best result with more iterations
    if best_result is not None and _remaining_time() > 40.0:
        refined_positions, refined_side = _optimize_with_outer_angle(
            [list(p) for p in best_result[0]], best_result[3])
        if refined_positions is not None and refined_side < best_score:
            best_score = refined_side
            best_result = (refined_positions, (0.0, 0.0), refined_side, best_result[3])
    
    if best_result is not None:
        return best_result
    
    # Fallback: known valid configuration with generous spacing
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 3.8
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = 6.5
    for k in range(4):
        a = np.radians(30 + 60 * k)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0
