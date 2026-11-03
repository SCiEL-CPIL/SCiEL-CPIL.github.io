"""
Strategy: Simulated Annealing with Dynamic Container Constraint.

This approach fundamentally differs from the topology-driven constraint relaxation
by not fixing the contact graph. Instead, it uses a stochastic local search
(Simulated Annealing) that:
1. Starts with a loose outer hexagon.
2. Randomly perturbs positions and rotations of inner hexagons.
3. Accepts moves based on an energy function that penalizes overlaps heavily
   and the outer side length.
4. Gradually shrinks the target outer side length (annealing schedule).
5. This allows the system to escape local minima of fixed topologies and find
   denser, potentially non-symmetric packings.
"""

import time
import numpy as np
import random

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    # Generate all vertices for all hexagons
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    # Transform to outer hexagon coordinate system
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Support function for regular hexagon centered at origin
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-7):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    # Check overlaps using SAT
    def sat_overlap(v1, v2):
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i+1)%6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return False
        return True

    for i in range(N):
        for j in range(i+1, N):
            if sat_overlap(verts_list[i], verts_list[j]):
                return False
            
    # Check containment
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    def point_in_poly(pt, poly):
        x, y = pt
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    for verts in verts_list:
        for v in verts:
            if not point_in_poly(v, outer_verts):
                return False
    return True

def _energy(pos, ang, target_side, outer_center, outer_angle):
    """
    Energy function:
    1. Penalty for overlaps (hard constraint).
    2. Penalty for exceeding target_side (soft constraint during annealing).
    3. Actual outer side length (to minimize).
    """
    # Calculate actual outer side needed
    actual_side = _min_outer_side(pos, ang, outer_center, outer_angle)
    
    # Overlap penalty
    overlap_penalty = 0.0
    sqrt3 = np.sqrt(3)
    min_dist = 2 * sqrt3
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            # Simple distance proxy for overlap (less accurate than SAT but fast)
            # For hexagons, min distance between centers is 2*sqrt(3) if aligned, 
            # but can be less if rotated. However, for SA, a distance penalty 
            # based on a slightly smaller threshold helps push them apart.
            # A more robust check would be needed, but for speed we use distance.
            # Actually, let's use a conservative min distance for centers.
            # The minimum distance between centers of two non-overlapping unit hexagons
            # is 2*sqrt(3) * cos(pi/6) ? No, that's for specific orientations.
            # The inradius is sqrt(3)/2. The distance between parallel edges is sqrt(3).
            # Center to center distance for touching hexagons (same orientation) is 2*sqrt(3).
            # If rotated, the minimum distance can be smaller? No, the circumradius is 1.
            # The minimum distance between centers of two non-overlapping unit hexagons
            # is actually sqrt(3) + sqrt(3) = 2*sqrt(3) if they are aligned edge-to-edge.
            # Wait, if they are vertex-to-vertex, distance is 2.
            # So the minimum center distance is 2.
            # Let's use a penalty if d < 2.0 + small_epsilon.
            if d < 2.0:
                overlap_penalty += (2.0 - d) ** 2
    
    # Constraint penalty: if actual_side > target_side, penalize heavily
    constraint_penalty = 0.0
    if actual_side > target_side:
        constraint_penalty = (actual_side - target_side) ** 2 * 100.0
    
    return actual_side + overlap_penalty + constraint_penalty

def construct():
    global START_TIME, DEADLINE
    
    # Initial configuration: Hexagonal packing
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 0: Center
    pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2 (4 hexagons) - Place them in a way that fits
    # Let's try a symmetric arrangement for Ring 2
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        # Distance 6 was derived for touching ring 1.
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_side = _min_outer_side(best_pos, best_ang, (0.0, 0.0), 0.0)
    
    # Simulated Annealing
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Start with a loose target
    current_target = best_side * 1.2
    final_target = best_side * 0.95 # Aim to improve
    
    # Annealing parameters
    T = 1.0
    cooling_rate = 0.9995
    T_min = 0.01
    
    # Steps per temperature
    steps_per_T = 500
    
    # Random state for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    current_energy = _energy(pos, ang, current_target, outer_center, outer_angle)
    
    steps_done = 0
    
    while time.time() < DEADLINE:
        # Anneal
        while T > T_min:
            for _ in range(steps_per_T):
                if time.time() > DEADLINE - 5.0:
                    break
                    
                # Pick a random hexagon to move
                idx = np.random.randint(N)
                
                # Move parameters
                dp = np.random.normal(0, 0.1, 2)
                da = np.random.normal(0, 0.1)
                
                new_pos = pos.copy()
                new_ang = ang.copy()
                new_pos[idx] += dp
                new_ang[idx] += da
                
                new_energy = _energy(new_pos, new_ang, current_target, outer_center, outer_angle)
                delta_e = new_energy - current_energy
                
                # Metropolis criterion
                if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                    pos = new_pos
                    ang = new_ang
                    current_energy = new_energy
                    
                    # Update best if valid and better
                    # We only check validity occasionally to save time, 
                    # but for the final result we must be sure.
                    # For SA, we accept invalid states (high energy) to escape minima.
                    # But we track the best VALID state.
                    # Let's do a quick check: if energy is low, it's likely valid.
                    # We will do a full validity check every 100 steps or if energy is very low.
                    if steps_done % 100 == 0 or current_energy < 0.1:
                        # Check if this state is actually valid (no overlaps, inside target)
                        # Note: _energy uses a distance proxy for overlaps, so it's not perfect.
                        # We need a real check for the "best" tracking.
                        # To save time, we only do full SAT check if current_energy is very low.
                        if current_energy < 0.5:
                            inner_list = [[float(pos[i,0]), float(pos[i,1]), float(ang[i])] for i in range(N)]
                            # The target_side is the constraint, but we need the actual min side
                            actual_side = _min_outer_side(pos, ang, outer_center, outer_angle)
                            if _check_validity(inner_list, outer_center, actual_side, outer_angle):
                                if actual_side < best_side:
                                    best_pos = pos.copy()
                                    best_ang = ang.copy()
                                    best_side = actual_side
                                    
            T *= cooling_rate
            # Gradually tighten the target
            # Linear interpolation from start to end
            # This is a rough heuristic
            current_target = current_target * 0.999 + final_target * 0.001
            
            steps_done += steps_per_T
            
        # If we finish cooling, restart with a tighter target if we have time
        if time.time() < DEADLINE - 10.0:
            T = 1.0
            current_target = best_side * 1.05
            final_target = best_side * 0.98
        else:
            break
            
    # Final refinement with Nelder-Mead on the best found configuration
    if time.time() < DEADLINE - 5.0:
        x0 = np.concatenate([best_pos.ravel(), best_ang])
        
        def _objective_refine(params):
            n = N
            p = params[:2*n].reshape(n, 2)
            a = params[2*n:]
            side = _min_outer_side(p, a, outer_center, outer_angle)
            
            # Penalty for overlaps (using distance proxy for speed in optimizer)
            penalty = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    dx = p[i, 0] - p[j, 0]
                    dy = p[i, 1] - p[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 2.0:
                        penalty += 100.0 * (2.0 - d)**2
                        
            return side + penalty
        
        try:
            from scipy.optimize import minimize
            res = minimize(_objective_refine, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < _objective_refine(x0):
                x_opt = res.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                ang_opt = x_opt[2*N:]
                
                inner_list = [[float(pos_opt[i,0]), float(pos_opt[i,1]), float(ang_opt[i])] for i in range(N)]
                side_opt = _min_outer_side(pos_opt, ang_opt, outer_center, outer_angle)
                
                if _check_validity(inner_list, outer_center, side_opt, outer_angle):
                    if side_opt < best_side:
                        best_pos = pos_opt
                        best_ang = ang_opt
                        best_side = side_opt
        except Exception:
            pass

    # Finalize
    inner_list = [[float(best_pos[i,0]), float(best_pos[i,1]), float(best_ang[i])] for i in range(N)]
    final_side = _min_outer_side(best_pos, best_ang, outer_center, outer_angle)
    
    # Final validity check
    if not _check_validity(inner_list, outer_center, final_side, outer_angle):
        # Fallback to initial configuration
        pos_f = np.zeros((N, 2))
        ang_f = np.zeros(N)
        pos_f[0] = [0, 0]
        for i in range(6):
            a = i * np.pi / 3
            pos_f[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
        for i in range(4):
            a = ring2_angles[i]
            pos_f[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
            
        inner_list = [[float(pos_f[i,0]), float(pos_f[i,1]), float(ang_f[i])] for i in range(N)]
        final_side = _min_outer_side(pos_f, ang_f, outer_center, outer_angle)
        
    return inner_list, outer_center, float(final_side), outer_angle

if __name__ == "__main__":
    construct()
