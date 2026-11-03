"""
Strategy: Smooth Gradient-Based Optimization with Simulated Annealing Restarts.

Key Differences from Current Program:
1. Objective Function: Uses a smooth approximation of the minimum distance between
   hexagons (via center distance minus inradius sum) instead of non-smooth SAT or
   penalty-based contact constraints. This makes the objective differentiable.
2. Optimizer: Uses L-BFGS-B (gradient-based) which is significantly faster and more
   effective for smooth, high-dimensional problems than Nelder-Mead.
3. Search Strategy: Employs a random restart schedule with increasing "temperature"
   (perturbation magnitude) to escape local minima, simulating a coarse-to-fine
   search.
4. Containment: Uses the exact support function for the outer hexagon, which is
   also smooth (max of linear functions), allowing gradients to guide the solution
   towards the boundary.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill

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
    # For a regular hexagon with side S centered at origin, aligned with axes:
    # The constraint for a point (lx, ly) to be inside is:
    # |lx| <= S
    # |lx + ly/sqrt3| <= S
    # |lx - ly/sqrt3| <= S
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-5):
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

def _smooth_objective(params):
    """
    Smooth objective: 
    1. Outer side length (smooth max of linear functions)
    2. Penalty for overlap: sum of max(0, min_dist - actual_dist)^2
       Using center distance as a proxy for min_dist (inradius sum = 2*sqrt(3)).
       This is differentiable and guides the optimizer to separate hexagons.
    """
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    
    # 1. Outer side length
    side = _min_outer_side(pos, ang, (0.0, 0.0), 0.0)
    
    # 2. Overlap penalty
    # Minimum distance between centers for non-overlapping unit hexagons is 2*sqrt(3)
    min_center_dist = 2 * np.sqrt(3)
    
    penalty = 0.0
    # Vectorized distance calculation
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_center_dist:
                # Smooth penalty: quadratic for violations
                penalty += (min_center_dist - d)**2
    
    # Weight the penalty heavily to enforce separation
    return side + 100.0 * penalty

def _initial_configuration():
    """
    Generate a specific initial configuration for 11 hexagons.
    Topology:
    - Center: (0,0)
    - Ring 1: 6 hexagons at distance 2*sqrt(3) from center.
    - Ring 2: 4 hexagons at distance 6 from center, in the gaps.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 0: Center
    pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    return pos, ang

def construct():
    global START_TIME, DEADLINE
    
    best_score = np.inf
    best_params = None
    
    # Initial configuration
    pos_init, ang_init = _initial_configuration()
    x0 = np.concatenate([pos_init.ravel(), ang_init])
    
    # Evaluate initial
    init_score = _smooth_objective(x0)
    if init_score < best_score:
        best_score = init_score
        best_params = x0.copy()
    
    # Simulated Annealing-like Random Restarts
    # We will perform multiple restarts with increasing perturbation.
    # The "temperature" controls the magnitude of random perturbations.
    
    num_restarts = 50
    base_temp = 0.5
    
    for i in range(num_restarts):
        if time.time() > DEADLINE - 10.0:
            break
            
        # Temperature decays or oscillates? Let's use a cooling schedule.
        # Start with high temp to explore, then cool down to refine.
        # However, since we do local optimization each time, we can keep temp moderate
        # to find diverse basins.
        
        # Randomly perturb the best solution found so far (or initial)
        # To ensure diversity, sometimes perturb the initial, sometimes the best.
        if i % 3 == 0:
            base = x0
        else:
            base = best_params
        
        temp = base_temp * (1.0 + np.random.rand())
        
        x_start = base + np.random.normal(0, temp, base.shape)
        
        # Local optimization with L-BFGS-B
        # This is much faster than Nelder-Mead for smooth functions
        try:
            res = minimize(_smooth_objective, x_start, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
                # Optional: Verify validity and update best if valid
                # But we trust the smooth objective to keep us valid if penalty is high enough
                # Let's do a quick validity check to be safe for the final return
        except Exception:
            pass
    
    # Final Refinement: Run a tighter L-BFGS-B on the best found
    if time.time() < DEADLINE - 5.0:
        try:
            res_final = minimize(_smooth_objective, best_params, method='L-BFGS-B',
                                 options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Extract best solution
    pos_best = best_params[:2*N].reshape(N, 2)
    ang_best = best_params[2*N:]
    
    inner_list = [[float(pos_best[i,0]), float(pos_best[i,1]), float(ang_best[i])] for i in range(N)]
    final_side = _min_outer_side(pos_best, ang_best, (0.0, 0.0), 0.0)
    
    # Final validity check
    if not _check_validity(inner_list, (0.0, 0.0), final_side, 0.0):
        # Fallback to initial configuration if invalid
        pos_f, ang_f = _initial_configuration()
        inner_list = [[float(pos_f[i,0]), float(pos_f[i,1]), float(ang_f[i])] for i in range(N)]
        final_side = _min_outer_side(pos_f, ang_f, (0.0, 0.0), 0.0)
        
    return inner_list, (0.0, 0.0), float(final_side), 0.0

if __name__ == "__main__":
    construct()
