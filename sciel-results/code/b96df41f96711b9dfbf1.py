"""Packing 12 unit hexagons in a regular hexagon - Vectorized SA + L-BFGS-B polish."""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices and normals
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.radians(np.arange(30, 360, 60))
    unit_hex_normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))

    def get_all_vertices(positions, angles_rad):
        """
        positions: (N, 2)
        angles_rad: (N,)
        returns: (N, 6, 2)
        """
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        # (N, 6)
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
        
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        return all_verts

    def get_all_normals(angles_rad):
        """
        angles_rad: (N,)
        returns: (N, 6, 2)
        """
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        ux = unit_hex_normals[:, 0]
        uy = unit_hex_normals[:, 1]
        
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
        
        all_normals = np.stack((rx, ry), axis=-1)
        return all_normals

    def check_disjointness(positions, angles_rad):
        """
        Returns True if all hexagons are disjoint.
        Uses vectorized SAT where possible.
        """
        verts = get_all_vertices(positions, angles_rad) # (N, 6, 2)
        normals = get_all_normals(angles_rad) # (N, 6, 2)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.0: # dist > 2.0 (diameter)
                    continue
                
                v_i = verts[i]
                v_j = verts[j]
                
                disjoint = False
                # Check normals of i
                for k in range(6):
                    n = normals[i, k]
                    proj_i = np.dot(v_i, n)
                    proj_j = np.dot(v_j, n)
                    if np.max(proj_i) < np.min(proj_j) or np.max(proj_j) < np.min(proj_i):
                        disjoint = True
                        break
                
                if disjoint:
                    continue
                    
                # Check normals of j
                for k in range(6):
                    n = normals[j, k]
                    proj_i = np.dot(v_i, n)
                    proj_j = np.dot(v_j, n)
                    if np.max(proj_i) < np.min(proj_j) or np.max(proj_j) < np.min(proj_i):
                        disjoint = True
                        break
                        
                if not disjoint:
                    return False
                    
        return True

    def compute_min_side_fast(inner, center, angle):
        """
        inner: list of (x, y, angle_deg)
        center: (x, y)
        angle: float (degrees)
        """
        positions = np.array([(p[0], p[1]) for p in inner])
        angles_rad = np.radians(np.array([p[2] for p in inner]))
        all_verts = get_all_vertices(positions, angles_rad).reshape(-1, 2)
        offsets = all_verts - np.array(center)
        
        phi_outer_rad = np.radians(angle)
        c_o = np.cos(phi_outer_rad)
        s_o = np.sin(phi_outer_rad)
        
        ux = unit_hex_normals[:, 0]
        uy = unit_hex_normals[:, 1]
        
        rx = ux * c_o - uy * s_o
        ry = ux * s_o + uy * c_o
        outer_normals = np.column_stack((rx, ry)) # (6, 2)
        
        max_extent = 0.0
        for k in range(6):
            n = outer_normals[k]
            proj = np.dot(offsets, n)
            max_proj = np.max(proj)
            if max_proj > max_extent:
                max_extent = max_proj
                
        side = (2.0 / math.sqrt(3)) * max_extent
        return side

    def objective(params):
        """
        params: (3*N + 4,)
        [x0, y0, a0, ..., x11, y11, a11, cx, cy, phi_outer]
        """
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles_deg[i] = params[3*i+2]
            
        cx = params[3*N]
        cy = params[3*N+1]
        phi_outer_deg = params[3*N+3]
        
        angles_rad = np.radians(angles_deg)
        
        if not check_disjointness(positions, angles_rad):
            return 1e6
            
        inner = [(positions[i,0], positions[i,1], angles_deg[i]) for i in range(N)]
        side = compute_min_side_fast(inner, (cx, cy), phi_outer_deg)
        return side

    def smooth_objective(params, beta=50.0):
        """
        Smooth surrogate: soft-max containment + quadratic separation penalty.
        """
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles_deg[i] = params[3*i+2]
            
        cx = params[3*N]
        cy = params[3*N+1]
        phi_outer_deg = params[3*N+3]
        
        angles_rad = np.radians(angles_deg)
        
        all_verts = get_all_vertices(positions, angles_rad).reshape(-1, 2)
        center = np.array([cx, cy])
        offsets = all_verts - center
        
        phi_outer_rad = np.radians(phi_outer_deg)
        c_o = np.cos(phi_outer_rad)
        s_o = np.sin(phi_outer_rad)
        
        ux = unit_hex_normals[:, 0]
        uy = unit_hex_normals[:, 1]
        
        rx = ux * c_o - uy * s_o
        ry = ux * s_o + uy * c_o
        outer_normals = np.column_stack((rx, ry)) # (6, 2)
        
        max_extent = 0.0
        for k in range(6):
            n = outer_normals[k]
            proj = np.dot(offsets, n)
            # Soft-max
            max_e = (1.0/beta) * np.log(np.sum(np.exp(beta * proj)))
            if max_e > max_extent:
                max_extent = max_e
        
        side = (2.0 / math.sqrt(3)) * max_extent
        
        # Separation penalty
        sep_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 2.8:
                    sep_penalty += (2.8 - dist_sq) ** 2
        
        return side + 10.0 * sep_penalty

    def smooth_objective_grad(params, beta=50.0):
        """
        Numerical gradient of smooth objective.
        """
        eps = 1e-7
        grad = np.zeros_like(params)
        f0 = smooth_objective(params, beta)
        for i in range(len(params)):
            params[i] += eps
            f1 = smooth_objective(params, beta)
            params[i] -= eps
            grad[i] = (f1 - f0) / eps
        return grad

    def initial_guess():
        """
        Honeycomb-like initial guess.
        1 center, 6 ring, 5 ring 2.
        """
        params = np.zeros(3*N + 4)
        positions = []
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
        params[3*N+3] = 0.0
        return params

    best_params = initial_guess()
    best_score = objective(best_params)
    
    current_params = best_params.copy()
    current_score = smooth_objective(best_params, 50.0)
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    step_size = 0.5
    max_steps = 50000
    
    step_count = 0
    while time.time() < deadline - 15.0 and step_count < max_steps:
        step_count += 1
        new_params = current_params.copy()
        
        idx = random.randint(0, 3*N + 3)
        if idx == 3*N + 2:
            continue
            
        current_step = step_size * (0.5 + 0.5 * T)
        new_params[idx] += random.gauss(0, current_step)
        
        # Use smooth objective for SA to allow passing through overlaps
        new_score = smooth_objective(new_params, 50.0)
        
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            
            # Only update best if it's actually valid (hard check)
            # To save time, we check every 10th step or when score improves significantly
            if step_count % 10 == 0 or new_score < best_score - 0.1:
                hard_score = objective(current_params)
                if hard_score < best_score:
                    best_score = hard_score
                    best_params = current_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final hard check on best_params
    final_hard_score = objective(best_params)
    if final_hard_score < best_score:
        best_score = final_hard_score

    # L-BFGS-B polish phase
    if time.time() < deadline - 5.0 and best_score < 1e5:
        # Try polishing from best solution
        start_params = best_params.copy()
        try:
            result = minimize(
                smooth_objective,
                start_params,
                method='L-BFGS-B',
                jac=smooth_objective_grad,
                args=(50.0,),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < 1e5:
                polished = result.x
                polished_score = objective(polished)
                if polished_score < best_score:
                    best_score = polished_score
                    best_params = polished
        except Exception:
            pass

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+3])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final validation
    valid = True
    positions = np.array([(p[0], p[1]) for p in inner])
    angles_rad = np.radians(np.array([p[2] for p in inner]))
    if not check_disjointness(positions, angles_rad):
        valid = False
    
    if not valid:
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
