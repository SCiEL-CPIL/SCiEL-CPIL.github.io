import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
sqrt3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def hex_vertices(cx, cy, theta):
        """Vertices of a unit regular hexagon centered at (cx,cy) with rotation theta."""
        angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        return cx + np.cos(angles), cy + np.sin(angles)

    def support_function(cx, cy, theta, outer_cx, outer_cy, outer_theta):
        """
        For a unit hexagon at (cx,cy) with rotation theta, compute the 6 support values
        relative to an outer hexagon centered at (outer_cx, outer_cy) with rotation outer_theta.
        The outer hexagon has side length L; we need L such that all vertices are inside.
        Return the required L for this hexagon.
        """
        vx, vy = hex_vertices(cx, cy, theta)
        # Outer hexagon normals (pointing outward from center)
        normal_angles = np.array([math.radians(30 + 60 * j) + outer_theta for j in range(6)])
        nc = np.cos(normal_angles)
        ns = np.sin(normal_angles)
        
        # Project vertices relative to outer center onto each normal
        rx = vx - outer_cx
        ry = vy - outer_cy
        projs = rx[:, np.newaxis] * nc[np.newaxis, :] + ry[:, np.newaxis] * ns[np.newaxis, :]
        max_proj = np.max(projs)
        # For a regular hexagon with side L, the apothem is L*sqrt(3)/2
        # So L = max_proj / (sqrt(3)/2)
        return max_proj / (sqrt3 / 2.0)

    def objective(params):
        """
        params: [x1,y1,theta1, ..., x12,y12,theta12, outer_cx, outer_cy, outer_theta, L]
        Minimize L + penalty for overlaps.
        """
        # Unpack
        centers = params[:36].reshape(12, 3)
        outer_cx = params[36]
        outer_cy = params[37]
        outer_theta = params[38]
        L = params[39]
        
        # Compute required L for each hexagon
        required_Ls = np.zeros(12)
        for i in range(12):
            required_Ls[i] = support_function(centers[i,0], centers[i,1], centers[i,2],
                                              outer_cx, outer_cy, outer_theta)
        
        max_required = np.max(required_Ls)
        
        # Overlap penalty: for each pair, compute minimum distance between hexagons
        # Use a simple proxy: center-to-center distance minus 2*inradius
        # Inradius of unit hexagon = sqrt(3)/2
        inradius = sqrt3 / 2.0
        min_center_dist = 2 * inradius  # = sqrt(3)
        
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_center_dist:
                    penalty += (min_center_dist - dist) ** 2
        
        # Objective: minimize L, with penalty for overlaps
        # Also penalize if L is less than required
        overlap_penalty = 0.0
        for i in range(12):
            if L < required_Ls[i]:
                overlap_penalty += (required_Ls[i] - L) ** 2
        
        return L + 100.0 * penalty + 100.0 * overlap_penalty

    def gradient(params):
        """Numerical gradient via finite differences."""
        eps = 1e-6
        grad = np.zeros_like(params)
        f0 = objective(params)
        for i in range(len(params)):
            params[i] += eps
            f1 = objective(params)
            grad[i] = (f1 - f0) / eps
            params[i] -= eps
        return grad

    # Initial configuration: 12 hexagons on a hexagonal lattice
    # Place them in a 4x3 grid pattern on hex lattice
    # Axial coordinates for a compact arrangement
    # Use a hexagonal cluster: center + 6 neighbors + 5 more
    # Actually, let's use a simple rectangular-ish arrangement on hex grid
    
    # Hex lattice positions: for axial (q,r), Cartesian: x = 1.5*q, y = sqrt3*(r + q/2)
    # Let's place 12 hexagons in a compact shape
    # A 3x4 arrangement in axial coords:
    axial_coords = []
    # Row 0: q=-1,0,1 ; r=0
    # Row 1: q=-1,0,1 ; r=1  
    # Row 2: q=-1,0,1 ; r=2
    # Row 3: q=-1,0,1 ; r=3
    # That's 12 positions
    
    for r in range(4):
        for q in range(-1, 2):
            axial_coords.append((q, r))
    
    # Convert to Cartesian
    init_centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        init_centers.append([x, y, 0.0])
    
    # Center the configuration
    xs = [c[0] for c in init_centers]
    ys = [c[1] for c in init_centers]
    cx_mean = sum(xs) / 12
    cy_mean = sum(ys) / 12
    for c in init_centers:
        c[0] -= cx_mean
        c[1] -= cy_mean
    
    # Initial params
    init_params = []
    for c in init_centers:
        init_params.extend([c[0], c[1], c[2]])
    init_params.extend([0.0, 0.0, 0.0, 10.0])  # outer center, angle, initial L
    
    # Try optimization
    best_L = float('inf')
    best_params = None
    
    # First, compute a good initial L
    init_L = 0.0
    for i in range(12):
        L_i = support_function(init_params[i*3], init_params[i*3+1], init_params[i*3+2],
                               0.0, 0.0, 0.0)
        if L_i > init_L:
            init_L = L_i
    init_params[39] = init_L
    
    if time.time() < deadline:
        result = minimize(objective, init_params, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 20000})
        if time.time() < deadline:
            if result.fun < best_L:
                best_L = result.fun
                best_params = result.x
    
    # If optimization didn't help, try a few random restarts
    if time.time() < deadline - 30:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb initial positions
            perturbed = list(init_params[:36])
            for i in range(len(perturbed)):
                perturbed[i] += np.random.normal(0, 0.3)
            
            # Recompute L
            L_est = 0.0
            for i in range(12):
                L_i = support_function(perturbed[i*3], perturbed[i*3+1], perturbed[i*3+2],
                                       0.0, 0.0, 0.0)
                if L_i > L_est:
                    L_est = L_i
            
            p = perturbed + [0.0, 0.0, 0.0, L_est]
            
            try:
                res = minimize(objective, p, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000})
                if time.time() > deadline - 5:
                    break
                if res.fun < best_L:
                    best_L = res.fun
                    best_params = res.x
            except Exception:
                continue
    
    # Extract best configuration
    if best_params is None:
        best_params = init_params
    
    # Compute the actual L from the best params
    centers = best_params[:36].reshape(12, 3)
    outer_cx = best_params[36]
    outer_cy = best_params[37]
    outer_theta = best_params[38]
    
    # Compute true L (max over all hexagons)
    true_L = 0.0
    for i in range(12):
        L_i = support_function(centers[i,0], centers[i,1], centers[i,2],
                               outer_cx, outer_cy, outer_theta)
        if L_i > true_L:
            true_L = L_i
    
    # Build inner hexagons
    inner = []
    for i in range(12):
        inner.append([centers[i,0], centers[i,1], math.degrees(centers[i,2])])
    
    return inner, (outer_cx, outer_cy), true_L, math.degrees(outer_theta)
