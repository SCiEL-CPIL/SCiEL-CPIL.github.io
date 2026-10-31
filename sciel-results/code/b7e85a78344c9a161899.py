"""
Circle packing in a perimeter-4 rectangle.
Uses a penalty-based optimization approach with L-BFGS-B.
Variables: centers (2N), width (1), radii (N).
Objective: Maximize sum of radii, penalizing overlaps and boundary violations.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 110.0  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    
    # Objective function for L-BFGS-B
    # params: [x1, y1, ..., xN, yN, w, r1, ..., rN]
    def objective(params):
        centers = params[:2*N].reshape(N, 2)
        w = params[2*N]
        radii = params[2*N+1:]
        
        # Constrain w to be in (0, 2)
        if w <= 0.01 or w >= 1.99:
            return 1e10
        h = 2.0 - w
        
        # Sum of radii to maximize (so minimize negative sum)
        val = -np.sum(radii)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    penalty += overlap**2
        
        # Penalty for boundary violations
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            # Distance to walls
            d_left = x
            d_right = w - x
            d_bottom = y
            d_top = h - y
            
            # Ensure circle is inside
            # If r > d, penalty
            if r > d_left: penalty += (r - d_left)**2
            if r > d_right: penalty += (r - d_right)**2
            if r > d_bottom: penalty += (r - d_bottom)**2
            if r > d_top: penalty += (r - d_top)**2
            
            # Also, centers should be inside the box (soft constraint)
            # If center is outside, big penalty
            if x < 0 or x > w or y < 0 or y > h:
                penalty += 100.0

        # Weight for penalties
        # As we progress, we might want to increase penalty weight, 
        # but for a single call, a fixed high weight is okay if we start with small radii.
        weight = 100.0
        return val + weight * penalty

    def run_optimization(initial_centers, initial_w, initial_radii):
        # Flatten initial params
        params0 = np.concatenate([initial_centers.flatten(), [initial_w], initial_radii])
        
        # Bounds
        # Centers: no strict bounds, but penalty handles it. 
        # However, L-BFGS-B is better with bounds.
        # Let's bound centers to [0, 2] and [0, 2] roughly.
        # w: [0.1, 1.9]
        # radii: [0, 1.0]
        
        bounds = []
        for _ in range(2*N):
            bounds.append((0.0, 2.0))
        bounds.append((0.1, 1.9))
        for _ in range(N):
            bounds.append((0.0, 1.0))
            
        res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.fun < 1e6:
            # Extract best radii
            best_radii = res.x[2*N+1:]
            # Check validity: recompute max possible radii for the found centers/box
            # to ensure we don't return an invalid configuration if penalty wasn't perfect
            best_centers = res.x[:2*N].reshape(N, 2)
            best_w = res.x[2*N]
            best_h = 2.0 - best_w
            
            # Clamp centers to box
            best_centers[:, 0] = np.clip(best_centers[:, 0], 0, best_w)
            best_centers[:, 1] = np.clip(best_centers[:, 1], 0, best_h)
            
            # Calculate actual max radii
            final_radii = np.zeros(N)
            for i in range(N):
                x, y = best_centers[i]
                max_r = min(x, best_w - x, y, best_h - y)
                
                for j in range(N):
                    if i != j:
                        d = np.linalg.norm(best_centers[i] - best_centers[j])
                        max_r = min(max_r, d / 2.0)
                        
                final_radii[i] = max(0, max_r)
            
            # If the optimized radii are significantly larger than the geometrically possible ones,
            # it means the penalty allowed overlap. We must use the geometrically valid radii.
            # The score is sum of radii.
            return np.sum(final_radii), best_centers, final_radii, best_w
            
        else:
            return 0, initial_centers, initial_radii, initial_w

    best_score = 0
    best_result = None
    
    # Strategy: Try a few different initial configurations
    # 1. Grid layout
    # 2. Random layouts
    
    attempts = 0
    max_attempts = 5 # Keep low to respect time
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE:
            break
            
        if attempt == 0:
            # Grid 5x5 (20 circles) + 1 in center? Or 4x5+1?
            # 21 circles. 5x5 grid has 25. We need 21.
            # Let's use a 5x5 grid but remove 4 corners? Or just use 21 in a 5x5 grid.
            side = 5
            centers = []
            radii = []
            # Fill first 21 cells of a 5x5 grid
            for i in range(side):
                for j in range(side):
                    if len(centers) < N:
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        centers.append([x, y])
                        # Initial radius estimate: half cell size
                        radii.append(0.5 / side * 0.9)
            
            centers = np.array(centers)
            radii = np.array(radii)
            # Scale to fit in a square of size 1? 
            # Perimeter 4 -> if square, side=1.
            # Our grid is in [0,1]x[0,1]. So w=1, h=1.
            w = 1.0
        else:
            # Random initialization
            # Try different aspect ratios
            w = 1.0 + (attempt - 1) * 0.2 # 1.2, 1.4, 1.6, 1.8
            if w > 1.9: w = 1.9
            h = 2.0 - w
            
            centers = np.random.rand(N, 2) * np.array([w, h])
            # Initial radii small
            radii = np.full(N, 0.05)
            
        # Run optimization
        score, c, r, w_opt = run_optimization(centers, w, radii)
        
        if score > best_score:
            best_score = score
            best_result = (c, r, w_opt)
            
        # Check time
        if time.time() > DEADLINE - 10:
            break

    if best_result is None:
        # Fallback to seed-like solution
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    c, r, w_opt = best_result
    # Construct output array: (x, y, r)
    result = np.column_stack((c, r))
    return result
