import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple lattice
        # Generate a valid lattice configuration
        axial = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        pts = np.array(pts)
        
        # Calculate required S
        # Outer hexagon centered at 0,0, rotation 0
        # Normals at 0, 60, 120, 180, 240, 300
        normals = np.array([
            [math.cos(math.radians(a)), math.sin(math.radians(a))]
            for a in [0, 60, 120, 180, 240, 300]
        ])
        
        max_proj = 0
        for n in normals:
            projs = np.abs(pts @ n)
            max_proj = max(max_proj, np.max(projs))
        
        S = (max_proj + APOTHEM) / APOTHEM
        # Actually, if D is max proj of center, the hexagon needs to cover the whole inner hexagon.
        # Inner hexagon has apothem A_inner = sqrt(3)/2.
        # So max distance from outer center to outer vertex of inner hexagon is D + A_inner?
        # No.
        # Distance from center to side of outer hexagon is A_outer.
        # Condition: Max over all points p in inner hex i of (p . n) <= A_outer.
        # Max p . n = (center_i . n) + A_inner.
        # So A_outer >= Max_i (center_i . n) + A_inner.
        # A_outer = S * sqrt(3) / 2.
        # So S * sqrt(3) / 2 >= Max_i D_i + sqrt(3) / 2.
        # S >= (Max_i D_i + sqrt(3)/2) / (sqrt(3)/2) = Max_i D_i / A_inner + 1.
        # Wait.
        # D_i is projection of center.
        # Max extent of hex i in direction n is D_i + A_inner.
        # So A_outer >= max_i (D_i + A_inner).
        # S * A_inner >= max_i D_i + A_inner.
        # S >= (max_i D_i + A_inner) / A_inner = max_i D_i / A_inner + 1.
        
        S_calc = max_proj / APOTHEM + 1.0
        
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        center = [0.0, 0.0]
        side = float(S_calc)
        rotation = 0.0
        return inner, center, side, rotation

    # Optimization
    # Variables: x0, y0, x1, y1, ... x10, y10
    # Fixed orientation (0 degrees)
    # Objective: Minimize S
    # S = (max_i D_i + APOTHEM) / APOTHEM
    
    # Initial guess from lattice
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    init_pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_pts.append([x, y])
    init_pts = np.array(init_pts)
    init_x = init_pts.flatten()
    
    # Normals for containment and overlap
    # For regular hexagons with flat sides aligned to axes?
    # Standard hexagon: Vertices at 0, 60, ...
    # Normals at 30, 90, ...
    # Let's use normals at 0, 60, ... for simplicity (pointy top)
    normals = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in [0, 60, 120, 180, 240, 300]
    ])
    
    def objective(params):
        pts = params.reshape(N, 2)
        
        # Containment term
        # D_i = max_k | pts[i] . n_k |
        projs = np.abs(pts @ normals.T) # Shape (N, 6)
        D_i = np.max(projs, axis=1)     # Shape (N,)
        max_D = np.max(D_i)
        
        # S term
        S_term = (max_D + APOTHEM) / APOTHEM
        
        # Overlap penalty
        # For each pair i, j
        overlap_penalty = 0.0
        # Min distance for non-overlap in direction n is 2 * APOTHEM = sqrt(3)
        min_dist = 2 * APOTHEM
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                # Max projection of diff
                d_proj = np.max(np.abs(diff @ normals.T))
                if d_proj < min_dist:
                    # Overlap
                    # Penalty is square of penetration
                    pen = min_dist - d_proj
                    overlap_penalty += pen * pen
        
        # Weights
        # If we make S small, overlaps increase.
        # We want to minimize S + lambda * penalty
        # But S is roughly 3.5. Penalty can be large.
        # Dynamic penalty?
        # Start with small penalty, increase if overlaps persist.
        # For simplicity, use a large weight to ensure validity, then minimize S.
        # But SLSQP might struggle with hard constraints.
        # Let's use a penalty that grows.
        # Or just minimize S + 10 * penalty.
        
        return S_term + 10.0 * overlap_penalty

    # Use Nelder-Mead
    best_result = None
    best_S = float('inf')
    
    # Multiple restarts
    restarts = 10
    for r in range(restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Perturb initial guess
        if r == 0:
            x0 = init_x
        else:
            noise = np.random.normal(0, 0.1, size=N*2)
            x0 = init_x + noise
            
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.success or res.fun < best_S:
            # Verify
            pts = res.x.reshape(N, 2)
            projs = np.abs(pts @ normals.T)
            D_i = np.max(projs, axis=1)
            max_D = np.max(D_i)
            S_val = (max_D + APOTHEM) / APOTHEM
            
            # Check overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    d_proj = np.max(np.abs(diff @ normals.T))
                    if d_proj < 2 * APOTHEM - 1e-6:
                        valid = False
                        break
                if not valid: break
            
            if valid and S_val < best_S:
                best_S = S_val
                best_result = res.x
                # Save best
                best_x = res.x

    if best_result is not None:
        pts = best_result.reshape(N, 2)
        # Recalculate S exactly
        projs = np.abs(pts @ normals.T)
        max_D = np.max(np.max(projs, axis=1))
        S_final = (max_D + APOTHEM) / APOTHEM
        
        # Add tiny safety margin
        S_final *= 1.000001
        
        inner = []
        for i in range(N):
            inner.append([float(pts[i, 0]), float(pts[i, 1]), 0.0])
        
        center = [0.0, 0.0]
        side = float(S_final)
        rotation = 0.0
        
        return inner, center, side, rotation
    else:
        # Fallback to initial lattice
        return construct() # Recursion risk? No, time check will trigger fallback logic if we structure it right.
        # Better to return the lattice calc directly.
        
# Refactor fallback to avoid recursion
def _get_lattice_solution():
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append([x, y])
    pts = np.array(pts)
    normals = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in [0, 60, 120, 180, 240, 300]
    ])
    projs = np.abs(pts @ normals.T)
    max_proj = np.max(np.max(projs, axis=1))
    S_calc = max_proj / APOTHEM + 1.0
    inner = [[float(x), float(y), 0.0] for x, y in pts]
    return inner, [0.0, 0.0], float(S_calc), 0.0

# Fix the construct function to use _get_lattice_solution
def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return _get_lattice_solution()

    N = 11
    APOTHEM = math.sqrt(3) / 2.0
    normals = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in [0, 60, 120, 180, 240, 300]
    ])
    
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    init_pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_pts.append([x, y])
    init_pts = np.array(init_pts)
    init_x = init_pts.flatten()
    
    def objective(params):
        pts = params.reshape(N, 2)
        projs = np.abs(pts @ normals.T)
        D_i = np.max(projs, axis=1)
        max_D = np.max(D_i)
        S_term = (max_D + APOTHEM) / APOTHEM
        
        overlap_penalty = 0.0
        min_dist = 2 * APOTHEM
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                d_proj = np.max(np.abs(diff @ normals.T))
                if d_proj < min_dist:
                    pen = min_dist - d_proj
                    overlap_penalty += pen * pen
        
        return S_term + 10.0 * overlap_penalty

    best_S = float('inf')
    best_x = init_x
    
    for r in range(10):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        x0 = init_x if r == 0 else init_x + np.random.normal(0, 0.1, size=N*2)
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000})
            pts = res.x.reshape(N, 2)
            projs = np.abs(pts @ normals.T)
            max_D = np.max(np.max(projs, axis=1))
            S_val = (max_D + APOTHEM) / APOTHEM
            
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    d_proj = np.max(np.abs(diff @ normals.T))
                    if d_proj < 2 * APOTHEM - 1e-6:
                        valid = False
                        break
                if not valid: break
            
            if valid and S_val < best_S:
                best_S = S_val
                best_x = res.x
        except Exception:
            pass

    if best_S < float('inf'):
        pts = best_x.reshape(N, 2)
        projs = np.abs(pts @ normals.T)
        max_D = np.max(np.max(projs, axis=1))
        S_final = (max_D + APOTHEM) / APOTHEM
        S_final *= 1.000001
        inner = [[float(pts[i, 0]), float(pts[i, 1]), 0.0] for i in range(N)]
        return inner, [0.0, 0.0], float(S_final), 0.0
    else:
        return _get_lattice_solution()

if __name__ == "__main__":
    construct()
