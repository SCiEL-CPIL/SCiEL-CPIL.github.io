import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    # Precompute normals for the outer hexagon (aligned)
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Support of unit hexagon in these directions is apothem = sqrt(3)/2
    apothem_inner = np.sqrt(3) / 2
    
    # Initial positions: Hexagonal grid
    # Basis vectors for hex grid with spacing sqrt(3)
    d = np.sqrt(3)
    u = np.array([d, 0])
    v = np.array([d/2, d*np.sqrt(3)/2])
    
    # Generate grid points
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u + j * v
            points.append(p)
    
    # Sort by distance to origin to get the densest cluster
    points.sort(key=lambda p: np.dot(p, p))
    
    # Take first N points
    init_pts = np.array(points[:N])
    
    # Calculate a valid initial L based on the grid
    max_sup = 0.0
    for p in init_pts:
        for n in normals:
            sup = np.dot(n, p)
            if sup > max_sup:
                max_sup = sup
    L_init = 1.0 + max_sup / apothem_inner
    
    # Flatten variables: x0, y0, x1, y1, ..., L
    x0 = np.zeros(2*N + 1)
    for i in range(N):
        x0[2*i] = init_pts[i][0]
        x0[2*i+1] = init_pts[i][1]
    x0[2*N] = L_init
    
    def objective(vars):
        # L is the last variable
        return vars[2*N]
        
    def constraints(vars):
        L = vars[2*N]
        pts = vars[:2*N].reshape(N, 2)
        
        cons = []
        
        # 1. Containment Constraints
        # n . p <= (L-1) * apothem_inner
        # Add a small epsilon for safety
        eps_contain = 1e-6
        rhs = (L - 1) * apothem_inner - eps_contain
        
        for i in range(N):
            p = pts[i]
            for k in range(6):
                n = normals[k]
                # n . p - rhs <= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda p=p, n=n, rhs=rhs: rhs - np.dot(n, p)
                })
        
        # 2. Non-overlap Constraints
        # ||p_i - p_j||^2 >= 3 + epsilon
        # Use a slightly larger value to ensure strict non-overlap
        eps_overlap = 1e-6
        min_dist_sq = 3.0 + eps_overlap
        
        for i in range(N):
            for j in range(i+1, N):
                di = pts[i]
                dj = pts[j]
                cons.append({
                    'type': 'ineq',
                    'fun': lambda di=di, dj=dj: np.sum((di-dj)**2) - min_dist_sq
                })
                
        return cons

    # Run optimization
    best_L = L_init
    best_pts = init_pts
    start_time = time.time()
    deadline = 100.0 # Keep under 120s limit

    try:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                       options={'maxiter': 500, 'ftol': 1e-8})
        
        if time.time() > deadline:
            pass # Fall back to best found so far or initial
            
        if res.success or (time.time() < deadline and res.x[2*N] < best_L):
            # Check if the result is valid
            cand_L = res.x[2*N]
            cand_pts = res.x[:2*N].reshape(N, 2)
            
            valid = True
            # Check containment
            for p in cand_pts:
                for n in normals:
                    if np.dot(n, p) > (cand_L - 1) * apothem_inner + 1e-5:
                        valid = False
                        break
                if not valid: break
            
            # Check non-overlap
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        if np.sum((cand_pts[i]-cand_pts[j])**2) < 3.0 - 1e-5:
                            valid = False
                            break
                    if not valid: break
            
            if valid and cand_L < best_L:
                best_L = cand_L
                best_pts = cand_pts

    except Exception:
        pass

    # Construct output
    inner = []
    for p in best_pts:
        inner.append([p[0], p[1], 0.0])
    
    return inner, (0.0, 0.0), best_L, 0.0
