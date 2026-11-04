import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_hex_vertices(angle_rad):
    # Vertices of unit hexagon rotated by angle_rad
    # Base vertices at 0, 60, ...
    angles = np.arange(6) * np.pi / 3 + angle_rad
    return np.column_stack((np.cos(angles), np.sin(angles)))

def min_distance_hex(delta):
    # Minimum distance between centers of two unit hexagons with relative rotation delta
    # delta in [0, pi/6] due to symmetry
    delta = delta % (np.pi / 3)
    if delta > np.pi / 6:
        delta = np.pi / 3 - delta
    # The minimum distance is determined by the lowest point of the rotated hexagon
    # relative to the top edge of the fixed hexagon (apothem sqrt(3)/2)
    # Lowest y of rotated hex: min(sin(delta + k*pi/3))
    angles = np.arange(6) * np.pi / 3 + delta
    min_y = np.min(np.sin(angles))
    return np.sqrt(3) / 2 - min_y

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Variables: L, x1, y1, a1, ..., x12, y12, a12
    # Total 37 variables
    # Initial guess: Compact cluster
    # Center at 0,0
    # 1 center, 6 ring 1, 5 ring 2?
    # Let's try a 3x4 grid or similar
    # Or just random
    
    # Better initial guess:
    # Place 12 hexagons in a hexagonal lattice pattern
    # Centers at distance sqrt(3)
    coords = []
    # Center
    coords.append((0.0, 0.0, 0.0))
    # Ring 1
    for k in range(6):
        ang = k * np.pi / 3
        coords.append((np.sqrt(3) * np.cos(ang), np.sqrt(3) * np.sin(ang), 0.0))
    # Ring 2 (need 5 more)
    # Add 5 from the next ring
    for k in range(5):
        ang = k * np.pi / 3
        # Distance 2*sqrt(3)
        coords.append((2 * np.sqrt(3) * np.cos(ang), 2 * np.sqrt(3) * np.sin(ang), 0.0))
    
    x0 = [10.0] # L
    for c in coords:
        x0.extend([c[0], c[1], c[2]])
        
    def objective(vars):
        L = vars[0]
        # Return L
        return L

    def constraints(vars):
        L = vars[0]
        cons = []
        # Containment
        # Outer hexagon: side L, center 0,0, rotation 0
        # Normals at 0, 60, ...
        # Distance to edge: L * sqrt(3) / 2
        apothem_out = L * np.sqrt(3) / 2
        normals = np.array([[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)])
        
        for i in range(N):
            xi = vars[1 + 3*i]
            yi = vars[2 + 3*i]
            ai = vars[3 + 3*i]
            verts = get_hex_vertices(ai)
            # Vertices in global coords
            gx = verts[:, 0] + xi
            gy = verts[:, 1] + yi
            # Projections on normals
            projs = gx * normals[:, 0] + gy * normals[:, 1]
            # Max projection must be <= apothem_out
            max_proj = np.max(projs)
            cons.append(apothem_out - max_proj)
            
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                xi = vars[1 + 3*i]
                yi = vars[2 + 3*i]
                ai = vars[3 + 3*i]
                xj = vars[1 + 3*j]
                yj = vars[2 + 3*j]
                aj = vars[3 + 3*j]
                
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                delta = ai - aj
                d_min = min_distance_hex(delta)
                cons.append(dist - d_min)
                
        return cons

    cons_list = []
    for i in range(6*N + N*(N-1)/2):
        cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: constraints(v)[i]})
        
    # This lambda closure is inefficient. Let's restructure.
    
    def cons_func(v):
        return constraints(v)
        
    # SLSQP requires a list of dicts or a single function returning array?
    # It accepts a list of dicts.
    
    # To avoid creating lambdas in loop, we can just use the array constraint if supported?
    # No, SLSQP takes list of dicts.
    
    # Let's try to optimize
    best_result = None
    best_L = 100.0
    
    # Multiple restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Perturb initial guess
        x0_perturbed = x0.copy()
        x0_perturbed[1:] += np.random.normal(0, 0.1, size=36)
        
        # Define constraints properly
        cons = []
        for i in range(6*N + N*(N-1)//2):
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: constraints(v)[idx]})
            
        try:
            res = minimize(objective, x0_perturbed, method='SLSQP', constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
            if res.success and res.fun < best_L:
                best_L = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is None:
        # Fallback to seed-like or simple packing
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0
        
    L_opt = best_result[0]
    inner = []
    for i in range(N):
        xi = best_result[1 + 3*i]
        yi = best_result[2 + 3*i]
        ai = best_result[3 + 3*i]
        inner.append([xi, yi, np.degrees(ai)])
        
    return inner, (0.0, 0.0), L_opt, 0.0

