"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _hex_normals(cx, cy, angle_deg, side=1.0):
    """Returns the 6 outward normals of the hexagon."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    normals = []
    for k in range(6):
        # Normal to the edge between vertex k and k+1
        # Vertex k is at angle 60k, Vertex k+1 is at angle 60(k+1)
        # The midpoint of the edge is at angle 60k + 30
        # The normal points in that direction
        theta = math.radians(60 * k + 30)
        nx = math.cos(theta)
        ny = math.sin(theta)
        # Rotate normal by angle_deg
        rx = ca * nx - sa * ny
        ry = sa * nx + ca * ny
        normals.append((rx, ry))
    return normals

def _support_func(cx, cy, angle_deg, nx, ny, side=1.0):
    """Max of dot product of hex vertices with direction (nx, ny)."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    max_dot = 0.0
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        rx = ca * x - sa * y
        ry = sa * x + ca * y
        d = nx * rx + ny * ry
        if d > max_dot:
            max_dot = d
    return max_dot + nx * cx + ny * cy

def _objective(params):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, X, Y, Phi, S]
    """
    # Extract inner hexagons
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i + 1]
        a = params[3*i + 2]
        inner.append((x, y, a))
    
    # Extract outer hexagon
    X = params[3*N]
    Y = params[3*N + 1]
    Phi = params[3*N + 2]
    S = params[3*N + 3]
    
    if S < 0.1:
        S = 0.1
    if S > 100.0:
        S = 100.0

    # Calculate collision penalty
    col_penalty = 0.0
    for i in range(N):
        xi, yi, ai = inner[i]
        ni = _hex_normals(xi, yi, ai)
        for j in range(i + 1, N):
            xj, yj, aj = inner[j]
            nj = _hex_normals(xj, yj, aj)
            
            # Check all axes from both hexagons
            axes = ni + nj
            for (nx, ny) in axes:
                # Support functions
                # h_i(n) = max dot product of H_i with n
                # h_j(-n) = max dot product of H_j with -n = -min dot product of H_j with n
                # Gap g(n) = h_i(n) - h_j(n) + (c_j - c_i) . n ? No.
                # Let's use the standard projection intervals.
                # Proj_i = [min_i, max_i]
                # Proj_j = [min_j, max_j]
                # Overlap if max(min_i, min_j) < min(max_i, max_j)
                # Gap = max(min_i, min_j) - min(max_i, max_j)
                # If Gap > 0, they are separated.
                # We want to penalize Gap < 0 (overlap).
                
                min_i = float('inf')
                max_i = float('-inf')
                a_rad = math.radians(ai)
                ca, sa = math.cos(a_rad), math.sin(a_rad)
                for k in range(6):
                    theta = math.radians(60 * k)
                    vx = xi + ca * math.cos(theta) - sa * math.sin(theta)
                    vy = yi + sa * math.cos(theta) + ca * math.sin(theta)
                    d = nx * vx + ny * vy
                    if d < min_i: min_i = d
                    if d > max_i: max_i = d
                
                min_j = float('inf')
                max_j = float('-inf')
                a_rad_j = math.radians(aj)
                ca_j, sa_j = math.cos(a_rad_j), math.sin(a_rad_j)
                for k in range(6):
                    theta = math.radians(60 * k)
                    vx = xj + ca_j * math.cos(theta) - sa_j * math.sin(theta)
                    vy = yj + sa_j * math.cos(theta) + ca_j * math.sin(theta)
                    d = nx * vx + ny * d
                    if d < min_j: min_j = d
                    if d > max_j: max_j = d

                # Calculate gap
                # If intervals [min_i, max_i] and [min_j, max_j] overlap, gap is negative.
                # gap = max(min_i, min_j) - min(max_i, max_j)
                gap = max(min_i, min_j) - min(max_i, max_j)
                if gap < 0:
                    col_penalty += gap * gap

    # Calculate containment penalty
    cont_penalty = 0.0
    # Outer hexagon normals
    outer_normals = []
    phi_rad = math.radians(Phi)
    cphi, sphi = math.cos(phi_rad), math.sin(phi_rad)
    for k in range(6):
        theta = math.radians(60 * k + 30)
        nx = math.cos(theta)
        ny = math.sin(theta)
        rx = cphi * nx - sphi * ny
        ry = sphi * nx + cphi * ny
        outer_normals.append((rx, ry))
    
    apothem_outer = S * math.sqrt(3) / 2.0
    
    for i in range(N):
        xi, yi, ai = inner[i]
        verts = _hex_vertices(xi, yi, ai)
        for (vx, vy) in verts:
            for (nx, ny) in outer_normals:
                # Distance of vertex from origin in direction n
                val = nx * vx + ny * vy
                # Limit is X*nx + Y*ny + apothem_outer
                limit = nx * X + ny * Y + apothem_outer
                if val > limit:
                    excess = val - limit
                    cont_penalty += excess * excess

    # Objective: Minimize S + penalties
    # Weights
    lambda_col = 10.0
    lambda_cont = 10.0
    
    return S + lambda_col * col_penalty + lambda_cont * cont_penalty

def construct():
    t0 = time.time()
    random.seed(42)
    
    best_score = float('inf')
    best_params = None
    
    # Initial guesses
    # 1. Flower pattern
    def make_flower():
        params = np.zeros(3*N + 4)
        # Center
        params[0] = 0.0
        params[1] = 0.0
        params[2] = 0.0
        # 6 around
        r = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            idx = 3 + 3*i
            params[idx] = r * math.cos(angle)
            params[idx+1] = r * math.sin(angle)
            params[idx+2] = 0.0
        # 5 more further out
        r2 = r * 1.8
        for i in range(5):
            angle = math.radians(60 * i + 30)
            idx = 3 + 3*(6+i)
            params[idx] = r2 * math.cos(angle)
            params[idx+1] = r2 * math.sin(angle)
            params[idx+2] = 0.0
        
        # Outer hexagon
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        params[3*N+3] = 10.0 # Initial large side
        return params

    # 2. Random
    def make_random():
        params = np.zeros(3*N + 4)
        for i in range(N):
            params[3*i] = random.uniform(-5, 5)
            params[3*i+1] = random.uniform(-5, 5)
            params[3*i+2] = random.uniform(0, 60)
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        params[3*N+3] = 10.0
        return params

    initializers = [make_flower, make_random, make_random, make_random]
    
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 30:
            break
            
        x0 = init_func()
        
        # Define bounds
        # Inner hexagons: x, y in [-10, 10], angle in [0, 60]
        # Outer: X, Y in [-10, 10], Phi in [0, 60], S in [1, 20]
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))
            bounds.append((-10, 10))
            bounds.append((0, 60))
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 60))
        bounds.append((1, 20))
        
        # Optimize
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        best_params = make_flower()
        best_score = _objective(best_params)

    # Extract final result
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        a = best_params[3*i + 2]
        inner.append([x, y, a])
    
    center = (best_params[3*N], best_params[3*N + 1])
    side = best_params[3*N + 3]
    angle = best_params[3*N + 2]
    
    return inner, center, side, angle

