import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140

def hex_width_in_direction(angle, phi):
    """Half-width of unit hexagon (rotation angle) in direction phi."""
    delta = np.mod(phi - angle, np.pi/3) - np.pi/6
    delta = np.abs(delta)
    if delta <= np.pi/6:
        return np.sqrt(3)/2 / np.cos(delta)
    else:
        return 1.0

def overlap_penalty(c1, a1, c2, a2):
    max_pen = 0.0
    for k in range(6):
        for base in [a1, a2]:
            n_ang = base + k*np.pi/3
            n = np.array([np.cos(n_ang), np.sin(n_ang)])
            phi = np.arctan2(n[1], n[0])
            w1 = hex_width_in_direction(a1, phi)
            w2 = hex_width_in_direction(a2, phi)
            d = (c2 - c1) @ n
            pen = w1 + w2 - abs(d)
            if pen > max_pen:
                max_pen = pen
    return max_pen

def boundary_penalty(c, a, S):
    verts_ang = a + np.arange(6) * np.pi/3
    verts = np.column_stack([c[0] + np.cos(verts_ang), c[1] + np.sin(verts_ang)])
    normals = np.column_stack([np.cos(np.pi/6 + np.arange(6) * np.pi/3),
                                np.sin(np.pi/6 + np.arange(6) * np.pi/3)])
    max_viol = 0.0
    for v in verts:
        proj = normals @ v
        viol = np.max(proj) - S * np.sqrt(3) / 2
        if viol > max_viol:
            max_viol = viol
    return max_viol

def cost(params, S):
    cost_val = 0.0
    for i in range(N):
        ci = params[3*i:3*i+2]
        ai = params[3*i+2]
        bp = boundary_penalty(ci, ai, S)
        cost_val += bp * bp
        for j in range(i+1, N):
            cj = params[3*j:3*j+2]
            aj = params[3*j+2]
            op = overlap_penalty(ci, ai, cj, aj)
            cost_val += op * op
    return cost_val

def initial_params(S):
    """Hexagonal lattice initial guess scaled to fit in S."""
    params = []
    # 4 columns, 3 rows on hex lattice
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * 3.0
            y = (row - 1.0) * 3 * np.sqrt(3) / 2
            params.extend([x, y, 0.0])
    return np.array(params)

def construct():
    start = time.time()
    
    S_lo = 4.0
    S_hi = 15.0
    best_params = initial_params(S_hi)
    
    for it in range(12):
        if time.time() > DEADLINE - 10:
            break
        S_mid = (S_lo + S_hi) / 2
        x0 = initial_params(S_mid)
        
        try:
            res = minimize(cost, x0, args=(S_mid,), method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12})
            if res.fun < 1e-5:
                S_hi = S_mid
                best_params = res.x
            else:
                S_lo = S_mid
        except Exception:
            S_lo = S_mid
    
    # Polish
    if time.time() < DEADLINE - 5:
        S = S_hi
        try:
            res = minimize(cost, best_params, args=(S,), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14})
            if res.fun < 1e-8:
                best_params = res.x
        except Exception:
            pass
    
    inner = []
    for i in range(N):
        x, y, a = best_params[3*i], best_params[3*i+1], np.degrees(best_params[3*i+2])
        inner.append([float(x), float(y), float(a)])
    
    return inner, (0.0, 0.0), float(S_hi), 0.0
