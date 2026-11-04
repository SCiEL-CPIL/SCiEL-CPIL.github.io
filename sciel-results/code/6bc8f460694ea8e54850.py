import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3)

def construct():
    sqrt3 = SQRT3
    centers = [[0.0, 0.0]]
    for k in range(6):
        ang = k * np.pi / 3
        centers.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)])
    for k in range(5):
        ang = k * 2 * np.pi / 5
        centers.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang)])
    
    x0 = []
    for c in centers:
        x0.extend([c[0], c[1], 0.0])
    x0.append(5.5)
    
    def compute_penalty(v):
        R = v[-1]
        inradius = R * SQRT3 / 2
        penalty = 0.0
        for i in range(N):
            xi = v[2*i]
            yi = v[2*i+1]
            ti = v[2*N + i]
            for k in range(6):
                ang = ti + k * np.pi / 3
                vx = xi + np.cos(ang)
                vy = yi + np.sin(ang)
                for m in range(6):
                    n_ang = np.pi/2 + m * np.pi / 3
                    proj = vx * np.cos(n_ang) + vy * np.sin(n_ang)
                    if proj > inradius:
                        penalty += (proj - inradius) ** 2
        for i in range(N):
            for j in range(i+1, N):
                dx = v[2*i] - v[2*j]
                dy = v[2*i+1] - v[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2
        return R + 50 * penalty
    
    result = minimize(compute_penalty, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-3, 'fatol': 1e-3})
    
    v = result.x
    R = v[-1]
    
    inner = []
    for i in range(N):
        xi = v[2*i]
        yi = v[2*i+1]
        ti = v[2*N + i]
        inner.append([xi, yi, np.degrees(ti)])
    
    return inner, (0.0, 0.0), R, 0.0
