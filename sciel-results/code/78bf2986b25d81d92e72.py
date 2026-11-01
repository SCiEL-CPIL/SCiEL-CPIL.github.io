import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start = time.time()
    deadline = start + 140.0

    def objective(x):
        if time.time() > deadline:
            return 1e10
        pos = x[:2*N].reshape(N, 2)
        oside = x[2*N]
        
        if oside < 1.0:
            return 1e10
        
        pen = 0.0
        outer_inradius = oside * np.sqrt(3) / 2
        
        for i in range(N):
            cx, cy = pos[i]
            for k in range(6):
                va = np.radians(60.0 * k)
                vx = cx + np.cos(va)
                vy = cy + np.sin(va)
                for m in range(6):
                    ma = np.radians(60.0 * m)
                    d = vx * np.cos(ma) + vy * np.sin(ma)
                    if d > outer_inradius:
                        pen += (d - outer_inradius) ** 2
        
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                if d < 2.0:
                    pen += (2.0 - d) ** 2
        
        return oside + 100.0 * pen

    sqrt3 = np.sqrt(3.0)
    init_pos = [
        [0.0, 0.0],
        [2.0, 0.0], [1.0, sqrt3], [-1.0, sqrt3],
        [-2.0, 0.0], [-1.0, -sqrt3], [1.0, -sqrt3],
        [3.0, 0.0], [2.0, sqrt3], [0.0, 2.0 * sqrt3], [-2.0, sqrt3]
    ]
    x0 = []
    for p in init_pos:
        x0.extend(p)
    x0.append(10.0)

    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})

    if time.time() > deadline:
        inner = [[init_pos[i][0], init_pos[i][1], 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 15.0, 0.0

    xopt = res.x
    pos = xopt[:2*N].reshape(N, 2)
    oside = max(xopt[2*N], 1.0)

    inner = [[pos[i, 0], pos[i, 1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), oside, 0.0
