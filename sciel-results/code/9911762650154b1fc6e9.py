import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def construct():
    if not _time_ok():
        return [[0.0,0.0,0.0]]*11, [0.0,0.0], 10.0, 0.0

    # candidate axial lists (same as current, plus a few)
    candidates = [ ... ]  # list of lists of (q,r)

    def axial_to_cartesian(axial_list):
        return np.array([[math.sqrt(3)*(q + r/2.0), 1.5*r] for q,r in axial_list])

    def D_of_beta(pts, beta):
        angs = beta + math.pi/6.0 + np.arange(6)*math.pi/3.0
        normals = np.stack([np.cos(angs), np.sin(angs)], axis=1)
        projs = pts @ normals.T
        ranges = projs.max(axis=0) - projs.min(axis=0)
        return 0.5*float(ranges.max())

    def center_of_beta(pts, beta, D):
        angs = beta + math.pi/6.0 + np.arange(6)*math.pi/3.0
        normals = np.stack([np.cos(angs), np.sin(angs)], axis=1)
        projs = pts @ normals.T
        a = projs.min(axis=0); b = projs.max(axis=0)
        A_ub = np.vstack([normals, -normals])
        b_ub = np.concatenate([b + D, D - a])
        try:
            res = linprog(c=np.zeros(2), A_ub=A_ub, b_ub=b_ub, bounds=[(None,None),(None,None)], method='highs')
            if res.status == 0:
                return float(res.x[0]), float(res.x[1])
        except Exception:
            pass
        return float(pts[:,0].mean()), float(pts[:,1].mean())

    def optimize_1d(pts):
        best_beta = 0.0; best_D = float('inf')
        for i in range(150):
            beta = i*(math.pi/6)/149
            D = D_of_beta(pts, beta)
            if D < best_D:
                best_D = D; best_beta = beta
        try:
            res = minimize_scalar(lambda b: D_of_beta(pts, b), bounds=(0, math.pi/6), method='bounded', options={'maxiter': 800, 'xatol': 1e-13})
            if res.fun < best_D:
                best_D = float(res.fun); best_beta = float(res.x)
        except Exception:
            pass
        try:
            lo = max(0.0, best_beta - 0.02); hi = min(math.pi/6, best_beta + 0.02)
            if hi - lo > 1e-12:
                res = minimize_scalar(lambda b: D_of_beta(pts, b), bounds=(lo, hi), method='bounded', options={'maxiter': 800, 'xatol': 1e-15})
                if res.fun < best_D:
                    best_D = float(res.fun); best_beta = float(res.x)
        except Exception:
            pass
        return best_beta, best_D

    def polish_3d(pts, beta, cx, cy):
        def obj(p):
            cxp, cyp, betap = p
            betap = ((betap % (math.pi/6)) + math.pi/6) % (math.pi/6)
            angs = betap + math.pi/6.0 + np.arange(6)*math.pi/3.0
            normals = np.stack([np.cos(angs), np.sin(angs)], axis=1)
            projs = np.abs((pts - np.array([cxp,cyp])) @ normals.T)
            return (float(projs.max()) + APOTHEM)/APOTHEM
        best_S = obj([cx,cy,beta])
        best_x = [cx,cy,beta]
        starts = [[cx,cy,beta],[cx,cy,beta+0.01],[cx,cy,beta-0.01],[cx+0.02,cy,beta],[cx,cy+0.02,beta]]
        for s in starts:
            if not _time_ok(): break
            try:
                res = minimize(obj, s, method='Nelder-Mead', options={'maxiter': 800, 'xatol':1e-12,'fatol':1e-12})
                if res.fun < best_S:
                    best_S = float(res.fun); best_x = list(res.x)
            except Exception:
                pass
        cxp, cyp, betap = best_x
        betap = ((betap % (math.pi/6)) + math.pi/6) % (math.pi/6)
        return cxp, cyp, betap, best_S

    best_S = float('inf'); best_pts=None; best_beta=0.0; best_center=None
    for axial in candidates:
        if not _time_ok(): break
        try:
            pts = axial_to_cartesian(axial)
            beta, D = optimize_1d(pts)
            S = (D + APOTHEM)/APOTHEM
            if S < best_S:
                best_S = S; best_pts = pts; best_beta = beta
        except Exception:
            continue

    if best_pts is None:
        # fallback
        axial = candidates[0]
        best_pts = axial_to_cartesian(axial)
        best_beta = 0.0
        D = D_of_beta(best_pts, best_beta)
        best_S = (D + APOTHEM)/APOTHEM

    # polish the best
    cx0, cy0 = float(best_pts[:,0].mean()), float(best_pts[:,1].mean())
    try:
        cx, cy, beta, S = polish_3d(best_pts, best_beta, cx0, cy0)
        if S < best_S:
            best_S = S; best_beta = beta
    except Exception:
        pass

    D_best = best_S*APOTHEM - APOTHEM
    cx, cy = center_of_beta(best_pts, best_beta, D_best + 1e-9)
    angle_deg = math.degrees(best_beta)

    inner = [[float(x), float(y), float(angle_deg)] for x,y in best_pts]
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    return inner, center, side, rotation
