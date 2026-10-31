import numpy as np
from scipy.optimize import linprog
import time

N = 11

def hexagon_vertices(cx, cy, s, rot_deg):
    rot = np.radians(rot_deg)
    return np.array([[cx + s*np.cos(rot + k*np.pi/3), cy + s*np.sin(rot + k*np.pi/3)] for k in range(6)])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    def make_cluster(rot_deg):
        base = []
        for i in range(3):
            base.append((i * sqrt3, 0.0))
        for i in range(4):
            base.append((sqrt3/2 + i * sqrt3, -1.5))
        for i in range(3):
            base.append((i * sqrt3, -3.0))
        
        xs = [p[0] for p in base]
        ys = [p[1] for p in base]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        positions = [(p[0]-cx, p[1]-cy) for p in base]
        
        rot = np.radians(rot_deg)
        c, s = np.cos(rot), np.sin(rot)
        return [(c*x - s*y, s*x + c*y, rot_deg) for (x,y) in positions]
    
    def min_outer_side(inner, outer_rot_deg):
        all_v = np.vstack([hexagon_vertices(x, y, 1.0, a) for x,y,a in inner])
        orot = np.radians(outer_rot_deg)
        normals = np.array([[np.cos(orot + np.pi/6 + k*np.pi/3), np.sin(orot + np.pi/6 + k*np.pi/3)] for k in range(6)])
        ov = np.array([[np.cos(orot + k*np.pi/3), np.sin(orot + k*np.pi/3)] for k in range(6)])
        h_out = np.max(ov @ normals.T, axis=1)
        
        nv = len(all_v)
        A = np.zeros((12*nv, 3))
        b = np.zeros(12*nv)
        for k in range(6):
            nk = normals[k]
            proj = all_v @ nk
            hk = h_out[k]
            A[6*k:6*k+nv, 0] = -nk[0]
            A[6*k:6*k+nv, 1] = -nk[1]
            A[6*k:6*k+nv, 2] = -hk
            b[6*k:6*k+nv] = -proj
            A[6*k+nv:6*k+2*nv, 0] = nk[0]
            A[6*k+nv:6*k+2*nv, 1] = nk[1]
            A[6*k+nv:6*k+2*nv, 2] = -hk
            b[6*k+nv:6*k+2*nv] = proj
        
        res = linprog([0,0,1], A_ub=A, b_ub=b, bounds=[(None,None),(None,None),(0,None)], method='highs')
        if res.success:
            return res.x[2], (res.x[0], res.x[1])
        return None, None
    
    best_s, best_inner, best_center, best_orot = float('inf'), None, None, 0
    
    for cr in range(0, 60, 2):
        if time.time() > deadline: break
        inner = make_cluster(cr)
        for orot in range(0, 60, 3):
            if time.time() > deadline: break
            s, c = min_outer_side(inner, orot)
            if s is not None and s < best_s:
                best_s, best_inner, best_center, best_orot = s, inner, c, orot
    
    if best_inner is None:
        return make_cluster(0), (0.0, 0.0), 10.0, 0.0
    return best_inner, best_center, best_s, best_orot
