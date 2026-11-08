import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 11

def _min_outer_S(points, phi_deg=0.0):
    normals = np.array([[np.cos(np.radians(30+k*60)), np.sin(np.radians(30+k*60))] for k in range(6)])
    phi = np.radians(phi_deg)
    verts = []
    for p in points:
        for k in range(6):
            theta = phi + k*np.pi/3
            verts.append([p[0]+np.cos(theta), p[1]+np.sin(theta)])
    verts = np.array(verts)
    c = np.array([0.,0.,1.])
    A = np.zeros((len(verts)*6, 3))
    b = np.zeros(len(verts)*6)
    idx = 0
    for v in verts:
        for k in range(6):
            A[idx] = [normals[k,0], normals[k,1], -np.sqrt(3)/2]
            b[idx] = np.dot(v, normals[k])
            idx += 1
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(None,None)]*2+[(0,None)], method='highs')
    return res.fun if res.success else 1e10

def construct():
    t0 = time.time()
    dl = t0 + 100.0
    
    d = np.sqrt(3)
    axial = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),(2,-1),(2,-2),(1,-2),(-1,2)]
    p0 = np.array([[d*(q+r/2), d*r*np.sqrt(3)/2] for q,r in axial])
    
    def obj(pf):
        if time.time() > dl:
            return 1e10
        P = pf.reshape(N,2)
        for i in range(N):
            for j in range(i+1,N):
                dx,dy = P[i,0]-P[j,0], P[i,1]-P[j,1]
                if dx*dx+dy*dy < 3.0-1e-9:
                    return 1e10
        return _min_outer_S(P, 0.0)
    
    best_S = obj(p0.flatten())
    best_p = p0.copy()
    
    if time.time() < dl - 10.0:
        try:
            res = minimize(obj, p0.flatten(), method='Nelder-Mead', options={'maxiter':5000,'xatol':1e-8,'fatol':1e-8})
            if res.fun < best_S and res.fun < 1e9:
                best_S = res.fun
                best_p = res.x.reshape(N,2)
        except Exception:
            pass
    
    inner = [[best_p[i,0], best_p[i,1], 0.0] for i in range(N)]
    return inner, (0.0,0.0), best_S, 0.0
