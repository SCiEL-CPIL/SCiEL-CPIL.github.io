import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + np.radians(60 * i)
        verts.append([cx + s * np.cos(a), cy + s * np.sin(a)])
    return np.array(verts)

def sat_disjoint(v1, v2, tol=1e-10):
    """SAT test for two convex polygons. Returns True if disjoint."""
    def normals(v):
        ns = []
        for i in range(len(v)):
            e = v[(i+1)%len(v)] - v[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn > 1e-15:
                ns.append(n/nn)
        return ns
    
    for n in normals(v1) + normals(v2):
        p1 = v1 @ n
        p2 = v2 @ n
        if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
            return True
    return False

def all_disjoint(inner, tol=1e-10):
    vs = [hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner]
    for i in range(len(vs)):
        for j in range(i+1, len(vs)):
            if not sat_disjoint(vs[i], vs[j], tol):
                return False
    return True

def enclosing_hex(points, n_rot=8):
    """Min enclosing regular hexagon side, center, rotation."""
    best_s, best_c, best_r = np.inf, None, 0
    for rd in range(0, 180, 180//n_rot):
        angs = [np.radians(rd + 60*k) for k in range(6)]
        ns = np.array([[np.cos(a), np.sin(a)] for a in angs])
        proj = points @ ns.T
        pmin, pmax = proj.min(0), proj.max(0)
        
        # Binary search on s
        lo, hi = 0.0, np.max(pmax - pmin) / np.sqrt(3) * 1.5
        
        def feas(s):
            for k in range(3):
                l1 = pmax[k] - s*np.sqrt(3)/2
                l2 = -s*np.sqrt(3)/2 - pmin[k+3]
                u1 = pmin[k] + s*np.sqrt(3)/2
                u2 = s*np.sqrt(3)/2 - pmax[k+3]
                if max(l1, l2) > min(u1, u2):
                    return False
            return True
        
        for _ in range(25):
            mid = (lo+hi)/2
            if feas(mid): hi = mid
            else: lo = mid
        
        if hi < best_s:
            best_s = hi
            # Find center
            cv = []
            for k in range(3):
                l1 = pmax[k] - hi*np.sqrt(3)/2
                l2 = -hi*np.sqrt(3)/2 - pmin[k+3]
                u1 = pmin[k] + hi*np.sqrt(3)/2
                u2 = hi*np.sqrt(3)/2 - pmax[k+3]
                cv.append((max(l1,l2) + min(u1,u2))/2)
            A = np.array([[np.cos(np.radians(rd)), np.sin(np.radians(rd))],
                          [np.cos(np.radians(rd+60)), np.sin(np.radians(rd+60))]])
            try:
                best_c = np.linalg.solve(A, cv[:2])
            except:
                best_c = np.mean(points, 0)
            best_r = rd
    
    return best_s, best_c, best_r

def construct():
    t0 = time.time()
    dl = t0 + 100
    
    d = np.sqrt(3)
    pos = [[0,0]]
    for a in [0,60,120,180,240,300]:
        r = np.radians(a)
        pos.append([d*np.cos(r), d*np.sin(r)])
    for p in [[2*d,0],[d,d*np.sqrt(3)],[-d,d*np.sqrt(3)],[-2*d,0]]:
        pos.append(p)
    
    inner = [[p[0],p[1],0.0] for p in pos]
    
    av = np.vstack([hex_vertices(p[0],p[1],1.0,0.0) for p in pos])
    s0,c0,r0 = enclosing_hex(av, 8)
    
    best_inner, best_c, best_s, best_r = inner[:], [c0[0],c0[1]], s0, r0
    
    # Optimize positions
    def obj(params):
        it = [[params[2*i], params[2*i+1], 0.0] for i in range(11)]
        vs = [hex_vertices(r[0],r[1],1.0,r[2]) for r in it]
        pen = 0
        for i in range(11):
            for j in range(i+1,11):
                if not sat_disjoint(vs[i], vs[j], 1e-9):
                    # Estimate overlap
                    def norm(v):
                        ns=[]
                        for k in range(6):
                            e=v[(k+1)%6]-v[k]
                            n=np.array([-e[1],e[0]])
                            nn=np.linalg.norm(n)
                            if nn>1e-15: ns.append(n/nn)
                        return ns
                    mn=np.inf
                    for n in norm(vs[i])+norm(vs[j]):
                        p1=vs[i]@n; p2=vs[j]@n
                        mn=min(mn, min(p2.min()-p1.max(), p1.min()-p2.max()))
                    if mn < 0: pen += mn*mn
        if pen > 0.001: return 50+pen
        av2=np.vstack(vs)
        s,_,_ = enclosing_hex(av2, 4)
        return s
    
    x0=np.array([p[0] for p in pos]+[p[1] for p in pos])
    res=minimize(obj, x0, method='Nelder-Mead', options={'maxiter':200,'xatol':1e-3,'fatol':1e-3})
    
    if time.time() < dl and res.fun < best_s:
        ni=[[res.x[2*i],res.x[2*i+1],0.0] for i in range(11)]
        if all_disjoint(ni):
            av2=np.vstack([hex_vertices(r[0],r[1],1.0,r[2]) for r in ni])
            s,c,r=enclosing_hex(av2,8)
            if s<best_s:
                best_inner,best_c,best_s,best_r=ni,[c[0],c[1]],s,r
    
    return best_inner, best_c, best_s, best_r
