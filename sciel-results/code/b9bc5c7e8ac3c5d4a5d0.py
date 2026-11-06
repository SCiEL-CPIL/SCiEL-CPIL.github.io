import numpy as np
from scipy.optimize import minimize
import time

_t0 = time.time()
_DEADLINE = 110.0

def _pts(p):
    x1,y1,x2,y2,x3,y3 = p
    return np.array([
        [-1.0,0.0],[1.0,0.0],
        [x1,y1],[x1,-y1],
        [x2,y2],[x2,-y2],
        [x3,y3],[x3,-y3]
    ], dtype=float)

def _neglogV_and_grad(p):
    pts = _pts(p)
    G = np.zeros((8,2))
    F = 0.0
    for i in range(8):
        for j in range(i+1,8):
            dx = pts[i,0]-pts[j,0]; dy = pts[i,1]-pts[j,1]
            d2 = dx*dx+dy*dy
            if d2 < 1e-14:
                return 1e18, np.zeros(6)
            F += -np.log(np.sqrt(d2))
            G[i,0] += dx/d2; G[i,1] += dy/d2
            G[j,0] += -dx/d2; G[j,1] += -dy/d2
    g = np.zeros(6)
    g[0] = -(G[2,0]+G[3,0])
    g[1] = -(G[2,1]-G[3,1])
    g[2] = -(G[4,0]+G[5,0])
    g[3] = -(G[4,1]-G[5,1])
    g[4] = -(G[6,0]+G[7,0])
    g[5] = -(G[6,1]-G[7,1])
    return F, g

def _neglogV(p):
    return _neglogV_and_grad(p)[0]

def _make_constraints():
    cons = []
    for i in range(8):
        for j in range(i+1,8):
            def f(p, i=i, j=j):
                pts = _pts(p)
                dx = pts[i,0]-pts[j,0]; dy = pts[i,1]-pts[j,1]
                return 2.0 - np.sqrt(dx*dx+dy*dy)
            cons.append({'type':'ineq','fun':f})
    return cons

_CONS = _make_constraints()

def _run_slsqp(p0, maxiter=500, tol=1e-10):
    res = minimize(_neglogV, p0, jac=_neglogV_and_grad,
                   method='SLSQP', constraints=_CONS,
                   options={'maxiter':maxiter,'ftol':tol})
    return res

def _score(pts):
    # product over ordered pairs / 8^8
    n = 8
    V = 1.0
    for i in range(n):
        for j in range(i+1,n):
            dx = pts[i,0]-pts[j,0]; dy = pts[i,1]-pts[j,1]
            V *= np.sqrt(dx*dx+dy*dy)
    return (V*V)/(n**n)

def construct():
    global _t0
    best_p = None
    best_F = np.inf
    # starts
    starts = []
    # regular octagon
    s2 = np.sqrt(2)/2
    starts.append([s2, s2, 0.0, 1.0, -s2, s2])
    # random starts
    rng = np.random.default_rng(12345)
    for _ in range(40):
        p = []
        for _ in range(3):
            r = 0.9*np.sqrt(rng.random())
            th = rng.random()*2*np.pi
            p.append(r*np.cos(th)); p.append(r*np.sin(th))
        starts.append(p)
    
    for p0 in starts:
        if time.time() - _t0 > _DEADLINE - 10:
            break
        try:
            res = _run_slsqp(p0)
            if res.success or res.fun < best_F:
                # check feasibility
                pts = _pts(res.x)
                feas = True
                for i in range(8):
                    for j in range(i+1,8):
                        if np.hypot(pts[i,0]-pts[j,0],pts[i,1]-pts[j,1]) > 2.0+1e-6:
                            feas = False
                if feas and res.fun < best_F:
                    best_F = res.fun
                    best_p = res.x
        except Exception:
            pass
    
    if best_p is None:
        best_p = starts[0]
    
    # polish
    if time.time() - _t0 < _DEADLINE - 5:
        try:
            res = minimize(_neglogV, best_p, jac=_neglogV_and_grad,
                           method='SLSQP', constraints=_CONS,
                           options={'maxiter':2000,'ftol':1e-14})
            pts = _pts(res.x)
            feas = True
            for i in range(8):
                for j in range(i+1,8):
                    if np.hypot(pts[i,0]-pts[j,0],pts[i,1]-pts[j,1]) > 2.0+1e-6:
                        feas = False
            if feas and res.fun < best_F:
                best_p = res.x
                best_F = res.fun
        except Exception:
            pass
    
    pts = _pts(best_p)
    # ensure distinct
    for i in range(8):
        for j in range(i+1,8):
            if np.hypot(pts[i,0]-pts[j,0],pts[i,1]-pts[j,1]) < 1e-6:
                pts[j,1] += 1e-6
    return [[float(pts[i,0]), float(pts[i,1])] for i in range(8)]
