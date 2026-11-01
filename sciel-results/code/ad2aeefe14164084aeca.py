import time
import numpy as np
from scipy.optimize import minimize

N = 11
DSTART = time.time()
DEADLINE = 150.0
MARGIN = 12.0
DMIN = np.sqrt(3.0)
GAP = 0.03
DREQ = DMIN + GAP
VERTS = np.arange(6) * (np.pi / 3.0)

def _h_inner(phi):
    return np.max(np.cos(phi - VERTS))

def _S_fixed(beta, o, c):
    n = np.stack([np.cos(beta + np.pi/6 + i*np.pi/3),
                  np.sin(beta + np.pi/6 + i*np.pi/3)] for i in range(6))
    phi = np.arctan2(n[:,1], n[:,0])
    h = _h_inner(phi)
    proj = c @ n.T
    M = proj.max(axis=0) + h
    S = ((M - o @ n.T) / (np.sqrt(3.0)/2.0)).max()
    return S

def _robust_S(beta, o, c):
    cands = [beta, beta + np.pi/6, -beta, -beta + np.pi/6]
    return max(_S_fixed(b, o, c) for b in cands)

def _objective(x):
    c = x[:22].reshape(11, 2)
    beta = x[22]
    o = c.mean(axis=0)
    S = _robust_S(beta, o, c)
    pen = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            d = np.linalg.norm(c[i] - c[j])
            if d < DREQ:
                pen += (DREQ - d) ** 2
    return S + 1000.0 * pen

def _lattice_cluster():
    e1 = np.array([DMIN, 0.0]); e2 = np.array([DMIN/2, 1.5])
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            p = i*e1 + j*e2
            pts.append((np.linalg.norm(p), p))
    pts.sort(key=lambda t: t[0])
    return np.array([t[1] for t in pts[:11]])

def _finalize(c, beta):
    o = c.mean(axis=0)
    S = _robust_S(beta, o, c)
    return c, o, S, beta

def _valid(c, o, S, beta):
    # disjointness
    for i in range(11):
        for j in range(i+1, 11):
            if np.linalg.norm(c[i]-c[j]) < DMIN - 1e-7:
                return False
    # containment under all 4 orientations
    for b in [beta, beta+np.pi/6, -beta, -beta+np.pi/6]:
        n = np.stack([np.cos(b+np.pi/6+i*np.pi/3), np.sin(b+np.pi/6+i*np.pi/3)] for i in range(6))
        # inner hex vertices
        for k in range(11):
            for v in range(6):
                p = c[k] + np.array([np.cos(v*np.pi/3), np.sin(v*np.pi/3)])
                if (n @ (p - o)) > S*np.sqrt(3.0)/2.0 + 1e-7:
                    return False
    return True

def construct():
    t_end = DSTART + DEADLINE - MARGIN
    base = _lattice_cluster()
    # fallback: scaled lattice to ensure disjoint
    # scale so min pair dist >= DREQ
    best = None
    # fallback config
    c_fb = base * 1.0
    # ensure disjoint by scaling
    def _minpair(c):
        m = 1e9
        for i in range(11):
            for j in range(i+1,11):
                m = min(m, np.linalg.norm(c[i]-c[j]))
        return m
    sc = DREQ / _minpair(c_fb) if _minpair(c_fb) < DREQ else 1.0
    c_fb = c_fb * max(sc, 1.0)
    c_fb, o_fb, S_fb, beta_fb = _finalize(c_fb, 0.0)
    best = (S_fb, c_fb, o_fb, beta_fb)

    rng = np.random.default_rng(0)
    restarts = 0
    while time.time() < t_end and restarts < 40:
        restarts += 1
        # initial cluster: base with perturbation, or a random-ish compact set
        c0 = base + rng.normal(0, 0.15, size=(11,2))
        beta0 = rng.uniform(0, np.pi/3)
        x0 = np.concatenate([c0.flatten(), [beta0]])
        res = minimize(_objective, x0, method='Nelder-Mead',
                       options={'maxiter': 1200, 'xatol': 1e-6, 'fatol': 1e-8})
        x = res.x
        c = x[:22].reshape(11,2); beta = x[22]
        # polish beta with c fixed
        betas = np.linspace(0, np.pi/3, 241)
        o = c.mean(axis=0)
        Ss = [_robust_S(b, o, c) for b in betas]
        beta = betas[int(np.argmin(Ss))]
        c, o, S, beta = _finalize(c, beta)
        if _valid(c, o, S, beta) and S < best[0]:
            best = (S, c, o, beta)
    S, c, o, beta = best
    # safety margin
    S = S + 0.02
    inner = [[float(c[k,0]), float(c[k,1]), 0.0] for k in range(11)]
    return inner, (float(o[0]), float(o[1])), float(S), float(np.degrees(beta))
