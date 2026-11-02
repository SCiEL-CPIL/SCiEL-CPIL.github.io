import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time

N = 12

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0

    # ---------- geometry helpers ----------
    def hex_vertices(cx, cy, side, ang):
        a = ang + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + side*np.cos(a), cy + side*np.sin(a)])

    def sat_min_overlap(V1, V2):
        edges = np.roll(V1, -1, 0) - V1
        edges = np.vstack([edges, np.roll(V2, -1, 0) - V2])
        n = np.column_stack([edges[:,1], -edges[:,0]])
        nn = np.linalg.norm(n, axis=1, keepdims=True) + 1e-15
        n = n / nn
        p1 = V1 @ n.T; p2 = V2 @ n.T
        ov = np.minimum(p1.max(0), p2.max(0)) - np.maximum(p1.min(0), p2.min(0))
        return float(ov.min())

    def sat_penetration(V_inner, V_outer):
        edges = np.roll(V_outer, -1, 0) - V_outer
        n = np.column_stack([edges[:,1], -edges[:,0]])
        nn = np.linalg.norm(n, axis=1, keepdims=True) + 1e-15
        n = n / nn
        p_in = V_inner @ n.T
        p_out = (V_outer @ n.T).max(0)
        return float(np.maximum(0.0, np.max(p_in.max(0) - p_out)))

    # ---------- symmetric layout from 8 params ----------
    def expand_sym(params):
        """
        params = [s, r1, a1, r2, a2, r3, a3, r4, a4]
        - s: outer side length
        - r1,a1: center hex (r1=0 forced)
        - r2,a2: ring 1 (6 copies at 60° spacing)
        - r3,a3: ring 2 partial (6 copies)
        - r4,a4: ring 3 partial (6 copies, but we only use 5 to get 12 total)
        Total: 1 + 6 + 5 = 12
        """
        s = params[0]
        # Center hexagon at origin
        pos = np.zeros((N, 2))
        angles = np.zeros(N)
        pos[0] = [0.0, 0.0]
        angles[0] = params[1]  # rotation of center
        
        # Ring 1: 6 hexagons at radius r2, angles a2 + k*60°
        r2, a2 = params[2], params[3]
        for k in range(6):
            idx = 1 + k
            ang = a2 + np.deg2rad(60 * k)
            pos[idx] = [r2 * np.cos(ang), r2 * np.sin(ang)]
            angles[idx] = a2 + np.deg2rad(60 * k)  # rotate to align with radial direction
        
        # Ring 2: 5 hexagons at radius r3, angles a3 + k*60° (skip one for 5 total)
        r3, a3 = params[4], params[5]
        for k in range(5):
            idx = 7 + k
            ang = a3 + np.deg2rad(60 * k)
            pos[idx] = [r3 * np.cos(ang), r3 * np.sin(ang)]
            angles[idx] = a3 + np.deg2rad(60 * k)
        
        return pos, angles, s

    # ---------- objective with symmetry ----------
    def objective(params):
        if time.time() > DEADLINE:
            return 1e8
        
        pos, angles, s = expand_sym(params)
        
        if s < 2.0:
            return 1e6
        
        # Compute vertices for each hexagon
        base = np.array([[1.0,0.0],[0.5,np.sqrt(3)/2],[-0.5,np.sqrt(3)/2],
                         [-1.0,0.0],[-0.5,-np.sqrt(3)/2],[0.5,-np.sqrt(3)/2]])
        
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            c = np.cos(angles[i]); sn = np.sin(angles[i])
            R = np.array([[c,-sn],[sn,c]])
            verts[i] = pos[i][None,:] + (base @ R.T)
        
        outer = hex_vertices(0.0, 0.0, s, 0.0)
        
        # Containment check
        oedges = np.roll(outer,-1,0)-outer
        on = np.column_stack([oedges[:,1],-oedges[:,0]])
        on /= (np.linalg.norm(on,axis=1,keepdims=True)+1e-15)
        p_in = np.einsum('nij,jk->nik', verts, on.T)
        p_out = (outer @ on.T).max(0)
        contain = np.maximum(0.0, np.max(p_in.max(1)-p_out[None,:], axis=1))
        
        # Pairwise overlap
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                A = verts[i]; B = verts[j]
                eA = np.roll(A,-1,0)-A
                eB = np.roll(B,-1,0)-B
                ed = np.vstack([eA,eB])
                nn = np.column_stack([ed[:,1],-ed[:,0]])
                nn /= (np.linalg.norm(nn,axis=1,keepdims=True)+1e-15)
                pA = A@nn.T; pB = B@nn.T
                ov = np.minimum(pA.max(0),pB.max(0))-np.maximum(pA.min(0),pB.min(0))
                pen += max(0.0, ov.min())
        
        return s + 100.0*contain.sum() + 100.0*pen

    def is_valid_sym(params, tol=1e-4):
        pos, angles, s = expand_sym(params)
        base = np.array([[1.0,0.0],[0.5,np.sqrt(3)/2],[-0.5,np.sqrt(3)/2],
                         [-1.0,0.0],[-0.5,-np.sqrt(3)/2],[0.5,-np.sqrt(3)/2]])
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            c = np.cos(angles[i]); sn = np.sin(angles[i])
            R = np.array([[c,-sn],[sn,c]])
            verts[i] = pos[i][None,:] + (base @ R.T)
        outer = hex_vertices(0.0, 0.0, s, 0.0)
        for i in range(N):
            if sat_penetration(verts[i], outer) > tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                if sat_min_overlap(verts[i], verts[j]) > tol:
                    return False
        return True

    # ---------- bounds for DE ----------
    # [s, a_center, r1, a1, r2, a2, r3, a3]
    bounds = [(2.5, 5.0),  # s
              (0.0, 60.0),  # center angle
              (0.0, 0.0),   # r1 = 0 (center)
              (0.0, 60.0),  # a1
              (1.5, 4.0),   # r2 (ring 1 radius)
              (0.0, 60.0),  # a2
              (3.0, 6.0),   # r3 (ring 2 radius)
              (0.0, 60.0)]  # a3

    # ---------- run DE ----------
    best_params = None
    best_s = np.inf
    
    try:
        res = differential_evolution(objective, bounds, seed=42,
                                      maxiter=50, tol=1e-8, popsize=20,
                                      polish=True, init='sobol')
        if res.success or res.fun < best_s:
            if is_valid_sym(res.x):
                best_params = res.x.copy()
                best_s = res.x[0]
    except Exception:
        pass

    # ---------- basin hopping polish ----------
    if time.time() < DEADLINE - 15:
        if best_params is not None:
            from scipy.optimize import basinhopping
            try:
                bh_res = basinhopping(objective, best_params,
                                       minimizer_kwargs={'bounds': bounds},
                                       niter=30, T=0.05, stepsize=0.3,
                                       seed=123)
                if is_valid_sym(bh_res.x):
                    if bh_res.x[0] < best_s:
                        best_params = bh_res.x.copy()
                        best_s = bh_res.x[0]
            except Exception:
                pass

    # ---------- fallback ----------
    if best_params is None:
        # Use a known valid configuration
        best_params = np.array([3.5, 0.0, 0.0, 0.0, 2.5, 0.0, 4.5, 0.0])
        # Expand and verify
        pos, angles, s = expand_sym(best_params)
        if not is_valid_sym(best_params):
            # Enlarge until valid
            for s_try in np.arange(3.0, 6.0, 0.1):
                test_params = best_params.copy()
                test_params[0] = s_try
                if is_valid_sym(test_params):
                    best_params = test_params
                    break
            else:
                best_params = np.array([5.0, 0.0, 0.0, 0.0, 2.5, 0.0, 4.5, 0.0])

    pos, angles, s = expand_sym(best_params)

    inner = [[float(pos[i,0]), float(pos[i,1]), float(np.rad2deg(angles[i]))] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = float(s)
    outer_angle_deg = 0.0
    return inner, outer_center, outer_side, outer_angle_deg
