import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute all C(14,3) = 364 triangle index triples
_idx = np.arange(N)
_II, _JJ, _KK = np.meshgrid(_idx, _idx, _idx, indexing='ij')
_mask = (_II < _JJ) & (_JJ < _KK)
_I = _II[_mask].ravel()
_J = _JJ[_mask].ravel()
_K = _KK[_mask].ravel()

def _min_area(pts):
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _all_areas(pts):
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)

def _hull_area(pts):
    try:
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
        h = ConvexHull(pts)
        return h.volume
    except Exception:
        return 0.0

def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return _min_area(pts) / ha

def _sym_pts(half):
    return np.vstack([half, -half])

def _sym_score(half):
    return _score(_sym_pts(half))

def _objective_sym(flat):
    half = flat.reshape(N_HALF, 2)
    if np.any(~np.isfinite(half)):
        return 1e10
    s = _sym_score(half)
    return -s

def _soft_objective_sym(flat, beta=30.0):
    half = flat.reshape(N_HALF, 2)
    if np.any(~np.isfinite(half)):
        return 1e10
    pts = _sym_pts(half)
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    mz = np.max(z)
    if mz < -500:
        log_sum = mz
    else:
        log_sum = mz + np.log(np.sum(np.exp(z - mz)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e10
    return ha / soft_min

def _push_pull(half, max_iter=500):
    """Push-pull local search: identify worst triangle, push its vertices outward."""
    best = half.copy()
    best_s = _sym_score(best)
    for it in range(max_iter):
        pts = _sym_pts(best)
        areas = _all_areas(pts)
        worst_idx = np.argmin(areas)
        wi, wj, wk = _I[worst_idx], _J[worst_idx], _K[worst_idx]
        
        centroid = np.mean(pts, axis=0)
        improved_any = False
        for idx in [wi, wj, wk]:
            if idx < N_HALF:
                hi = idx
            else:
                hi = idx - N_HALF
            p = best[hi]
            c = centroid if idx < N_HALF else -centroid
            direction = p - c
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = np.array([1.0, 0.0])
            
            improved = False
            for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
                new_half = best.copy()
                new_half[hi] = best[hi] + step * direction
                new_s = _sym_score(new_half)
                if new_s > best_s + 1e-12:
                    best = new_half
                    best_s = new_s
                    improved = True
                    improved_any = True
                    break
            if not improved:
                for step in [0.02, 0.01, 0.005]:
                    new_half = best.copy()
                    new_half[hi] = best[hi] - step * direction
                    new_s = _sym_score(new_half)
                    if new_s > best_s + 1e-12:
                        best = new_half
                        best_s = new_s
                        improved = True
                        improved_any = True
                        break
        
        if not improved_any:
            rng = np.random.default_rng(it)
            for _ in range(20):
                new_half = best + rng.normal(0, 0.01, size=(N_HALF, 2))
                new_s = _sym_score(new_half)
                if new_s > best_s + 1e-12:
                    best = new_half
                    best_s = new_s
                    improved_any = True
                    break
        
        if not improved_any:
            break
    
    return best, best_s

def _nm_optimize(flat_init, maxiter=2000):
    try:
        res = minimize(_objective_sym, flat_init, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < 0:
            half = res.x.reshape(N_HALF, 2)
            s = -res.fun
            return half, s
    except Exception:
        pass
    return None, -np.inf

def _powell_optimize(flat_init, maxiter=1000):
    try:
        res = minimize(_objective_sym, flat_init, method='Powell',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'ftol': 1e-8})
        if res.fun < 0:
            half = res.x.reshape(N_HALF, 2)
            s = -res.fun
            return half, s
    except Exception:
        pass
    return None, -np.inf

def _lbfgs_soft_optimize(flat_init, maxiter=3000, beta=30.0):
    try:
        res = minimize(lambda x: _soft_objective_sym(x, beta=beta), flat_init, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        half = res.x.reshape(N_HALF, 2)
        s = _sym_score(half)
        return half, s
    except Exception:
        pass
    return None, -np.inf

def _adaptive_lbfgs(flat_init, maxiter_total=5000):
    """Adaptive beta: start smooth, increase sharpness."""
    best_half = flat_init.reshape(N_HALF, 2).copy()
    best_s = _sym_score(best_half)
    betas = [10.0, 30.0, 100.0, 300.0]
    for beta in betas:
        if time.time() > deadline_global - 5.0:
            break
        try:
            res = minimize(lambda x: _soft_objective_sym(x, beta=beta), best_half.flatten(),
                           method='L-BFGS-B', options={'maxiter': maxiter_total // len(betas),
                           'ftol': 1e-14, 'gtol': 1e-10})
            half = res.x.reshape(N_HALF, 2)
            s = _sym_score(half)
            if s > best_s:
                best_half = half.copy()
                best_s = s
        except Exception:
            pass
    return best_half, best_s

def _asymmetric_polish(pts_init, maxiter=2000):
    """Try small asymmetric perturbations from a symmetric solution."""
    best_pts = pts_init.copy()
    best_s = _score(best_pts)
    
    def _obj_asym(flat):
        pts = flat.reshape(N, 2)
        if np.any(~np.isfinite(pts)):
            return 1e10
        return -_score(pts)
    
    try:
        res = minimize(_obj_asym, best_pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        if res.fun < -best_s:
            pts = res.x.reshape(N, 2)
            s = -res.fun
            if s > best_s:
                best_pts = pts.copy()
                best_s = s
    except Exception:
        pass
    
    # Also try SLSQP with constraints
    try:
        res2 = minimize(_obj_asym, best_pts.flatten(), method='SLSQP',
                        options={'maxiter': 500, 'ftol': 1e-10})
        if res2.fun < -best_s:
            pts = res2.x.reshape(N, 2)
            s = -res2.fun
            if s > best_s:
                best_pts = pts.copy()
                best_s = s
    except Exception:
        pass
    
    return best_pts, best_s

deadline_global = 0.0

def construct():
    global deadline_global
    start = time.time()
    deadline_global = start + 130.0
    
    best_half = None
    best_score = -np.inf
    best_pts = None
    best_pts_score = -np.inf
    
    def update_half(half, score):
        nonlocal best_half, best_score
        if score > best_score:
            best_score = score
            best_half = half.copy()
    
    def update_pts(pts, score):
        nonlocal best_pts, best_pts_score
        if score > best_pts_score:
            best_pts_score = score
            best_pts = pts.copy()
    
    def remaining():
        return deadline_global - time.time()
    
    # Generate diverse initializations
    inits = []
    a7 = np.linspace(0, 2*np.pi, N_HALF, endpoint=False)
    inits.append(np.column_stack([np.cos(a7), np.sin(a7)]))
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        pert = np.column_stack([np.cos(a7), np.sin(a7)]) + rng.normal(0, 0.15, size=(N_HALF, 2))
        inits.append(pert)
    
    for rx, ry in [(1.0, 0.7), (1.1, 0.8), (0.9, 0.85), (1.2, 0.6), (1.0, 0.5)]:
        inits.append(np.column_stack([rx*np.cos(a7), ry*np.sin(a7)]))
    
    for r1, r2, off in [(0.3, 0.8, 0.0), (0.4, 0.7, 0.3), (0.5, 0.6, 0.5)]:
        a1 = np.linspace(-0.5, 0.5, 3) + off
        a2 = np.linspace(np.pi-0.5, np.pi+0.5, 4) + off
        p1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        inits.append(np.vstack([p1, p2]))
    
    for _ in range(4):
        r = np.sqrt(rng.uniform(0.1, 1.0, N_HALF))
        theta = rng.uniform(0, 2*np.pi, N_HALF)
        inits.append(np.column_stack([r*np.cos(theta), r*np.sin(theta)]))
    
    def halton_1d(i, base):
        f = 1.0
        r = 0.0
        while i:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    hp = np.array([[halton_1d(i, 2), halton_1d(i, 3)] for i in range(1, N_HALF+1)])
    inits.append(hp)
    
    # Phase 1: NM + Push-Pull on each init
    for init in inits:
        if remaining() < 20.0:
            break
        flat = init.flatten()
        
        half_nm, s_nm = _nm_optimize(flat, maxiter=1500)
        if half_nm is not None:
            update_half(half_nm, s_nm)
        
        if best_half is not None and remaining() > 15.0:
            half_pp, s_pp = _push_pull(best_half, max_iter=300)
            update_half(half_pp, s_pp)
        
        if remaining() > 15.0:
            half_pw, s_pw = _powell_optimize(flat, maxiter=500)
            if half_pw is not None:
                update_half(half_pw, s_pw)
    
    # Phase 2: Adaptive soft-min L-BFGS-B
    if remaining() > 15.0:
        start_points = []
        if best_half is not None:
            start_points.append(best_half.flatten())
        start_points.append(inits[0].flatten())
        start_points.append(inits[1].flatten())
        
        for sp in start_points:
            if remaining() < 10.0:
                break
            half_ad, s_ad = _adaptive_lbfgs(sp, maxiter_total=4000)
            update_half(half_ad, s_ad)
            
            if remaining() > 10.0:
                half_nm2, s_nm2 = _nm_optimize(half_ad.flatten(), maxiter=1000)
                if half_nm2 is not None:
                    update_half(half_nm2, s_nm2)
    
    # Phase 3: Iterative Push-Pull
    if best_half is not None and remaining() > 10.0:
        for _ in range(3):
            if remaining() < 8.0:
                break
            half_pp, s_pp = _push_pull(best_half, max_iter=400)
            update_half(half_pp, s_pp)
            
            if remaining() > 8.0:
                half_nm3, s_nm3 = _nm_optimize(best_half.flatten(), maxiter=1000)
                if half_nm3 is not None:
                    update_half(half_nm3, s_nm3)
    
    # Phase 4: Asymmetric polish from best symmetric solution
    if best_half is not None and remaining() > 10.0:
        sym_pts = _sym_pts(best_half)
        asym_pts, asym_s = _asymmetric_polish(sym_pts, maxiter=2000)
        update_pts(asym_pts, asym_s)
        
        # Try a few random asymmetric perturbations
        for _ in range(3):
            if remaining() < 8.0:
                break
            pert = sym_pts + rng.normal(0, 0.02, size=(N, 2))
            asym_pts2, asym_s2 = _asymmetric_polish(pert, maxiter=1000)
            update_pts(asym_pts2, asym_s2)
    
    # Phase 5: Final polish
    if best_half is not None and remaining() > 5.0:
        half_final, s_final = _push_pull(best_half, max_iter=500)
        update_half(half_final, s_final)
        
        if remaining() > 3.0:
            half_nm4, s_nm4 = _nm_optimize(best_half.flatten(), maxiter=800)
            if half_nm4 is not None:
                update_half(half_nm4, s_nm4)
    
    # Decide which result to return
    if best_pts is not None and best_pts_score > (best_score if best_half is not None else -np.inf):
        return best_pts
    elif best_half is not None:
        return _sym_pts(best_half)
    else:
        return _sym_pts(inits[0])
