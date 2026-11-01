import numpy as np
import time
from scipy.optimize import minimize, basinhopping

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-10:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_smooth(x, tau):
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    exp_terms = np.exp((sq_dists - m_max) / tau)
    soft_max = m_max + tau * np.log(np.sum(exp_terms))
    
    # Soft min
    m_min = np.min(sq_dists)
    exp_terms_min = np.exp(-(sq_dists - m_min) / tau)
    soft_min = m_min - tau * np.log(np.sum(exp_terms_min))
    
    return soft_max - soft_min

def local_minimize(x0, tau_local=20.0):
    res = minimize(
        objective_smooth, 
        x0, 
        args=(tau_local,), 
        method='L-BFGS-B', 
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    return res.x, res.fun, res.nit

def construct():
    start_time = time.time()
    timeout = 140.0  # Leave 10s margin
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts

    # Initial candidates
    rng = np.random.default_rng(42)
    
    # 1. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 points. Try placing them at the "poles" of the icosahedron or random.
    # The icosahedron has 5-fold symmetry. The "poles" are not vertices.
    # Let's just add 2 random points near the surface.
    p1 = rng.normal(size=D) * 0.1
    p2 = rng.normal(size=D) * 0.1
    p1 = p1 / np.linalg.norm(p1) if np.linalg.norm(p1) > 1e-6 else np.array([1,0,0])
    p2 = p2 / np.linalg.norm(p2) if np.linalg.norm(p2) > 1e-6 else np.array([0,1,0])
    
    # Scale ico to unit sphere, p1, p2 to unit sphere
    pts_ico = np.vstack([ico, p1.reshape(1,3), p2.reshape(1,3)])
    candidates = [pts_ico.flatten()]
    
    # 2. Random points on sphere
    for _ in range(5):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Random points in box
    for _ in range(5):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    # Basin Hopping setup
    # Use a lower tau for the global search (smoother landscape)
    tau_search = 5.0
    # Use a higher tau for the local refinement (sharper landscape)
    tau_local = 20.0
    
    def local_func(x):
        x_min, f_min, nit = local_minimize(x, tau_local)
        return f_min, x_min, {'nit': nit}

    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout:
            break
            
        try:
            res = basinhopping(
                local_func,
                init_x,
                niter=10, # Number of basin hopping iterations
                stepsize=0.5,
                minimizer_kwargs={'method': 'L-BFGS-B'}, # This is ignored if local_func is a custom callable returning x
                # Actually, basinhopping expects a function that returns (f, x, info) if it's a "minimizer"
                # Or it can take a standard objective and a minimizer.
                # Let's use the standard interface:
                # objective: function(x) -> float
                # minimizer: function(x) -> result
                # But local_func above returns (f, x, info).
                # Let's define a proper local minimizer for basinhopping.
            )
        except Exception:
            pass

    # Let's re-implement the basin hopping loop manually or use scipy correctly.
    # scipy.optimize.basinhopping(func, x0, ...)
    # func is the objective.
    # minimizer is a callable that takes x0 and returns a result with .x and .fun.
    
    def bh_minimizer(x0):
        res = minimize(
            objective_smooth, 
            x0, 
            args=(tau_local,), 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return res

    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout:
            break
        
        try:
            res_bh = basinhopping(
                objective_smooth,
                init_x,
                args=(tau_search,), # Objective used for acceptance
                minimizer=bh_minimizer, # Local minimizer uses tau_local
                niter=15,
                stepsize=0.5,
                interval=10,
                disp=False
            )
            if res_bh.x is not None:
                update_best(res_bh.x.reshape(N, D))
        except Exception:
            pass

    # Fallback if basin hopping failed or was too slow
    if best_points is None:
        # Just polish the first candidate
        try:
            res = minimize(
                objective_smooth, 
                candidates[0], 
                args=(tau_local,), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                best_points = res.x.reshape(N, D)
                best_score = raw_score(best_points)
        except Exception:
            best_points = candidates[0].reshape(N, D)
            best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
