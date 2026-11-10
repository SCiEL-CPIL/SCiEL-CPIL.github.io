import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a combination of global search (Differential Evolution), symmetry reduction, and local refinement.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    def score(points):
        """Compute R = (d_min / d_max)^2"""
        if points is None or len(points) != n:
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(params):
        """
        Objective function to minimize: -d_min / d_max
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return -ratio

    def smooth_objective(params, beta=100.0):
        """
        Smooth surrogate using soft-min and soft-max via log-sum-exp.
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        
        # Soft min
        neg_dists = -beta * dists
        max_neg_dists = np.max(neg_dists)
        log_sum_exp_neg = max_neg_dists + np.log(np.sum(np.exp(neg_dists - max_neg_dists)))
        d_min_soft = -log_sum_exp_neg / beta
        
        # Soft max
        pos_dists = beta * dists
        max_pos_dists = np.max(pos_dists)
        log_sum_exp_pos = max_pos_dists + np.log(np.sum(np.exp(pos_dists - max_pos_dists)))
        d_max_soft = log_sum_exp_pos / beta
        
        if d_max_soft < 1e-10:
            return 0.0
            
        ratio = d_min_soft / d_max_soft
        return -ratio

    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        if not np.all(np.isfinite(pts)):
            return
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    def generate_hexagonal():
        """Generate 16 points in a hexagonal pattern."""
        h = np.sqrt(3) / 2
        pts = np.zeros((16, 2))
        for i in range(4):
            y = i * 2 * h
            for j in range(4):
                idx = i * 4 + j
                if i % 2 == 0:
                    x = j
                else:
                    x = j + 0.5
                pts[idx] = [x, y]
        return pts

    def generate_d2_symmetric(params):
        """
        Map 8 parameters (x1,y1, x2,y2, x3,y3, x4,y4) to 16 points with D2 symmetry.
        Points are (±xi, ±yi) for i=1..4.
        """
        pts = np.zeros((16, 2))
        for i in range(4):
            x = params[i*2]
            y = params[i*2+1]
            pts[i*4] = [x, y]
            pts[i*4+1] = [-x, y]
            pts[i*4+2] = [-x, -y]
            pts[i*4+3] = [x, -y]
        return pts

    def smooth_objective_d2(params, beta=100.0):
        """Smooth objective for D2 symmetric parameters."""
        pts = generate_d2_symmetric(params)
        return smooth_objective(pts.flatten(), beta)

    def refine(pts, label=""):
        """
        Multi-stage refinement with adaptive beta.
        """
        if time.time() > deadline - 20:
            return

        current = pts.copy()
        
        # Stage 1: Smooth L-BFGS-B with high beta (exploration)
        try:
            res1 = minimize(
                smooth_objective, 
                current.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res1.x is not None and np.all(np.isfinite(res1.x)):
                current = res1.x.reshape((n, 2))
        except Exception:
            pass

        if time.time() > deadline - 20:
            update_best(current)
            return

        # Stage 2: Smooth L-BFGS-B with lower beta (refinement)
        try:
            res2 = minimize(
                lambda p: smooth_objective(p, beta=50.0), 
                current.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                current = res2.x.reshape((n, 2))
        except Exception:
            pass

        if time.time() > deadline - 20:
            update_best(current)
            return

        # Stage 3: Raw L-BFGS-B on full 32 dims
        try:
            res3 = minimize(
                objective, 
                current.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res3.x is not None and np.all(np.isfinite(res3.x)):
                current = res3.x.reshape((n, 2))
        except Exception:
            pass

        if time.time() > deadline - 20:
            update_best(current)
            return

        # Stage 4: Powell (Derivative-free)
        try:
            res4 = minimize(
                objective, 
                current.flatten(), 
                method='Powell', 
                options={'maxiter': 150, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            if res4.x is not None and np.all(np.isfinite(res4.x)):
                current = res4.x.reshape((n, 2))
        except Exception:
            pass

        update_best(current)

    # 1. Hexagonal Initialization
    hex_pts = generate_hexagonal()
    refine(hex_pts, "hex")

    # 2. Differential Evolution on D2 symmetric space (8 parameters)
    # This explores the symmetry-reduced space globally
    np.random.seed(42)
    
    bounds_d2 = [(-3, 3)] * 8
    try:
        # DE is stochastic, so we run it with a reasonable maxiter and polish the result
        # Using popsize=15 for 8 dims is decent. maxiter=50 is fast.
        res_de = differential_evolution(
            smooth_objective_d2, 
            bounds_d2, 
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            update_interval=10
        )
        if res_de.x is not None and np.all(np.isfinite(res_de.x)):
            pts_de = generate_d2_symmetric(res_de.x)
            refine(pts_de, "de_d2")
    except Exception:
        pass

    # 3. Structured D2 starts
    d2_starts = [
        # Ring-like configuration
        np.array([1.0, 0.0, 0.5, 0.866, 0.866, 0.5, 0.0, 1.0]),
        # Square grid in first quadrant
        np.array([0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5]),
        # Radial spread
        np.array([0.3, 0.3, 1.0, 0.7, 1.5, 1.0, 2.0, 1.5]),
        # Hexagonal-inspired
        np.array([0.5, 0.0, 1.5, 0.0, 1.0, 0.866, 0.0, 1.0]),
        # Concentric
        np.array([0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0]),
    ]
    
    for start_params in d2_starts:
        if time.time() > deadline - 30:
            break
        try:
            res_sym = minimize(
                smooth_objective_d2, 
                start_params, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_sym.x is not None and np.all(np.isfinite(res_sym.x)):
                pts_sym = generate_d2_symmetric(res_sym.x)
                refine(pts_sym, "d2_structured")
        except Exception:
            pass
    
    # Random D2 restarts
    for _ in range(3):
        if time.time() > deadline - 30:
            break
        params_init = np.random.rand(8) * 4 + 0.2
        try:
            res_sym = minimize(
                smooth_objective_d2, 
                params_init, 
                method='L-BFGS-B', 
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_sym.x is not None and np.all(np.isfinite(res_sym.x)):
                pts_sym = generate_d2_symmetric(res_sym.x)
                refine(pts_sym, "d2_random")
        except Exception:
            pass

    # 4. Random Restarts (Full 32D)
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        rand_pts = np.random.rand(n, 2) * 6 - 3
        refine(rand_pts, f"rand_{i}")

    # 5. Final Polish on best found with very tight tolerances
    if best_points is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_final.x is not None and np.all(np.isfinite(res_final.x)):
                cand = res_final.x.reshape((n, 2))
                update_best(cand)
        except Exception:
            pass
        
        # One more Powell pass on the polished result
        if best_points is not None and time.time() < deadline - 5:
            try:
                res_powell = minimize(
                    objective, 
                    best_points.flatten(), 
                    method='Powell', 
                    options={'maxiter': 300, 'ftol': 1e-14, 'xtol': 1e-14}
                )
                if res_powell.x is not None and np.all(np.isfinite(res_powell.x)):
                    cand = res_powell.x.reshape((n, 2))
                    update_best(cand)
            except Exception:
                pass

    if best_points is None:
        return hex_pts

    return best_points
