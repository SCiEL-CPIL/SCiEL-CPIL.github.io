import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    
    def get_diameter(pts):
        dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
        return np.max(dists)

    def calc_score(pts):
        dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, 1.0)
        # Ensure no zeros
        dists[dists < 1e-10] = 1e-10
        log_prod = np.sum(np.log(dists))
        D = np.max(dists)
        if D < 1e-9:
            return -1e10
        # Score is Delta / n^n where Delta is product of distances.
        # log(Delta) = log_prod.
        # log(Score) = log_prod - n*log(n) - n*log(D) ? 
        # Wait, the prompt says: "score is Delta/n^n ... after scaling the set to diameter 2".
        # If we scale to diameter 2, D becomes 2.
        # So Score = Delta_scaled / n^n.
        # Delta_scaled = Delta_raw * (2/D)^{n(n-1)/2}? No, product of distances.
        # If we scale coordinates by factor s, distances scale by s.
        # Product of n(n-1) ordered pairs? Or unordered?
        # "product over ordered pairs". There are n(n-1) ordered pairs.
        # So Delta scales by s^{n(n-1)}.
        # We want to scale such that D_new = 2. So s = 2/D.
        # Delta_new = Delta_raw * (2/D)^{n(n-1)}.
        # Score = Delta_new / n^n.
        # log(Score) = log(Delta_raw) + n(n-1)*log(2/D) - n*log(n).
        # log(Delta_raw) = sum_{i!=j} log(d_{ij}).
        # So we maximize: sum_{i!=j} log(d_{ij}) + n(n-1)*log(2) - n(n-1)*log(D) - n*log(n).
        # Constants don't matter for optimization.
        # Maximize: sum_{i!=j} log(d_{ij}) - n(n-1)*log(D).
        
        return log_prod - n*(n-1)*np.log(D)

    def objective_sym(params):
        # params: [x, y]
        x, y = params
        if x <= y or y <= 0:
            return 1e10
        pts = np.array([
            [x, y], [-x, y], [-x, -y], [x, -y],
            [y, x], [-y, x], [-y, -x], [y, -x]
        ])
        return -calc_score(pts)

    # 1. Optimize the symmetric D4 family
    # Initial guess: Regular octagon has x = r*cos(pi/8), y = r*sin(pi/8).
    # Ratio y/x = tan(pi/8) = sqrt(2)-1 ~ 0.414
    x0 = [1.0, 0.5] 
    res_sym = minimize(objective_sym, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
    
    best_score = -np.inf
    best_points = None
    
    if res_sym.success or res_sym.fun < 1e9:
        x_opt, y_opt = res_sym.x
        pts_sym = np.array([
            [x_opt, y_opt], [-x_opt, y_opt], [-x_opt, -y_opt], [x_opt, -y_opt],
            [y_opt, x_opt], [-y_opt, x_opt], [-y_opt, -x_opt], [y_opt, -x_opt]
        ])
        score_sym = calc_score(pts_sym)
        if score_sym > best_score:
            best_score = score_sym
            best_points = pts_sym.flatten()

    # 2. Use the symmetric solution as a seed for full-space optimization
    # This allows breaking symmetry if it helps.
    
    def objective_full(params):
        pts = params.reshape(n, 2)
        return -calc_score(pts)

    def optimize_from_start(x0, method='Powell'):
        try:
            res = minimize(objective_full, x0, method=method, 
                           options={'maxiter': 200, 'maxfev': 5000})
            return res.x
        except Exception:
            return x0

    # Seed 1: The optimized symmetric point
    if best_points is not None:
        res_full = optimize_from_start(best_points)
        score_full = -objective_full(res_full)
        if score_full > best_score:
            best_score = score_full
            best_points = res_full

    # Seed 2: Regular Octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_oct = np.column_stack((np.cos(angles), np.sin(angles)))
    res_oct = optimize_from_start(pts_oct.flatten())
    score_oct = -objective_full(res_oct)
    if score_oct > best_score:
        best_score = score_oct
        best_points = res_oct

    # Seed 3: Random starts (fewer, as we have a strong symmetric seed)
    for i in range(3):
        np.random.seed(100 + i)
        pts_rand = np.random.randn(n, 2)
        # Scale to diameter 2
        d = get_diameter(pts_rand)
        pts_rand = pts_rand * (2.0 / d)
        res_rand = optimize_from_start(pts_rand.flatten())
        score_rand = -objective_full(res_rand)
        if score_rand > best_score:
            best_score = score_rand
            best_points = res_rand

    if best_points is None:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles))).flatten()

    pts_final = best_points.reshape(n, 2)
    d = get_diameter(pts_final)
    if d > 0:
        pts_final = pts_final * (2.0 / d)
        
    return [[float(p[0]), float(p[1])] for p in pts_final]
