import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    n = 8
    num_pairs = n * (n - 1) // 2  # 28

    # Fix P0=(0,0) and P4=(2,0) to remove translation/rotation/scale invariance
    # The remaining 6 points are free
    fixed_idx = [0, 4]
    free_idx = [i for i in range(n) if i not in fixed_idx]
    n_free = len(free_idx)  # 6
    n_dims = n_free * 2  # 12

    def unpack(x):
        p = np.zeros((n, 2))
        p[0] = [0.0, 0.0]
        p[4] = [2.0, 0.0]
        for i, idx in enumerate(free_idx):
            p[idx] = [x[2 * i], x[2 * i + 1]]
        return p

    def pairwise_dists(p):
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(np.maximum(dists_sq[iu], 1e-30))
        return d

    def objective(x, beta=50.0):
        """
        Minimize: -sum(log d_ij) + num_pairs * log(softmax(d_ij, beta))
        The soft-max term penalizes configurations where the maximum distance
        exceeds 2 (since we want diameter <= 2 after scaling).
        Actually, since we fix P0=(0,0) and P4=(2,0), the diameter is at least 2.
        We want to maximize the product of all pairwise distances subject to
        all distances <= 2. So we want to maximize sum(log d_ij) subject to
        max(d_ij) <= 2.
        
        The objective minimizes -sum(log d_ij) + penalty for d > 2.
        """
        p = unpack(x)
        d = pairwise_dists(p)
        
        sum_log_d = np.sum(np.log(d))
        
        # Penalty for distances > 2: use smooth max
        max_d = np.max(d)
        if max_d > 2.0:
            # Smooth penalty: use log-sum-exp to approximate max, then penalize
            scaled = beta * (d - 2.0)
            scaled = np.clip(scaled, -500, 500)
            log_sum = np.log(np.sum(np.exp(scaled)))
            soft_max_above_2 = 2.0 + log_sum / beta
            penalty = num_pairs * np.log(soft_max_above_2)
        else:
            penalty = 0.0
        
        return -sum_log_d + penalty

    # Regular octagon as initial guess
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_unit = np.column_stack((np.cos(angles), np.sin(angles)))
    # Center at (1,0), scale so that opposite vertices are at distance 2
    # Regular octagon with vertices at angles k*pi/4 has circumradius R
    # Distance between opposite vertices = 2R
    # We want 2R = 2, so R = 1
    # But we need P0=(0,0) and P4=(2,0). In a regular octagon centered at (1,0),
    # vertex 0 is at (1+cos(0), sin(0)) = (2, 0) and vertex 4 is at (1+cos(pi), sin(pi)) = (0, 0)
    # So P0 should be at angle pi and P4 at angle 0, or we can just set them.
    # Let's construct the regular octagon properly:
    # Vertices at angles (k + 0.5) * pi/4 for k=0..7? No.
    # Let's just use the fact that in a regular octagon inscribed in a circle of radius 1 centered at (1,0),
    # the vertices are at (1 + cos(theta_k), sin(theta_k)) where theta_k = k*pi/4.
    # Vertex 0: (2, 0), Vertex 4: (0, 0). 
    # So we want P0=(0,0) which is vertex 4, and P4=(2,0) which is vertex 0.
    # So the mapping is: our index 0 -> octagon vertex 4, our index 4 -> octagon vertex 0.
    # The other points: our index i -> octagon vertex (i - 4) mod 8? Let's just compute all and fix.
    
    theta = np.arange(n) * np.pi / 4
    reg_pts = np.column_stack((1.0 + np.cos(theta), np.sin(theta)))
    # reg_pts[0] = (2,0), reg_pts[4] = (0,0)
    # We want p[0] = (0,0) = reg_pts[4], p[4] = (2,0) = reg_pts[0]
    # So we need to permute: our point i should be reg_pts[(i + 4) % 8]
    perm = [(i + 4) % 8 for i in range(n)]
    reg_pts_perm = reg_pts[perm]
    # Now reg_pts_perm[0] = reg_pts[4] = (0,0), reg_pts_perm[4] = reg_pts[0] = (2,0). Good.
    
    x0 = np.array([reg_pts_perm[i] for i in free_idx]).flatten()
    
    best_x = x0.copy()
    best_obj = objective(x0)

    # Phase 1: BFGS from regular octagon
    try:
        res = minimize(objective, x0, method='BFGS',
                       options={'maxiter': 300, 'gtol': 1e-8, 'disp': False})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x.copy()
    except Exception:
        pass

    # Phase 2: Differential Evolution for global search
    if time.time() < deadline - 60:
        bounds = [(-2.0, 4.0)] * n_dims  # Generous bounds
        try:
            de_result = differential_evolution(
                objective, bounds, seed=42, maxiter=100, tol=1e-10,
                popsize=15, mutation=(0.3, 0.8), recombination=0.9,
                polish=True, init='latinhypercube'
            )
            if de_result.fun < best_obj:
                best_obj = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass

    # Phase 3: Multi-restart BFGS from diverse starting points
    if time.time() < deadline - 30:
        np.random.seed(123)
        restarts = []
        
        # Perturb regular octagon
        for scale in [0.1, 0.3, 0.5, 0.8]:
            for trial in range(4):
                noise = np.random.normal(0, scale, size=(n, 2))
                noise[0] = 0
                noise[4] = 0
                perturbed = reg_pts_perm + noise
                x0_pert = np.array([perturbed[i] for i in free_idx]).flatten()
                restarts.append(x0_pert)
        
        # Random starts
        for trial in range(15):
            x0_rand = np.random.uniform(-1, 3, size=n_dims)
            restarts.append(x0_rand)

        for x0_r in restarts:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(objective, x0_r, method='BFGS',
                               options={'maxiter': 200, 'gtol': 1e-7, 'disp': False})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x.copy()
            except Exception:
                continue

    # Phase 4: Basin hopping from best
    if time.time() < deadline - 20:
        def step(x):
            return x + np.random.normal(0, 0.4, size=x.shape)
        
        np.random.seed(42)
        try:
            from scipy.optimize import basinhopping
            res_bh = basinhopping(
                objective, best_x.copy(),
                minimizer_kwargs={'method': 'BFGS', 'options': {'maxiter': 150, 'gtol': 1e-7}},
                stepfunc=step, T=500.0, stepsize=0.4, niter=40, interval=10, disp=False
            )
            if res_bh.fun < best_obj:
                best_obj = res_bh.fun
                best_x = res_bh.x.copy()
        except Exception:
            pass

    # Phase 5: High-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B',
                           options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-15, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(objective, best_x, method='BFGS',
                           options={'maxiter': 3000, 'gtol': 1e-14, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    p = unpack(best_x)

    # Ensure distinct points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-8:
                p[j] += np.array([1e-6, 0])

    # Scale to ensure diameter is exactly 2
    max_d = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(p[i] - p[j])
            if d > max_d:
                max_d = d

    if max_d > 0 and abs(max_d - 2.0) > 1e-10:
        scale_factor = 2.0 / max_d
        p = p * scale_factor

    return [[float(p[i][0]), float(p[i][1])] for i in range(n)]
