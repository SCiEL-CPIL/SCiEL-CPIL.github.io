import numpy as np
from scipy.optimize import minimize, basinhopping
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    n = 8
    num_pairs = n * (n - 1) // 2  # 28

    # Fix P0 at (0,0) and P4 at (2,0)
    # Free indices: 1, 2, 3, 5, 6, 7
    fixed_idx = [0, 4]
    free_idx = [i for i in range(n) if i not in fixed_idx]
    n_free = len(free_idx)  # 6
    n_dims = n_free * 2

    def unpack(x):
        p = np.zeros((n, 2))
        p[0] = [0.0, 0.0]
        p[4] = [2.0, 0.0]
        for i, idx in enumerate(free_idx):
            p[idx] = [x[2 * i], x[2 * i + 1]]
        return p

    def objective(x, beta=50.0):
        """
        Minimize: -sum(log d_ij) + num_pairs * log(soft_max(d_ij, beta))
        The soft_max term acts as a smooth approximation to the maximum distance.
        Minimizing this encourages large pairwise distances (maximizing product)
        while penalizing any distance that exceeds the others significantly,
        effectively constraining the diameter.
        """
        p = unpack(x)
        # Compute all pairwise distances
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(np.maximum(dists_sq[iu], 1e-15))

        sum_log_d = np.sum(np.log(d))

        # Soft-max via log-sum-exp for numerical stability
        max_d = np.max(d)
        scaled = beta * (d - max_d)
        scaled = np.clip(scaled, -500, 500)
        log_sum = np.log(np.sum(np.exp(scaled)))
        soft_max = max_d + log_sum / beta

        val = -sum_log_d + num_pairs * np.log(soft_max)
        return val

    # Initial guess: Regular octagon
    # Generate regular octagon centered at (1,0) with radius 1
    # Vertices: (1 + cos(2*pi*k/8), sin(2*pi*k/8))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_unit = np.column_stack((np.cos(angles), np.sin(angles)))
    init_pts = pts_unit + np.array([1.0, 0.0])
    
    # Check which points are at (0,0) and (2,0)
    # k=4: (1-1, 0) = (0,0) -> Index 4
    # k=0: (1+1, 0) = (2,0) -> Index 0
    # So init_pts[0] is (2,0) and init_pts[4] is (0,0).
    # We fixed P0=(0,0) and P4=(2,0).
    # So we need to swap the roles or just extract the free coordinates from this configuration.
    # Let's create the vector x0 corresponding to this configuration in our fixed frame.
    
    x0 = np.zeros(n_dims)
    for i, idx in enumerate(free_idx):
        # We need to map the regular octagon points to our fixed frame.
        # In the regular octagon, the point at (0,0) is index 4.
        # The point at (2,0) is index 0.
        # Our fixed frame has P0=(0,0) and P4=(2,0).
        # So the point at regular_octagon[4] corresponds to our P0.
        # The point at regular_octagon[0] corresponds to our P4.
        # The other points are just shifted/rotated? No, the regular octagon is symmetric.
        # We just need the coordinates of the other 6 points relative to the frame where P0=(0,0) and P4=(2,0).
        # In the standard position, P0 (our fixed) is at (0,0).
        # The regular octagon point that ends up at (0,0) is the one at angle pi (k=4).
        # So we can just take the regular octagon points, subtract the position of k=4, 
        # and ensure k=0 ends up at (2,0).
        
        # Let's just use the standard regular octagon coordinates directly.
        # We defined init_pts as centered at (1,0).
        # init_pts[4] = (0,0). init_pts[0] = (2,0).
        # So if we set P0=init_pts[4] and P4=init_pts[0], the other points are just init_pts[k] - init_pts[4].
        # Since init_pts[4] is (0,0), the coordinates are just init_pts[k].
        
        x0[2*i] = init_pts[idx][0]
        x0[2*i+1] = init_pts[idx][1]

    best_x = x0.copy()
    best_obj = objective(x0)

    # Phase 1: Local optimization from regular octagon
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x.copy()
    except Exception:
        pass

    # Phase 2: Basin Hopping
    np.random.seed(42)
    
    def step(x):
        return x + np.random.normal(0, 0.5, size=x.shape)

    try:
        x_start_bh = best_x.copy()
        res_bh = basinhopping(objective, x_start_bh, 
                              minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7}},
                              stepfunc=step, T=1000.0, stepsize=0.5, niter=50, interval=10, disp=False)
        
        if res_bh.fun < best_obj:
            best_obj = res_bh.fun
            best_x = res_bh.x.copy()
            
    except Exception:
        pass

    # Phase 3: Multi-restart if time permits
    if time.time() < deadline - 20:
        restarts = []
        for scale in [0.1, 0.3, 0.5]:
            for trial in range(5):
                noise = np.random.normal(0, scale, size=n_dims)
                restarts.append(x0 + noise)
        
        for x0_r in restarts:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective, x0_r, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x.copy()
            except Exception:
                continue

    # Phase 4: High-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False})
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
