import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 10.0
    
    best_points = None
    best_score = -np.inf

    def compute_score(points):
        """Vectorized score computation."""
        if not np.all(np.isfinite(points)):
            return -np.inf
        if points.shape != (N, 2):
            return -np.inf
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - norms)
        
        return min(min_pair, min_boundary)

    def soft_min_obj_and_grad(x_flat, beta):
        """Soft-min objective and gradient (vectorized)."""
        points = x_flat.reshape(N, 2)
        
        # Pairwise constraints
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-24)
        np.fill_diagonal(dists, 1.0)  # Avoid self-constraint issues
        
        v_pair = dists / 2.0
        exp_pair = np.exp(-beta * v_pair)
        # Mask out diagonal
        mask = ~np.eye(N, dtype=bool)
        exp_pair[~mask] = 0.0
        
        sum_exp_pair = np.sum(exp_pair)
        
        # Boundary constraints
        norms = np.linalg.norm(points, axis=1)
        norms_safe = np.maximum(norms, 1e-12)
        v_bound = 1.0 - norms_safe
        exp_bound = np.exp(-beta * v_bound)
        sum_exp_bound = np.sum(exp_bound)
        
        total_sum = sum_exp_pair + sum_exp_bound
        
        if total_sum < 1e-300:
            total_sum = 1e-300
        
        obj = (1.0 / beta) * np.log(total_sum)
        
        # Gradient
        grad = np.zeros_like(x_flat)
        
        # Pairwise gradient contribution
        # d/dx_i of sum_{j!=i} exp(-beta * d_ij/2)
        # = sum_{j!=i} exp(-beta * d_ij/2) * (-beta/2) * (x_i - x_j)/d_ij
        grad_pair = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-12:
                    continue
                coeff = exp_pair[i, j] * (-beta / 2.0) / d
                grad_pair[i] += coeff * (points[i] - points[j])
        
        # Boundary gradient contribution
        # d/dx_i of exp(-beta * (1 - |x_i|)) = exp(...) * (-beta) * (-x_i/|x_i|)
        grad_bound = np.zeros_like(points)
        for i in range(N):
            norm_i = norms_safe[i]
            if norm_i < 1e-12:
                continue
            coeff = exp_bound[i] * (-beta)
            grad_bound[i] += coeff * (-points[i] / norm_i)
        
        total_grad = grad_pair + grad_bound
        grad = -(total_grad / total_sum).flatten()
        
        return obj, grad

    def optimize_from_init(points_init, time_budget):
        """Optimize from a given initial configuration."""
        x0 = points_init.flatten()
        best_local = points_init.copy()
        best_local_score = compute_score(points_init)
        
        start = time.time()
        
        # Phase 1: Global exploration with low beta
        betas_phase1 = [5.0, 10.0, 20.0, 40.0]
        for beta in betas_phase1:
            if time.time() - start > time_budget * 0.4:
                break
            try:
                res = minimize(
                    lambda x: soft_min_obj_and_grad(x, beta)[0],
                    x0,
                    jac=lambda x: soft_min_obj_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                x_opt = res.x
                points_opt = x_opt.reshape(N, 2)
                score = compute_score(points_opt)
                if score > best_local_score:
                    best_local_score = score
                    best_local = points_opt.copy()
                    x0 = x_opt
            except Exception:
                continue
        
        # Phase 2: Local refinement with high beta
        betas_phase2 = [50.0, 100.0, 200.0, 500.0]
        for beta in betas_phase2:
            if time.time() - start > time_budget * 0.8:
                break
            try:
                res = minimize(
                    lambda x: soft_min_obj_and_grad(x, beta)[0],
                    x0,
                    jac=lambda x: soft_min_obj_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x_opt = res.x
                points_opt = x_opt.reshape(N, 2)
                score = compute_score(points_opt)
                if score > best_local_score:
                    best_local_score = score
                    best_local = points_opt.copy()
                    x0 = x_opt
            except Exception:
                continue
        
        # Phase 3: Final polish with very high beta and tight tolerances
        if time.time() - start < time_budget:
            beta = 1000.0
            try:
                res = minimize(
                    lambda x: soft_min_obj_and_grad(x, beta)[0],
                    x0,
                    jac=lambda x: soft_min_obj_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                x_opt = res.x
                points_opt = x_opt.reshape(N, 2)
                score = compute_score(points_opt)
                if score > best_local_score:
                    best_local_score = score
                    best_local = points_opt.copy()
            except Exception:
                pass
        
        return best_local, best_local_score

    # Generate diverse initial configurations
    inits = []
    
    # 1. Optimized hexagonal lattice for 50 circles
    # Try different spacings and take the best valid one
    best_hex = None
    best_hex_score = -np.inf
    for spacing in np.linspace(0.18, 0.28, 20):
        pts = []
        for i in range(-6, 7):
            for j in range(-6, 7):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - spacing / 2:
                    pts.append([x, y])
        if len(pts) >= N:
            pts = np.array(pts)
            # Sort by distance from center, take closest N
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii)[:N]
            candidate = pts[idx]
            score = compute_score(candidate)
            if score > best_hex_score:
                best_hex_score = score
                best_hex = candidate
    if best_hex is not None:
        inits.append(best_hex)
    
    # 2. Ring-based initialization (mimicking known good packings)
    # Center + rings with counts approximating optimal
    ring_configs = [
        [0, 6, 12, 18, 14],  # 0+6+12+18+14 = 50
        [0, 6, 12, 16, 16],  # 0+6+12+16+16 = 50
        [0, 6, 10, 14, 20],  # 0+6+10+14+20 = 50
    ]
    for counts in ring_configs:
        pts = []
        if counts[0] > 0:
            pts.append([0.0, 0.0])
        ring_radii = [0.25, 0.45, 0.65, 0.85]
        for r, c in zip(ring_radii, counts[1:]):
            for i in range(c):
                theta = 2 * np.pi * i / c
                pts.append([r * np.cos(theta), r * np.sin(theta)])
        if len(pts) == N:
            inits.append(np.array(pts))
    
    # 3. Random initializations with different seeds
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        pts = []
        for _ in range(N):
            r = np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            pts.append([r * np.cos(theta), r * np.sin(theta)])
        inits.append(np.array(pts))
    
    # 4. Perturbed hex
    if best_hex is not None:
        np.random.seed(999)
        perturbed = best_hex + np.random.normal(0, 0.02, size=(N, 2))
        # Ensure inside unit circle
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 1.0 - 0.01
        perturbed[mask] *= (1.0 - 0.01) / norms[mask, np.newaxis]
        inits.append(perturbed)
    
    # Optimize each initial configuration
    total_time = time_limit - safety_margin
    n_inits = len(inits)
    
    # Allocate time: give more time to better initial configs
    # First pass: quick evaluation of all inits
    init_scores = []
    for init in inits:
        init_scores.append(compute_score(init))
    
    # Sort by initial score (descending)
    sorted_indices = np.argsort(-np.array(init_scores))
    
    # Allocate time proportional to rank (better inits get more time)
    weights = np.arange(n_inits, 0, -1)
    weights /= np.sum(weights)
    time_alloc = weights * total_time * 0.9  # 90% for optimization, 10% reserve
    
    for rank, idx in enumerate(sorted_indices):
        elapsed = time.time() - start_time
        if elapsed > total_time:
            break
        
        time_left = total_time - elapsed
        t_budget = min(time_alloc[rank], time_left)
        
        if t_budget < 1.0:  # Not enough time for meaningful optimization
            break
        
        init = inits[idx]
        res_points, res_score = optimize_from_init(init, t_budget)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # If we still have time, do a final polish on the best result
    if best_points is not None and time.time() - start_time < total_time - 5:
        remaining = total_time - (time.time() - start_time) - 5
        if remaining > 2:
            polished, polished_score = optimize_from_init(best_points, remaining)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
    
    if best_points is None:
        # Fallback
        if inits:
            best_points = inits[0]
        else:
            # Simple grid fallback
            pts = []
            spacing = 0.2
            for i in range(-5, 6):
                for j in range(-5, 6):
                    x = i * spacing
                    y = j * spacing
                    if np.hypot(x, y) < 0.95:
                        pts.append([x, y])
            if len(pts) >= N:
                best_points = np.array(pts[:N])
            else:
                best_points = np.zeros((N, 2))
                for i in range(N):
                    angle = 2 * np.pi * i / N
                    r = 0.5
                    best_points[i] = [r * np.cos(angle), r * np.sin(angle)]
    
    return best_points
