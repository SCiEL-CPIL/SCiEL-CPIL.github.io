import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s, safety margin

    best_result = None
    best_score = -np.inf

    def physics_potential(params):
        """Compute repulsive potential energy and its gradient for the physics-based solver.
        
        Uses a 1/r^6 repulsive potential between circles (modeled as hard spheres with
        effective radius = r_i + r_j), plus boundary repulsion. The force is the negative
        gradient of this potential.
        """
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        # Ensure radii are positive
        r = np.maximum(r, 1e-10)
        
        # Compute pairwise distances and forces
        force = np.zeros_like(xy)
        energy = 0.0
        
        # Pairwise repulsion: U = sum_{i<j} k / d_ij^6 where d_ij = |x_i - x_j| - (r_i + r_j)
        # Force on i from j: F_ij = -6*k / d_ij^7 * (x_i - x_j)/|x_i - x_j|
        k_rep = 1.0
        softening = 1e-6
        
        for i in range(n):
            for j in range(i+1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + softening)
                
                # Effective separation (can be negative if overlapping)
                sep = dist - (r[i] + r[j])
                
                if sep < 0.5:  # Only apply strong repulsion when close
                    # Use a smooth potential: U = k / (sep + eps)^6 for sep < threshold
                    eps = 0.01
                    denom = sep + eps
                    if denom > 1e-8:
                        U_pair = k_rep / denom**6
                        # dU/dsep = -6*k / denom^7
                        dU_dsep = -6.0 * k_rep / denom**7
                        # dsep/dx_i = (x_i - x_j)/dist
                        fx = dU_dsep * dx / dist
                        fy = dU_dsep * dy / dist
                        force[i, 0] += fx
                        force[i, 1] += fy
                        force[j, 0] -= fx
                        force[j, 1] -= fy
                        energy += U_pair
        
        # Boundary repulsion: push circles away from edges
        k_bound = 50.0
        for i in range(n):
            # Left wall
            d_left = xy[i, 0] - r[i]
            if d_left < 0.1:
                d = d_left + 0.01
                if d > 1e-8:
                    U_b = k_bound / d**4
                    dU_dd = -4.0 * k_bound / d**5
                    force[i, 0] += dU_dd
                    energy += U_b
            
            # Right wall
            d_right = 1.0 - xy[i, 0] - r[i]
            if d_right < 0.1:
                d = d_right + 0.01
                if d > 1e-8:
                    U_b = k_bound / d**4
                    dU_dd = -4.0 * k_bound / d**5
                    force[i, 0] -= dU_dd  # Force pushes away from right wall
                    energy += U_b
            
            # Bottom wall
            d_bottom = xy[i, 1] - r[i]
            if d_bottom < 0.1:
                d = d_bottom + 0.01
                if d > 1e-8:
                    U_b = k_bound / d**4
                    dU_dd = -4.0 * k_bound / d**5
                    force[i, 1] += dU_dd
                    energy += U_b
            
            # Top wall
            d_top = 1.0 - xy[i, 1] - r[i]
            if d_top < 0.1:
                d = d_top + 0.01
                if d > 1e-8:
                    U_b = k_bound / d**4
                    dU_dd = -4.0 * k_bound / d**5
                    force[i, 1] -= dU_dd
                    energy += U_b
        
        # Gradient: for xy, it's -force; for r, we need dU/dr
        grad_xy = -force.flatten()
        
        # Gradient with respect to radii
        grad_r = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + softening)
                sep = dist - (r[i] + r[j])
                
                if sep < 0.5:
                    eps = 0.01
                    denom = sep + eps
                    if denom > 1e-8:
                        # dU/dr_i = dU/dsep * dsep/dr_i = (-6k/denom^7) * (-1)
                        dU_dr = 6.0 * k_rep / denom**7
                        grad_r[i] += dU_dr
                        grad_r[j] += dU_dr
            
            # Boundary contributions to grad_r
            d_left = xy[i, 0] - r[i]
            if d_left < 0.1:
                d = d_left + 0.01
                if d > 1e-8:
                    grad_r[i] += 4.0 * k_bound / d**5  # dU/dr = -dU/dd * ddd/dr = -(-4k/d^5)*(-1) = -4k/d^5... wait
                    # Actually dU/dr_i where U = k/(d_left+0.01)^4, d_left = x-r
                    # dU/dr = dU/dd * dd/dr = (-4k/d^5) * (-1) = 4k/d^5
                    # But this should be negative (increasing r increases energy when near wall)
                    # Let me reconsider: if r increases, d_left decreases, so U increases
                    # So dU/dr should be positive when near wall
                    pass
            
            d_right = 1.0 - xy[i, 0] - r[i]
            if d_right < 0.1:
                d = d_right + 0.01
                if d > 1e-8:
                    grad_r[i] += 4.0 * k_bound / d**5
            
            d_bottom = xy[i, 1] - r[i]
            if d_bottom < 0.1:
                d = d_bottom + 0.01
                if d > 1e-8:
                    grad_r[i] += 4.0 * k_bound / d**5
            
            d_top = 1.0 - xy[i, 1] - r[i]
            if d_top < 0.1:
                d = d_top + 0.01
                if d > 1e-8:
                    grad_r[i] += 4.0 * k_bound / d**5
        
        return energy, np.concatenate([grad_xy, grad_r])

    def is_valid(result, tol=1e-8):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]

        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False

        return True

    def get_score(result):
        n = N
        r = result[2*n:]
        return np.sum(r)

    def push_apart(params, n_iters=100):
        """Physics-based relaxation using repulsive forces."""
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        # Use a simple iterative force-based approach
        for it in range(n_iters):
            if time.time() > deadline:
                break
                
            force = np.zeros_like(xy)
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist * 1.5:
                        # Repulsive force proportional to overlap
                        overlap = min_dist - dist
                        if overlap > 0:
                            f = 10.0 * overlap / min_dist  # Scaled force
                            force[i, 0] += f * dx / dist
                            force[i, 1] += f * dy / dist
                            force[j, 0] -= f * dx / dist
                            force[j, 1] -= f * dy / dist
            
            # Boundary forces
            for i in range(n):
                if xy[i, 0] < r[i] + 0.05:
                    force[i, 0] += 50.0 * (r[i] + 0.05 - xy[i, 0])
                if xy[i, 0] > 1 - r[i] - 0.05:
                    force[i, 0] -= 50.0 * (xy[i, 0] - (1 - r[i] - 0.05))
                if xy[i, 1] < r[i] + 0.05:
                    force[i, 1] += 50.0 * (r[i] + 0.05 - xy[i, 1])
                if xy[i, 1] > 1 - r[i] - 0.05:
                    force[i, 1] -= 50.0 * (xy[i, 1] - (1 - r[i] - 0.05))
            
            # Damped update
            lr = 0.02
            xy += lr * force
            
            # Clamp to valid region
            for i in range(n):
                xy[i, 0] = np.clip(xy[i, 0], r[i], 1 - r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], 1 - r[i])
        
        return np.concatenate([xy.flatten(), r])

    def refine_solution(params, max_iter=500, tol=1e-14):
        """Refine using L-BFGS-B on a combined objective that maximizes sum of radii
        while penalizing violations."""
        n = N
        
        def objective(p):
            xy = p[:2*n].reshape(n, 2)
            r = p[2*n:]
            
            # Penalty for violations
            pen = 0.0
            for i in range(n):
                if xy[i, 0] < r[i]:
                    pen += 1000 * (r[i] - xy[i, 0])**2
                if xy[i, 0] > 1 - r[i]:
                    pen += 1000 * (xy[i, 0] - (1 - r[i]))**2
                if xy[i, 1] < r[i]:
                    pen += 1000 * (r[i] - xy[i, 1])**2
                if xy[i, 1] > 1 - r[i]:
                    pen += 1000 * (xy[i, 1] - (1 - r[i]))**2
            
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_d = r[i] + r[j]
                    if dist < min_d:
                        pen += 1000 * (min_d - dist)**2
            
            return -np.sum(r) + pen
        
        try:
            result = minimize(objective, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12})
            return result.x
        except Exception:
            return params

    def get_initial_config(restart, rng):
        n = N
        side = int(np.ceil(np.sqrt(N)))

        if restart % 10 == 0:
            # Grid packing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart % 10 == 1:
            # Perturbed grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(n, 2))
        elif restart % 10 == 2:
            # Hexagonal packing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        elif restart % 10 == 3:
            # Random uniform
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        elif restart % 10 == 4:
            # 8x4 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 5:
            # 4x8 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 6:
            # Hexagonal variant
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.2 if row % 2 == 0 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.075
        elif restart % 10 == 7:
            # Clustered: 4 groups of 8
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub_i = i % 8
                sx = (sub_i % 3) / 3.0 + 0.5/3
                sy = (sub_i // 3) / 3.0 + 0.5/3
                xy[i] = [cx + (sx - 0.5)*0.4, cy + (sy - 0.5)*0.4]
                r[i] = 0.06
        elif restart % 10 == 8:
            # Random medium
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.06, 0.09, size=n)
        else:
            # Random small
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.03, 0.06, size=n)

        params = np.concatenate([xy.flatten(), r])
        return params

    def voronoi_based_packing(rng):
        """
        Use a Voronoi-like approach: start with a grid, then iteratively
        adjust centers and radii based on local Voronoi cell inradii.
        This ensures natural separation.
        """
        n = N
        # Start with a 6x6 grid (36 cells, use 32)
        side = 6
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        
        for i in range(n):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
            r[i] = 0.08
        
        # Add some randomness
        xy += rng.uniform(-0.03, 0.03, size=(n, 2))
        r += rng.uniform(-0.01, 0.01, size=n)
        r = np.clip(r, 0.03, 0.15)
        
        params = np.concatenate([xy.flatten(), r])
        params = push_apart(params, n_iters=80)
        
        # Refine
        refined = refine_solution(params, max_iter=400, tol=1e-14)
        
        if is_valid(refined):
            return refined
        else:
            repaired = push_apart(refined, n_iters=100)
            if is_valid(repaired):
                return repaired
            else:
                return params if is_valid(params) else None

    rng = np.random.RandomState(12345)

    # Phase 1: Many restarts with diverse initializations
    n_restarts = 60
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break

        # Use Voronoi-based approach for some restarts
        if restart % 5 == 0:
            params = voronoi_based_packing(rng)
            if params is None:
                continue
        else:
            params = get_initial_config(restart, rng)
            params = push_apart(params, n_iters=60)

        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

        # Refine with L-BFGS-B
        refined = refine_solution(params, max_iter=250, tol=1e-13)

        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            # Try to repair
            repaired = push_apart(refined, n_iters=80)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                # Shrink radii until valid
                xy = refined[:2*N].reshape(N, 2).copy()
                r = refined[2*N:].copy()
                for _ in range(200):
                    if is_valid(np.concatenate([xy.flatten(), r])):
                        break
                    r *= 0.95
                final_params = np.concatenate([xy.flatten(), r])
                if is_valid(final_params):
                    score = get_score(final_params)
                    if score > best_score:
                        best_score = score
                        best_result = final_params.copy()

    # Phase 2: High-precision polish on best solution
    if best_result is not None and time.time() < deadline - 5:
        polished = refine_solution(best_result, max_iter=600, tol=1e-15)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Try perturbations and re-refine
        for polish_iter in range(8):
            if time.time() > deadline - 2:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            xy_p += rng.uniform(-0.003, 0.003, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=40)
            polished2 = refine_solution(perturbed, max_iter=300, tol=1e-15)
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
