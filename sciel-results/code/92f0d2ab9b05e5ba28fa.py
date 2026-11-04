import numpy as np
import time
from scipy.optimize import minimize, least_squares

N = 300

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def remaining_time():
        return deadline - (time.time() - start_time)

    def is_finished():
        return (time.time() - start_time) > deadline

    def _compute_radius(p):
        """Compute the minimum radius for a given set of points."""
        p = np.asarray(p)
        if p.shape != (N, 2):
            return 0.0
        norms = np.linalg.norm(p, axis=1)
        min_bound = 1.0 - np.min(norms)
        if min_bound <= 0:
            return 0.0
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        return min(min_pair, min_bound)

    def _init_hex():
        """Initialize with hexagonal lattice."""
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points, radius = points[mask], radius[mask]
            if len(points) >= N:
                points = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
        return points

    def _hertzian_dynamics(x0, n_iter=1500, step=0.05):
        """Simple Hertzian dynamics to spread points."""
        p = x0.copy()
        for i in range(n_iter):
            if is_finished():
                break
            diff = p[:, np.newaxis, :] - p[np.newaxis, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            inv_dists = 1.0 / np.maximum(dists, 1e-6) ** 2
            np.fill_diagonal(inv_dists, 0.0)
            forces = np.einsum('ij,ijk->ik', inv_dists, diff)
            forces -= 0.1 * p
            p += step * forces
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            p = p * np.minimum(1.0, 0.99 / norms)
        return p

    def _soft_min_optimize(x0, beta=1000, max_iter=500):
        """Optimize using soft-min formulation with L-BFGS-B."""
        def get_points(x):
            return x.reshape(N, 2)

        def objective_and_grad(x):
            points = get_points(x)
            sq_norms = np.sum(points * points, axis=1)
            dot_prod = points @ points.T
            sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * dot_prod
            sq_dists = np.maximum(sq_dists, 0.0)
            dists = np.sqrt(sq_dists)
            iu = np.triu_indices(N, k=1)
            d_ij = dists[iu]
            norms = np.sqrt(sq_norms)
            d_b = 1.0 - norms
            v = np.concatenate([d_ij / 2.0, d_b])
            v_min = np.min(v)
            exp_v = np.exp(-beta * (v - v_min))
            S = np.sum(exp_v)
            weights = exp_v / S
            softmin_val = v_min - (1.0 / beta) * np.log(S)
            w_pairs = weights[:len(d_ij)]
            w_bound = weights[len(d_ij):]
            w_mat = np.zeros((N, N))
            w_mat[iu] = w_pairs
            w_sym = w_mat + w_mat.T
            d_safe = np.where(dists < 1e-9, 1e-9, dists)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :]
            factor = w_sym / (2.0 * d_safe)
            np.fill_diagonal(factor, 0.0)
            grad_points = np.einsum('ij,ijk->ik', factor, diff)
            norms_safe = np.where(norms < 1e-9, 1e-9, norms)
            grad_bound_points = -(w_bound[:, np.newaxis] * points) / norms_safe[:, np.newaxis]
            total_grad = (grad_points + grad_bound_points).flatten()
            obj_val = -softmin_val
            grad_val = -total_grad
            return obj_val, grad_val

        try:
            res = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': max_iter * 2}
            )
            return res.x
        except Exception:
            return x0

    def _slsqp_refine(x0, r_target, max_iter=200):
        """
        Refine using SLSQP with auxiliary variables for squared distances.
        Formulation:
        Variables: x (positions), t (target squared radius)
        Objective: minimize -t
        Constraints:
          1. ||x_i - x_j||^2 >= 2*t  for all i<j  (pair constraints)
          2. 1 - ||x_i|| >= sqrt(t)   for all i   (boundary constraints)
        
        We reformulate as:
          Minimize -t
          Subject to:
            ||x_i - x_j||^2 - 2*t >= 0  for all i<j
            1 - ||x_i|| - sqrt(t) >= 0  for all i
        """
        n_pairs = N * (N - 1) // 2
        n_vars = 2 * N + 1  # 2N positions + 1 for t
        
        def unpack(x):
            positions = x[:2*N].reshape(N, 2)
            t = x[2*N]
            return positions, t

        def objective(x):
            _, t = unpack(x)
            return -t

        def objective_grad(x):
            return np.concatenate([np.zeros(2*N), [-1.0]])

        def constraints(x):
            positions, t = unpack(x)
            cons = []
            
            # Pair constraints: ||x_i - x_j||^2 - 2*t >= 0
            # We'll compute these in batches to avoid memory issues
            iu = np.triu_indices(N, k=1)
            diff = positions[iu[0]] - positions[iu[1]]
            sq_dists = np.sum(diff * diff, axis=1)
            pair_cons = sq_dists - 2.0 * t
            cons.append(pair_cons)
            
            # Boundary constraints: 1 - ||x_i|| - sqrt(t) >= 0
            norms = np.linalg.norm(positions, axis=1)
            bound_cons = 1.0 - norms - np.sqrt(max(t, 0.0))
            cons.append(bound_cons)
            
            return np.concatenate(cons)

        def constraints_grad(x):
            positions, t = unpack(x)
            grads = np.zeros((len(constraints(x)), n_vars))
            
            # Pair constraint gradients
            iu = np.triu_indices(N, k=1)
            diff = positions[iu[0]] - positions[iu[1]]
            n_p = len(iu[0])
            
            for k in range(n_p):
                i, j = iu[0][k], iu[1][k]
                # d(||xi-xj||^2 - 2t)/dxi = 2(xi-xj)
                grads[k, 2*i:2*i+2] = 2.0 * diff[k]
                # d(||xi-xj||^2 - 2t)/dxj = -2(xi-xj)
                grads[k, 2*j:2*j+2] = -2.0 * diff[k]
                # d(||xi-xj||^2 - 2t)/dt = -2
                grads[k, 2*N] = -2.0
            
            # Boundary constraint gradients
            n_b = N
            offset = n_p
            norms = np.linalg.norm(positions, axis=1)
            norms_safe = np.where(norms < 1e-9, 1e-9, norms)
            sqrt_t = np.sqrt(max(t, 0.0))
            
            for k in range(n_b):
                i = k
                # d(1 - ||xi|| - sqrt(t))/dxi = -xi/||xi||
                grads[offset+k, 2*i:2*i+2] = -positions[i] / norms_safe[i]
                # d(1 - ||xi|| - sqrt(t))/dt = -1/(2*sqrt(t))
                if t > 1e-12:
                    grads[offset+k, 2*N] = -1.0 / (2.0 * np.sqrt(t))
            
            return grads

        # Initial guess
        positions, t_init = unpack(x0)
        # Estimate a good initial t
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        norms = np.linalg.norm(positions, axis=1)
        min_bound = 1.0 - np.min(norms)
        r_est = min(min_pair, min_bound) * 0.99
        t_est = r_est * r_est
        
        x_init = np.concatenate([positions.flatten(), [t_est]])
        
        # Bounds
        lb = np.concatenate([[-1.0]*N, [-1.0]*N, [1e-8]])
        ub = np.concatenate([[1.0]*N, [1.0]*N, [1.0]])
        
        # Constraints for SLSQP
        cons_slsqp = [
            {'type': 'ineq', 'fun': lambda x: constraints(x), 'jac': lambda x: constraints_grad(x)}
        ]
        
        try:
            res = minimize(
                objective,
                x_init,
                jac=objective_grad,
                method='SLSQP',
                bounds=list(zip(lb, ub)),
                constraints=cons_slsqp,
                options={'maxiter': max_iter, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < objective(x_init):
                new_positions = res.x[:2*N].reshape(N, 2)
                # Project back inside
                rad_new = np.linalg.norm(new_positions, axis=1)
                mask = rad_new > 1.0
                if np.any(mask):
                    new_positions[mask] /= rad_new[mask, np.newaxis]
                return new_positions
            else:
                return x0
        except Exception:
            return x0

    # Main optimization loop
    best_x = None
    best_score = 0.0

    # Step 1: Hexagonal initialization
    hex_points = _init_hex()
    score = _compute_radius(hex_points)
    if score > best_score:
        best_score = score
        best_x = hex_points.copy()

    if is_finished():
        return best_x

    # Step 2: Hertzian dynamics
    if remaining_time() > 30:
        hert_points = _hertzian_dynamics(hex_points, n_iter=1000, step=0.05)
        score = _compute_radius(hert_points)
        if score > best_score:
            best_score = score
            best_x = hert_points.copy()

    if is_finished():
        return best_x

    # Step 3: Soft-min optimization with annealing
    betas = [10, 50, 200, 1000, 5000]
    current_x = best_x.flatten()

    for beta in betas:
        if is_finished() or remaining_time() < 15:
            break
        max_iter = min(500, int(remaining_time() * 3))
        if max_iter < 50:
            break
        new_x = _soft_min_optimize(current_x, beta=beta, max_iter=max_iter)
        score = _compute_radius(new_x.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_x = new_x.reshape(N, 2).copy()
            current_x = new_x
        elif is_finished():
            break

    if is_finished():
        return best_x

    # Step 4: SLSQP refinement with auxiliary variables
    if remaining_time() > 20:
        r_target = best_score * 0.999
        refined = _slsqp_refine(best_x.flatten(), r_target, max_iter=100)
        score = _compute_radius(refined)
        if score > best_score:
            best_score = score
            best_x = refined.copy()

    if is_finished():
        return best_x

    # Step 5: Random restarts if time remains
    n_restarts = 3
    for i in range(n_restarts):
        if is_finished() or remaining_time() < 20:
            break
        perturbed = best_x + np.random.randn(*best_x.shape) * 0.05
        new_x = _soft_min_optimize(perturbed.flatten(), beta=1000, max_iter=200)
        score = _compute_radius(new_x.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_x = new_x.reshape(N, 2).copy()

    if is_finished():
        return best_x

    # Step 6: Final polish with high beta
    if remaining_time() > 10:
        polished = _soft_min_optimize(best_x.flatten(), beta=50000, max_iter=200)
        score = _compute_radius(polished.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_x = polished.reshape(N, 2).copy()

    return best_x.reshape(N, 2)
