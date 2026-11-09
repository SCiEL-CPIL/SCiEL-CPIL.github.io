import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def calc_score(x):
        """Compute the packing radius for given points."""
        if len(x) != N:
            return 0.0
        if not np.all(np.isfinite(x)):
            return 0.0
        norms = np.linalg.norm(x, axis=1)
        if np.any(norms > 1.0):
            return 0.0
        b_min = np.min(1.0 - norms)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        return min(min_pair, b_min)

    def hexagonal_init(spacing):
        """Generate points on a hexagonal lattice clipped to unit circle, pick N closest to origin."""
        points = []
        k = int(1.0 / spacing) + 3
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]

    def ring_init(n_rings, r_start, r_end):
        """Initialize with concentric rings."""
        pts = [[0.0, 0.0]]
        for ring in range(1, n_rings + 1):
            r = r_start + (r_end - r_start) * (ring - 1) / max(n_rings - 1, 1)
            if ring == 1:
                n_pts = 6
            else:
                n_pts = ring * 6
            n_pts = min(n_pts, N - len(pts))
            if n_pts <= 0:
                break
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        pts = np.array(pts[:N])
        if len(pts) < N:
            return None
        return pts

    # Generate diverse initializations
    candidates = []

    # Hexagonal grids with various spacings
    for s in np.linspace(0.10, 0.25, 25):
        pts = hexagonal_init(s)
        if pts is not None:
            candidates.append(pts)

    # Ring-based inits
    for n_rings in [5, 6, 7, 8, 9]:
        for r_start in [0.05, 0.1, 0.15, 0.2]:
            pts = ring_init(n_rings, r_start, 0.9)
            if pts is not None:
                candidates.append(pts)

    # Random inits
    for _ in range(15):
        pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, None] * 0.95
        candidates.append(pts)

    # Evaluate all candidates and pick the best few
    scored_candidates = []
    for c in candidates:
        r = calc_score(c)
        scored_candidates.append((r, c))
    scored_candidates.sort(key=lambda x: -x[0])

    # Take top candidates for optimization
    top_candidates = [c for r, c in scored_candidates[:6]]
    if not top_candidates:
        top_candidates = [np.random.rand(N, 2) * 0.8]

    best_x = top_candidates[0].copy()
    best_score = calc_score(best_x)

    # Optimization using soft-min formulation with L-BFGS-B
    def make_obj_func(beta_val):
        """Create objective function with given beta for soft-min."""
        def obj_func(flat_x):
            x = flat_x.reshape(N, 2)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            dist_boundary = 1.0 - np.linalg.norm(x, axis=1)

            iu = np.triu_indices(N, k=1)
            d_pairs = d[iu]
            d_bnds = dist_boundary
            all_d = np.concatenate([d_pairs, d_bnds])

            exponents = -beta_val * all_d
            max_exp = np.max(exponents)

            if not np.isfinite(max_exp):
                return 100.0, np.zeros_like(flat_x)

            sum_exp = np.sum(np.exp(exponents - max_exp))
            log_sum = max_exp + np.log(sum_exp)
            obj = log_sum / beta_val

            weights = np.exp(exponents - max_exp) / sum_exp
            grad = np.zeros_like(x)

            n_pairs = len(d_pairs)
            w_pairs = weights[:n_pairs]
            w_bnds = weights[n_pairs:]

            i_idx, j_idx = iu
            d_safe = np.where(d_pairs > 1e-8, d_pairs, 1e-8)
            vecs = x[i_idx] - x[j_idx]
            norm_vecs = vecs / d_safe[:, np.newaxis]

            np.add.at(grad, i_idx, -w_pairs[:, np.newaxis] * norm_vecs)
            np.add.at(grad, j_idx, w_pairs[:, np.newaxis] * norm_vecs)

            norms = np.linalg.norm(x, axis=1)
            norms_safe = np.where(norms > 1e-8, norms, 1e-8)
            norm_x = x / norms_safe[:, np.newaxis]

            np.add.at(grad, np.arange(N), w_bnds[:, np.newaxis] * norm_x)

            return obj, grad.flatten()
        return obj_func

    # Annealing schedule: start with low beta (smooth), increase for sharper min
    betas = [10, 30, 60, 100, 150, 200, 300]

    for cand_idx, current_x in enumerate(top_candidates):
        if time.time() > deadline - 10:
            break

        current_x = current_x.copy()
        local_best_x = current_x.copy()
        local_best_score = calc_score(current_x)

        for beta in betas:
            if time.time() > deadline - 10:
                break

            obj_func = make_obj_func(beta)
            try:
                res = minimize(obj_func, current_x.flatten(), method='L-BFGS-B', jac=True,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.x is not None:
                    new_x = res.x.reshape(N, 2)
                    new_score = calc_score(new_x)
                    if new_score > local_best_score:
                        local_best_score = new_score
                        local_best_x = new_x
                    current_x = new_x
            except Exception:
                pass

        if local_best_score > best_score:
            best_score = local_best_score
            best_x = local_best_x

    # Particle Swarm Optimization (PSO) to escape local minima
    def pso_optimize(initial_pts, max_iter=5000, n_particles=20):
        """PSO over full 100D configuration space to maximize packing radius."""
        dim = N * 2
        particles = np.zeros((n_particles, dim))
        velocities = np.zeros((n_particles, dim))
        pbest = np.zeros((n_particles, dim))
        pbest_scores = np.zeros(n_particles)
        gbest = initial_pts.flatten().copy()
        gbest_score = calc_score(initial_pts)

        # Initialize particles around initial solution
        for p in range(n_particles):
            if p == 0:
                particles[p] = initial_pts.flatten()
            else:
                particles[p] = initial_pts.flatten() + np.random.randn(dim) * 0.05
            # Clip to unit circle
            pts = particles[p].reshape(N, 2)
            norms = np.linalg.norm(pts, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                pts[mask] = pts[mask] / norms[mask, None] * 0.999
                particles[p] = pts.flatten()
            score = calc_score(particles[p].reshape(N, 2))
            pbest[p] = particles[p]
            pbest_scores[p] = score
            if score > gbest_score:
                gbest = particles[p].copy()
                gbest_score = score

        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5

        for it in range(max_iter):
            if time.time() > deadline - 8:
                break

            w = w_start + (w_end - w_start) * (it / max_iter)

            for p in range(n_particles):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                velocities[p] = (w * velocities[p] +
                                c1 * r1 * (pbest[p] - particles[p]) +
                                c2 * r2 * (gbest - particles[p]))

                # Clamp velocity - use adaptive max based on current best radius
                max_v = max(0.02, 0.08 * gbest_score)
                velocities[p] = np.clip(velocities[p], -max_v, max_v)

                particles[p] = particles[p] + velocities[p]

                # Clip to unit circle
                pts = particles[p].reshape(N, 2)
                norms = np.linalg.norm(pts, axis=1)
                mask = norms > 0.999
                if np.any(mask):
                    pts[mask] = pts[mask] / norms[mask, None] * 0.999
                    particles[p] = pts.flatten()

                score = calc_score(particles[p].reshape(N, 2))
                if score > pbest_scores[p]:
                    pbest[p] = particles[p]
                    pbest_scores[p] = score
                    if score > gbest_score:
                        gbest = particles[p].copy()
                        gbest_score = score

        return gbest.reshape(N, 2), gbest_score

    # Force-based local refinement (inspired by inspiration 1)
    def force_refine(pts, max_iter=3000, r_target=None):
        """Simple gradient-free local optimization using repulsion forces."""
        pts = pts.copy()
        if r_target is None:
            r_target = calc_score(pts) * 1.001
        dt = 0.005
        patience = 0
        best_local_r = calc_score(pts)
        best_local_pts = pts.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute forces
            forces = np.zeros_like(pts)
            for i in range(N):
                # Boundary force: push inward if too close to boundary
                norm_i = np.linalg.norm(pts[i])
                if norm_i > 1e-10:
                    margin = 1.0 - norm_i - r_target
                    if margin < 0:
                        # Push inward (toward origin) when margin is negative
                        forces[i] += (pts[i] / norm_i) * (margin * 2.0)
                
                # Pairwise repulsion
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d < 1e-8:
                        forces[i] += np.random.rand(2) * 0.01
                        forces[j] -= np.random.rand(2) * 0.01
                        continue
                    target_d = 2.0 * r_target
                    if d < target_d:
                        f = (target_d - d) / d
                        forces[i] += diff * f
                        forces[j] -= diff * f
                    elif d > target_d * 1.15:
                        f = (d - target_d) / d * 0.03
                        forces[i] -= diff * f
                        forces[j] += diff * f
            
            # Move points
            step = dt / (1.0 + 0.05 * it)
            new_pts = pts + forces * step
            # Clip to unit circle
            norms = np.linalg.norm(new_pts, axis=1)
            mask = norms > 0.999
            new_pts[mask] = new_pts[mask] / norms[mask, None] * 0.999
            
            # Check if improved
            new_r = calc_score(new_pts)
            if new_r > best_local_r + 1e-10:
                best_local_r = new_r
                best_local_pts = new_pts.copy()
                pts = new_pts
                patience = 0
                dt = min(dt * 1.03, 0.02)
                # Gradually increase target
                r_target = min(r_target * 1.0001, 0.13)
            else:
                patience += 1
                dt *= 0.95
                if patience > 30:
                    break
        
        return best_local_pts, best_local_r

    # Apply PSO to best solution to explore for better local minima
    if time.time() < deadline - 40:
        pso_time_budget = min(30.0, deadline - 40 - time.time())
        if pso_time_budget > 5.0:
            pso_start = time.time()
            # Run PSO with time-based iteration limit
            max_pso_iter = int(pso_time_budget / 0.008)  # estimate ~8ms per iter
            max_pso_iter = max(max_pso_iter, 500)
            pso_x, pso_r = pso_optimize(best_x, max_iter=max_pso_iter, n_particles=15)
            if pso_r > best_score:
                best_x = pso_x
                best_score = pso_r

    # Apply force refinement to best solution
    if time.time() < deadline - 20:
        refined_x, refined_r = force_refine(best_x, max_iter=2000)
        if refined_r > best_score:
            best_x = refined_x
            best_score = refined_r

    # Final polish with Nelder-Mead on the best solution
    if time.time() < deadline - 15:
        def nm_obj(flat_x):
            x = flat_x.reshape(N, 2)
            d = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            min_pair = np.min(d) / 2.0
            min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
            return -min(min_pair, min_bound)

        try:
            res_nm = minimize(nm_obj, best_x.flatten(), method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
            if res_nm.x is not None:
                final_x = res_nm.reshape(N, 2)
                final_score = calc_score(final_x)
                if final_score > best_score:
                    best_x = final_x
                    best_score = final_score
        except Exception:
            pass

    # One more L-BFGS pass with very high beta if time permits
    if time.time() < deadline - 5:
        try:
            obj_func = make_obj_func(800)
            res = minimize(obj_func, best_x.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.x is not None:
                final_x = res.x.reshape(N, 2)
                final_score = calc_score(final_x)
                if final_score > best_score:
                    best_x = final_x
                    best_score = final_score
        except Exception:
            pass

    # Final force refinement pass
    if time.time() < deadline - 3:
        refined_x, refined_r = force_refine(best_x, max_iter=1000, r_target=best_score * 1.0005)
        if refined_r > best_score:
            best_x = refined_x
            best_score = refined_r

    # Final short PSO burst if time permits
    if time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        if remaining > 2.0:
            max_pso_iter = int(remaining / 0.008)
            max_pso_iter = max(max_pso_iter, 200)
            pso_x, pso_r = pso_optimize(best_x, max_iter=max_pso_iter, n_particles=10)
            if pso_r > best_score:
                best_x = pso_x
                best_score = pso_r

    return best_x
