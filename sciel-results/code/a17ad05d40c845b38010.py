"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Combines smooth soft-min surrogate (vectorized log-sum-exp) with iterative beta sharpening,
SA global search, and high-precision tangent polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()

    def _remaining(margin=5.0):
        return 150.0 - (time.time() - start_time) - margin

    def _compute_radii(centers):
        """Exact max radii for fixed centers (vectorized)."""
        c = np.asarray(centers, dtype=float)
        n = len(c)
        r_bound = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                             np.minimum(c[:, 1], 1.0 - c[:, 1]))
        dx = c[:, None, 0] - c[None, :, 0]
        dy = c[:, None, 1] - c[None, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist, axis=1)
        radii = np.minimum(r_bound, 0.5 * min_dist)
        radii = np.maximum(radii, 1e-12)
        return radii

    def _is_valid(x, y, r, tol=1e-8):
        """Check if configuration is valid."""
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        if np.any(dist < min_dist - tol):
            return False
        return True

    def _soft_obj_and_grad(params, beta):
        """
        Smooth approximation of -sum(min_j c_ij) using log-sum-exp.
        params: (2N,) array of [x_0..x_{N-1}, y_0..y_{N-1}]
        Returns (objective_value, gradient).
        """
        x = params[:N]
        y = params[N:2*N]

        # Boundary constraints
        b = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)  # (N, 4)

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)
        half_dist = 0.5 * dist

        # Constraint matrix: for each i, constraints are [b_i0, b_i1, b_i2, b_i3, half_dist_i0, ..., half_dist_i(N-1)]
        # We'll compute soft-min per circle
        # Total constraints per circle: 4 + N = N+4
        # Build constraint array (N, N+4)
        c = np.zeros((N, N + 4))
        c[:, :4] = b
        c[:, 4:] = half_dist
        # Set diagonal half_dist to large value
        for i in range(N):
            c[i, 4 + i] = 1e10

        # Soft-min: r_i = -1/beta * log(sum_j exp(-beta * c[i,j]))
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        log_sum = np.log(sum_exp)
        r = -(max_neg + log_sum) / beta
        r = r[:, 0]

        # Weights w[i,j] = exp(-beta*c[i,j]) / sum_j' exp(-beta*c[i,j'])
        w = exp_terms / sum_exp  # (N, N+4)

        # Gradient of r_i w.r.t. x_k, y_k
        # dr_i/dx_k = sum_j w[i,j] * dc[i,j]/dx_k
        # For boundary constraints (j=0..3):
        #   j=0 (left): dc/dx_i = 1, dc/dy_i = 0
        #   j=1 (right): dc/dx_i = -1, dc/dy_i = 0
        #   j=2 (bottom): dc/dx_i = 0, dc/dy_i = 1
        #   j=3 (top): dc/dx_i = 0, dc/dy_i = -1
        # For pairwise constraints (j=4+m, m!=i):
        #   dc/dx_i = (x_i - x_m) / (2*dist_im)
        #   dc/dy_i = (y_i - y_m) / (2*dist_im)
        #   dc/dx_m = (x_m - x_i) / (2*dist_im)
        #   dc/dy_m = (y_m - y_i) / (2*dist_im)
        #   dc/dx_k (k!=i,m) = 0

        # Compute gradients
        dr_dx = np.zeros(N)
        dr_dy = np.zeros(N)

        # Boundary contributions
        # w[:, 0] is weight for left, w[:, 1] for right, etc.
        dr_dx += w[:, 0] * 1.0 + w[:, 1] * (-1.0)
        dr_dy += w[:, 2] * 1.0 + w[:, 3] * (-1.0)

        # Pairwise contributions
        # For each pair (i, m) with m != i:
        #   contribution to dr_dx[i] += w[i, 4+m] * (x_i - x_m) / (2*dist_im)
        #   contribution to dr_dx[m] += w[m, 4+i] * (x_m - x_i) / (2*dist_im)
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        # Vectorized pairwise gradient
        # w_pair[i, m] = w[i, 4+m] for m != i
        w_pair = w[:, 4:]  # (N, N)
        # Set diagonal to 0
        np.fill_diagonal(w_pair, 0.0)

        # dc_dx_i[i,m] = (x_i - x_m) / (2*dist_im) = dx[i,m] / (2*dist_im)
        dc_dx = dx / (2.0 * dist_safe)
        dc_dy = dy / (2.0 * dist_safe)
        np.fill_diagonal(dc_dx, 0.0)
        np.fill_diagonal(dc_dy, 0.0)

        # dr_dx[i] += sum_m w_pair[i,m] * dc_dx[i,m]
        dr_dx += np.sum(w_pair * dc_dx, axis=1)
        dr_dy += np.sum(w_pair * dc_dy, axis=1)

        # dr_dx[m] += sum_i w_pair[m,i] * dc_dx[m,i]
        # Note: w_pair[m,i] * dc_dx[m,i] where dc_dx[m,i] = (x_m - x_i)/(2*dist)
        # This is already included in the sum above since we sum over axis=1
        # Wait, let me reconsider.
        # dr_dx[k] = sum_i [contribution from circle i's soft-min to x_k]
        # For circle i's soft-min, the gradient w.r.t. x_k is:
        #   if k == i: sum_m w[i,4+m] * (x_i - x_m)/(2*dist_im) + boundary terms
        #   if k != i: w[k,4+i] * (x_k - x_i)/(2*dist_ki)  [from circle k's own soft-min, constraint involving i]
        #              + w[i,4+k] * (x_i - x_k)/(2*dist_ik)  [from circle i's soft-min, constraint involving k]
        # But since dist_ik = dist_ki, and (x_i - x_k) = -(x_k - x_i),
        # the two cross terms from different circles' soft-mins don't cancel; they add.

        # Actually, let me redo this more carefully.
        # Objective = -sum_i r_i where r_i = soft_min_i
        # d(objective)/dx_k = -sum_i dr_i/dx_k
        #
        # dr_i/dx_k:
        #   = sum_j w[i,j] * dc[i,j]/dx_k
        #   where j runs over all constraints for circle i
        #
        # For k == i:
        #   boundary: w[i,0]*1 + w[i,1]*(-1) for dx; w[i,2]*1 + w[i,3]*(-1) for dy
        #   pairwise j=4+m (m!=i): w[i,4+m] * (x_i - x_m)/(2*dist_im)
        #
        # For k != i:
        #   boundary: 0
        #   pairwise j=4+k: w[i,4+k] * (x_i - x_k)/(2*dist_ik)  [constraint in circle i's soft-min involving k]
        #
        # So:
        # dr_dx[i] = w[i,0] - w[i,1] + sum_{m!=i} w[i,4+m] * (x_i - x_m)/(2*dist_im)
        # dr_dx[k] (k != i) gets contribution from circle i: w[i,4+k] * (x_i - x_k)/(2*dist_ik)
        #
        # Total dr_dx[k] = sum_i dr_i/dx_k
        #   = [boundary terms for k] + sum_{m!=k} w[k,4+m]*(x_k-x_m)/(2*dist_km)  [from circle k's own soft-min]
        #     + sum_{i!=k} w[i,4+k]*(x_i-x_k)/(2*dist_ik)  [from other circles' soft-mins]

        # Let me recompute:
        dr_dx = np.zeros(N)
        dr_dy = np.zeros(N)

        # Boundary terms
        dr_dx += w[:, 0] - w[:, 1]
        dr_dy += w[:, 2] - w[:, 3]

        # Own soft-min pairwise terms: for circle i, sum_{m!=i} w[i,4+m] * (x_i - x_m)/(2*dist_im)
        # This is: sum_m w_pair[i,m] * dc_dx[i,m] where dc_dx[i,m] = dx[i,m]/(2*dist_im)
        own_dx = np.sum(w_pair * dc_dx, axis=1)
        own_dy = np.sum(w_pair * dc_dy, axis=1)
        dr_dx += own_dx
        dr_dy += own_dy

        # Cross terms: for circle k, contribution from circle i's soft-min (i != k):
        # w[i,4+k] * (x_i - x_k)/(2*dist_ik)
        # = w_pair[i,k] * dc_dx[i,k]
        # Sum over i != k: sum_i w_pair[i,k] * dc_dx[i,k]
        # Note: w_pair[i,k] is (N,N) matrix, dc_dx[i,k] is (N,N)
        # cross_dx[k] = sum_i w_pair[i,k] * dc_dx[i,k]
        cross_dx = np.sum(w_pair * dc_dx, axis=0)
        cross_dy = np.sum(w_pair * dc_dy, axis=0)
        dr_dx += cross_dx
        dr_dy += cross_dy

        # Gradient of objective = -sum_i r_i is -[dr_dx, dr_dy]
        grad = np.concatenate([-dr_dx, -dr_dy])
        obj_val = -np.sum(r)
        return obj_val, grad

    def _gradient_polish(centers, betas=(50, 100, 200, 500, 1000), max_iter_per_beta=200):
        """Polish using L-BFGS-B with iterative beta sharpening."""
        x0 = np.concatenate([centers[:, 0], centers[:, 1]])
        bounds = [(0.0, 1.0)] * (2 * N)

        for beta in betas:
            if _remaining(10) < 5:
                break

            def obj(x):
                v, _ = _soft_obj_and_grad(x, beta)
                return v

            def grad(x):
                _, g = _soft_obj_and_grad(x, beta)
                return g

            try:
                res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': max_iter_per_beta, 'gtol': 1e-10, 'ftol': 1e-14})
                x0 = np.clip(res.x, 0.0, 1.0)
            except Exception:
                pass

        return np.column_stack([x0[:N], x0[N:]])

    def _best_fit_local_search(centers, max_sweeps=30, tol=1e-10):
        """Best-fit: for each circle, find position that maximizes its radius."""
        centers = np.array(centers, dtype=float)
        n = len(centers)

        for sweep in range(max_sweeps):
            if _remaining(10) < 5:
                break
            improved = False
            for i in range(n):
                others = np.delete(centers, i, axis=0)
                cur_r = min(
                    centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1]
                )
                for oc in others:
                    d = np.hypot(centers[i, 0] - oc[0], centers[i, 1] - oc[1])
                    r = d / 2.0
                    if r < cur_r:
                        cur_r = r

                best_r = cur_r
                best_pos = centers[i].copy()
                search_range = max(cur_r * 0.5, 0.02)
                n_grid = 9
                for dx in np.linspace(-search_range, search_range, n_grid):
                    for dy in np.linspace(-search_range, search_range, n_grid):
                        nx = centers[i, 0] + dx
                        ny = centers[i, 1] + dy
                        if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                            continue
                        r = min(nx, 1.0 - nx, ny, 1.0 - ny)
                        for oc in others:
                            d = np.hypot(nx - oc[0], ny - oc[1])
                            rr = d / 2.0
                            if rr < r:
                                r = rr
                        if r > best_r + tol:
                            best_r = r
                            best_pos = np.array([nx, ny])

                if np.linalg.norm(best_pos - centers[i]) > tol:
                    centers[i] = best_pos
                    improved = True

            if not improved:
                break

        return centers

    def _simulated_annealing(centers, T0=0.1, T_end=0.001, n_iter=3000, rng=None):
        """Simulated annealing."""
        if rng is None:
            rng = np.random.RandomState(0)
        centers = np.array(centers, dtype=float)
        n = len(centers)
        current_score = np.sum(_compute_radii(centers))
        best_centers = centers.copy()
        best_score = current_score

        for it in range(n_iter):
            T = T0 * (T_end / T0) ** (it / max(n_iter - 1, 1))
            if _remaining(20) < 5:
                break

            i = rng.randint(n)
            move_size = 0.02 + 0.05 * (T / T0)
            new_pos = centers[i].copy()
            new_pos[0] += rng.normal(0, move_size)
            new_pos[1] += rng.normal(0, move_size)
            new_pos = np.clip(new_pos, 0.01, 0.99)

            old_pos = centers[i].copy()
            centers[i] = new_pos
            new_score = np.sum(_compute_radii(centers))

            delta = new_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
            else:
                centers[i] = old_pos

        return best_centers, best_score

    def _tangent_solve_single(centers, i, max_iter=50):
        """For circle i, solve the 2D subproblem: maximize r_i."""
        n = len(centers)
        pos = centers[i].copy()
        others = np.delete(centers, i, axis=0)

        def radius_at(p):
            r = min(p[0], 1.0 - p[0], p[1], 1.0 - p[1])
            for oc in others:
                d = np.hypot(p[0] - oc[0], p[1] - oc[1])
                rr = d / 2.0
                if rr < r:
                    r = rr
            return r

        cur_r = radius_at(pos)

        # Adaptive multi-scale grid search
        for scale in [max(cur_r * 0.5, 0.02), max(cur_r * 0.2, 0.008), max(cur_r * 0.08, 0.002), max(cur_r * 0.03, 0.0005)]:
            if _remaining(5) < 2:
                break
            n_g = 13
            best_r = cur_r
            best_p = pos.copy()
            for dx in np.linspace(-scale, scale, n_g):
                for dy in np.linspace(-scale, scale, n_g):
                    np_ = np.array([pos[0] + dx, pos[1] + dy])
                    if np_.min() < -0.01 or np_.max() > 1.01:
                        continue
                    r = radius_at(np_)
                    if r > best_r + 1e-14:
                        best_r = r
                        best_p = np_
            if np.linalg.norm(best_p - pos) > 1e-14:
                pos = best_p
                cur_r = best_r

        # Newton polish
        for _ in range(max_iter):
            if _remaining(5) < 2:
                break
            r = radius_at(pos)
            tol_active = 1e-10
            active = []
            for coord, axis in [(pos[0], 'left'), (1.0 - pos[0], 'right'),
                                (pos[1], 'bottom'), (1.0 - pos[1], 'top')]:
                if abs(coord - r) < tol_active:
                    active.append(('bnd', axis))
            for oc in others:
                d = np.hypot(pos[0] - oc[0], pos[1] - oc[1])
                if abs(d / 2.0 - r) < tol_active:
                    active.append(('cir', oc))

            if len(active) < 2:
                break

            grad_sum = np.zeros(2)
            count = 0
            for a in active:
                if a[0] == 'bnd':
                    if a[1] == 'left':
                        g = np.array([1.0, 0.0])
                    elif a[1] == 'right':
                        g = np.array([-1.0, 0.0])
                    elif a[1] == 'bottom':
                        g = np.array([0.0, 1.0])
                    else:
                        g = np.array([0.0, -1.0])
                else:
                    oc = a[1]
                    d = np.hypot(pos[0] - oc[0], pos[1] - oc[1])
                    if d < 1e-12:
                        continue
                    g = np.array([(pos[0] - oc[0]) / (2.0 * d),
                                  (pos[1] - oc[1]) / (2.0 * d)])
                grad_sum += g
                count += 1

            if count == 0:
                break
            grad_sum /= count
            grad_norm = np.linalg.norm(grad_sum)
            if grad_norm < 1e-12:
                break

            step_sizes = [0.001, 0.0005, 0.0001, 0.00005]
            improved = False
            for ss in step_sizes:
                new_pos = pos + grad_sum / grad_norm * ss
                if new_pos.min() < -0.01 or new_pos.max() > 1.01:
                    continue
                new_r = radius_at(new_pos)
                if new_r > r + 1e-14:
                    pos = new_pos
                    cur_r = new_r
                    improved = True
                    break
            if not improved:
                break

        return pos, cur_r

    def _high_precision_polish(centers, max_iter=100):
        """Very fine best-fit local search with tangent solving."""
        centers = np.array(centers, dtype=float)
        n = len(centers)

        for sweep in range(max_iter):
            if _remaining(5) < 2:
                break
            improved = False
            for i in range(n):
                new_pos, new_r = _tangent_solve_single(centers, i, max_iter=30)
                if np.linalg.norm(new_pos - centers[i]) > 1e-12:
                    centers[i] = new_pos
                    improved = True

            if not improved:
                break

        return centers

    def _grid_init(seed):
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def _hex_init(seed):
        rng = np.random.RandomState(seed)
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    cx = (i + 0.5) / side + (0.5 / side) * (j % 2)
                    cy = (j + 0.5) / side
                    centers.append([cx, cy])
        centers = np.array(centers[:N])
        centers += rng.uniform(-0.03, 0.03, size=centers.shape)
        return np.clip(centers, 0.05, 0.95)

    def _random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.05, 0.95, size=(N, 2))

    def _evaluate_and_update(centers, best_state):
        """Evaluate centers and update best if improved."""
        radii = _compute_radii(centers)
        x, y, r = centers[:, 0], centers[:, 1], radii
        if _is_valid(x, y, r, tol=1e-6):
            score = np.sum(radii)
            if score > best_state['score']:
                best_state['score'] = score
                best_state['centers'] = centers.copy()
                best_state['radii'] = radii.copy()
                return True
        return False

    best_state = {'score': -np.inf, 'centers': None, 'radii': None}

    # Phase 1: Multiple SA restarts with diverse initializations
    seeds = [0, 1, 7, 13, 23, 37, 53, 71, 89, 42, 99, 123, 200, 300, 500, 777, 1234]

    for seed in seeds:
        if _remaining(50) < 5:
            break

        if seed == 0:
            centers = _grid_init(seed)
        elif seed == 1:
            centers = _hex_init(seed)
        else:
            centers = _random_init(seed)

        # Quick best-fit
        centers = _best_fit_local_search(centers, max_sweeps=15)

        # SA
        rng = np.random.RandomState(seed + 100)
        centers_sa, _ = _simulated_annealing(centers, T0=0.08, T_end=0.0005,
                                             n_iter=2500, rng=rng)

        # Best-fit polish
        centers_bf = _best_fit_local_search(centers_sa, max_sweeps=30)
        _evaluate_and_update(centers_bf, best_state)

    # Phase 2: Gradient polish with iterative beta sharpening
    if best_state['centers'] is not None:
        if _remaining(30) > 5:
            centers_grad = _gradient_polish(best_state['centers'],
                                            betas=(50, 100, 200, 500, 1000),
                                            max_iter_per_beta=200)
            centers_grad = _best_fit_local_search(centers_grad, max_sweeps=25)
            _evaluate_and_update(centers_grad, best_state)

    # Phase 3: High-precision polish
    if best_state['centers'] is not None:
        if _remaining(15) > 5:
            centers_hp = _high_precision_polish(best_state['centers'], max_iter=100)
            _evaluate_and_update(centers_hp, best_state)

        # Multiple gradient + high-precision cycles
        if _remaining(8) > 5:
            for _ in range(4):
                if _remaining(5) < 5:
                    break
                centers_g = _gradient_polish(best_state['centers'],
                                             betas=(200, 500, 1000, 2000, 5000),
                                             max_iter_per_beta=150)
                centers_g = _high_precision_polish(centers_g, max_iter=80)
                if not _evaluate_and_update(centers_g, best_state):
                    break

    # Final: one more high-precision pass
    if best_state['centers'] is not None and _remaining(3) > 2:
        centers_final = _high_precision_polish(best_state['centers'], max_iter=50)
        _evaluate_and_update(centers_final, best_state)

    # Fallback
    if best_state['centers'] is None:
        centers = _grid_init(0)
        radii = _compute_radii(centers)
        best_state['centers'] = centers
        best_state['radii'] = radii

    # Final safety check
    x, y, r = best_state['centers'][:, 0], best_state['centers'][:, 1], best_state['radii']
    if not _is_valid(x, y, r, tol=1e-6):
        centers = _grid_init(0)
        radii = _compute_radii(centers)
        best_state['centers'] = centers
        best_state['radii'] = radii

    result = np.column_stack([best_state['centers'], best_state['radii']])
    return result

# EVOLVE-BLOCK-END
