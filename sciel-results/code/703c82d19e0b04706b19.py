import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

N = 50
DEADLINE_SECONDS = 140.0  # safety margin under the 150s kill


def _score(points):
    """Maximum valid radius for the given sphere centers."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)


def _objective_and_grad(c_flat, beta):
    """
    Smooth LSE surrogate of the packing objective.
    Returns (value, grad) for the function to MINIMIZE, i.e. -soft_min(g).
    g = [pair distances / 2, six wall distances per point].
    soft_min = -logsumexp(-beta*g)/beta  ->  minimize -soft_min.
    """
    c = c_flat.reshape(N, 3)

    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    tri_idx = np.triu_indices(N, k=1)
    d_pairs = d[tri_idx] / 2.0

    # Wall distances (6 per point)
    d_x_pos = 0.5 - c[:, 0]
    d_x_neg = c[:, 0] + 0.5
    d_y_pos = 0.5 - c[:, 1]
    d_y_neg = c[:, 1] + 0.5
    d_z_pos = 0.5 - c[:, 2]
    d_z_neg = c[:, 2] + 0.5
    wall_dists = np.concatenate([d_x_pos, d_x_neg,
                                 d_y_pos, d_y_neg,
                                 d_z_pos, d_z_neg])

    g_all = np.concatenate([d_pairs, wall_dists])

    # Soft min and its gradient (softmax of -beta*g)
    log_sum_exp = sp.logsumexp(-beta * g_all)
    val = -log_sum_exp / beta
    weights = np.exp(-beta * g_all - log_sum_exp)

    grad_c = np.zeros_like(c)

    # Pairwise part: d(d_ij/2)/dc_i = 0.5*(c_i-c_j)/d_ij
    for k in range(len(d_pairs)):
        w = weights[k]
        if w < 1e-12:
            continue
        i, j = tri_idx[0][k], tri_idx[1][k]
        diff_ij = c[i] - c[j]
        d_ij = np.linalg.norm(diff_ij)
        if d_ij > 1e-12:
            grad_c[i] += w * 0.5 * diff_ij / d_ij
            grad_c[j] -= w * 0.5 * diff_ij / d_ij

    # Wall part
    w_start = len(d_pairs)
    grad_c[:, 0] -= weights[w_start: w_start + N]
    grad_c[:, 0] += weights[w_start + N: w_start + 2 * N]
    grad_c[:, 1] -= weights[w_start + 2 * N: w_start + 3 * N]
    grad_c[:, 1] += weights[w_start + 3 * N: w_start + 4 * N]
    grad_c[:, 2] -= weights[w_start + 4 * N: w_start + 5 * N]
    grad_c[:, 2] += weights[w_start + 5 * N: w_start + 6 * N]

    # We minimize -soft_min, so flip sign.
    return -val, -grad_c.flatten()


def _polish(c, beta, max_iter=400, time_budget=8.0):
    """Steepest-ascent gradient polishing on the LSE surrogate."""
    start = time.time()
    c = c.copy()
    cur_val, cur_grad = _objective_and_grad(c.flatten(), beta)
    step = 0.02
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        if np.linalg.norm(cur_grad) < 1e-9:
            break
        # Line search for -objective (we want to increase soft_min)
        best = cur_val
        best_c = c
        for alpha in (step, step * 0.5, step * 0.25, step * 2.0):
            cand = c - alpha * cur_grad.reshape(N, 3)
            cand = np.clip(cand, -0.4999, 0.4999)
            v, g = _objective_and_grad(cand.flatten(), beta)
            if v < best:
                best = v
                best_c = cand
        if np.allclose(best_c, c):
            step *= 0.5
            continue
        c = best_c
        cur_val, cur_grad = _objective_and_grad(c.flatten(), beta)
    return c


def _initial_grid():
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid


def construct():
    start_time = time.time()
    best_points = _initial_grid()[:N].copy()
    best_score = _score(best_points)

    grid = _initial_grid()

    restart = 0
    # Beta schedule: anneal from smooth to sharp.
    betas = [12.0, 22.0, 40.0, 70.0, 110.0, 160.0, 220.0, 300.0]

    while time.time() - start_time < DEADLINE_SECONDS:
        beta = betas[min(restart, len(betas) - 1)]
        elapsed = time.time() - start_time
        remaining = DEADLINE_SECONDS - elapsed
        if remaining < 12.0:
            break

        # Choose initialization
        if restart == 0:
            c = grid[:N].copy()
            c += np.random.randn(N, 3) * 0.02
        elif restart == 1:
            c = grid[np.random.permutation(len(grid))[:N]].copy()
            c += np.random.randn(N, 3) * 0.03
        elif restart == 2:
            c = np.random.uniform(-0.42, 0.42, (N, 3))
        else:
            # Perturb the best found so far to escape local minima.
            c = best_points + np.random.randn(N, 3) * 0.04

        c = np.clip(c, -0.4999, 0.4999)

        # L-BFGS-B on the LSE surrogate
        try:
            res = opt.minimize(
                _objective_and_grad,
                c.flatten(),
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (3 * N),
                options={'maxiter': 200, 'maxfun': 2000,
                         'ftol': 1e-12, 'gtol': 1e-9, 'disp': False},
            )
            c = np.clip(res.x.reshape(N, 3), -0.4999, 0.4999)
        except Exception:
            pass

        # Gradient polish to extract residual radius
        if time.time() - start_time < DEADLINE_SECONDS - 6.0:
            c = _polish(c, beta,
                        max_iter=300,
                        time_budget=min(8.0, max(2.0, DEADLINE_SECONDS - (time.time() - start_time) - 4.0)))
            c = np.clip(c, -0.4999, 0.4999)

        sc = _score(c)
        if sc > best_score:
            best_score = sc
            best_points = c.copy()

        restart += 1

    # Final safety: ensure valid, finite, inside the cube.
    best_points = np.clip(best_points, -0.4999, 0.4999)
    if not np.all(np.isfinite(best_points)):
        best_points = grid[:N].copy()
    return best_points


if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
