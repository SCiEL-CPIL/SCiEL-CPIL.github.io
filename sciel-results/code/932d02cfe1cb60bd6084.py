"""Rewritten program for packing 50 equal circles in a unit circle.
Uses stable soft-min with Simulated Annealing + L-BFGS polish, ring-based initialization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin for 150s limit


def _score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair, min_bound)


def _soft_min_obj_and_grad(centers_flat, beta):
    """
    Stable soft-min objective with gradient.
    Minimizing this maximizes the minimum clearance.
    """
    n = N
    centers = centers_flat.reshape(n, 2)

    # Compute all pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)

    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]

    # Boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms

    # All clearances
    pair_clear = pair_dists / 2.0
    all_clear = np.concatenate([pair_clear, bound_dists])

    # Numerically stable log-sum-exp
    shifted = -beta * all_clear
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum = max_shifted + np.log(np.sum(exp_shifted))
    soft_min_val = -log_sum / beta

    # Weights (softmax)
    weights = exp_shifted / np.sum(exp_shifted)

    # Gradient accumulation
    grad = np.zeros_like(centers)

    # Pair contributions
    for k in range(len(pair_clear)):
        w = weights[k]
        if w < 1e-15:
            continue
        i, j = i_idx[k], j_idx[k]
        d = pair_dists[k]
        if d > 1e-12:
            vec = (centers[i] - centers[j]) / d
            grad[i] += 0.5 * w * vec
            grad[j] -= 0.5 * w * vec

    # Boundary contributions
    for i in range(n):
        w = weights[n * (n - 1) // 2 + i]
        if w < 1e-15:
            continue
        if norms[i] > 1e-12:
            grad[i] -= w * centers[i] / norms[i]

    # Objective is -soft_min
    obj = -soft_min_val
    grad = -grad

    return obj, grad.flatten()


def _rbfgs_step(centers, beta, lr=0.01):
    """Single gradient descent step."""
    _, grad_flat = _soft_min_obj_and_grad(centers.flatten(), beta)
    grad = grad_flat.reshape(N, 2)
    return centers - lr * grad


def _sa_loop(centers, start_time, deadline, T_start=0.05, T_end=0.001, n_steps=300, beta=100.0):
    """
    Simulated Annealing loop to escape local minima.
    """
    best = centers.copy()
    best_score = _score(best)
    current = centers.copy()
    current_score = best_score
    T = T_start
    T_factor = (T_end / T_start) ** (1.0 / max(n_steps, 1))

    for step in range(n_steps):
        if time.time() > deadline - 5:
            break

        # Gradient step
        new_centers = _rbfgs_step(current, beta, lr=0.005)

        # Keep inside circle
        norms = np.linalg.norm(new_centers, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            new_centers[mask] *= 0.999 / norms[mask]

        new_score = _score(new_centers)

        # Accept/reject
        if new_score >= current_score:
            current = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = new_centers.copy()
        else:
            delta = new_score - current_score
            if delta > -1e-10 and np.random.random() < np.exp(delta / T):
                current = new_centers
                current_score = new_score
            else:
                # Perturb current to escape
                if np.random.random() < 0.3:
                    idx = np.random.randint(N)
                    perturb = np.random.normal(0, 0.01, size=2)
                    current[idx] += perturb
                    norm = np.linalg.norm(current[idx])
                    if norm > 0.999:
                        current[idx] *= 0.999 / norm
                    current_score = _score(current)

        T *= T_factor

    return best, best_score


def _optimize_lbfgs(centers, beta, maxiter=200, gtol=1e-9):
    """Run L-BFGS-B on the soft-min objective."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            centers.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        if res.success or res.status in [1, 2, 4]:
            new_centers = res.x.reshape(N, 2)
            if np.all(np.isfinite(new_centers)):
                return new_centers
    except Exception:
        pass
    return centers


def _get_ring_init(n, seed=0):
    """Ring-based initialization matching known optimal structure."""
    points = []

    # Center
    points.append([0.0, 0.0])

    # Ring 1: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.252
        points.append([r * np.cos(angle), r * np.sin(angle)])

    # Ring 2: 12 points
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.504
        points.append([r * np.cos(angle), r * np.sin(angle)])

    # Ring 3: 18 points
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.756
        points.append([r * np.cos(angle), r * np.sin(angle)])

    # Ring 4: remaining (13 points)
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1) + np.pi / max(remaining, 1)
        r = 0.91
        points.append([r * np.cos(angle), r * np.sin(angle)])

    return np.array(points[:n])


def _get_hex_lattice(n):
    """Generate hexagonal lattice points inside unit circle."""
    r_est = 0.125
    s = 2 * r_est

    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.98:
                points.append([x, y])

    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = [[rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)] for _ in range(n)]

    points = np.array(points)

    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            points = np.vstack([points, [[radius * np.cos(angle), radius * np.sin(angle)]]])

    # Sort by distance from center, take closest n
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_random_init(n, seed=0):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        points.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(points)


def _polish(centers, start_time, deadline, beta_start=50.0, beta_end=1000.0, n_steps=8):
    """Progressive beta polish with L-BFGS."""
    best = centers.copy()
    best_score = _score(best)

    betas = np.linspace(beta_start, beta_end, n_steps)

    for beta in betas:
        if time.time() > deadline - 3:
            break

        new_centers = _optimize_lbfgs(best, beta, maxiter=150, gtol=1e-9)
        new_score = _score(new_centers)

        if new_score > best_score:
            best_score = new_score
            best = new_centers.copy()

    return best, best_score


def _construct():
    """Main construction function."""
    start_time = time.time()

    # Generate multiple initializations
    initializations = [
        _get_ring_init(N, seed=0),
        _get_hex_lattice(N),
        _get_random_init(N, seed=42),
        _get_ring_init(N, seed=1),
        _get_random_init(N, seed=123),
    ]

    best_centers = None
    best_score = 0.0

    # Optimize from each initialization
    for init in initializations:
        if time.time() > DEADLINE - 30:
            break

        # First: moderate beta L-BFGS
        centers = _optimize_lbfgs(init, beta=30.0, maxiter=150, gtol=1e-8)

        # Then: SA to escape local minima
        sa_deadline = min(DEADLINE - 25, start_time + 15)
        centers, sa_score = _sa_loop(centers, start_time, sa_deadline,
                                     T_start=0.03, T_end=0.001, n_steps=200, beta=50.0)

        # Then: progressive polish
        centers, score = _polish(centers, start_time, DEADLINE - 15,
                                 beta_start=50.0, beta_end=500.0, n_steps=6)

        if score > best_score:
            best_score = score
            best_centers = centers.copy()

    if best_centers is None:
        return initializations[0]

    # Final high-beta polish on the best result
    if time.time() < DEADLINE - 8:
        best_centers, best_score = _polish(best_centers, start_time, DEADLINE - 3,
                                           beta_start=300.0, beta_end=2000.0, n_steps=8)

    return best_centers


def construct():
    """Entry point: returns array of shape (N, 2) with circle centers."""
    return _construct()
