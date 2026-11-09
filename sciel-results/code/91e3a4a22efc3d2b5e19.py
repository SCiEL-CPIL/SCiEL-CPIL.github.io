"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _score(centers):
    """Compute the common radius (score) for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > CUBE_HALF):
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    mpd = np.min(dists)
    mbd = np.min(CUBE_HALF - np.abs(centers))
    return min(mpd / 2.0, mbd)


def _qp_objective(x, w_pair=1.0, w_bound=1.0):
    """
    QP-style objective: minimize negative sum of squared pairwise distances
    plus negative sum of squared boundary clearances.
    This is smooth and vectorizable.
    """
    c = x.reshape(N, 3)
    # Pairwise squared distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(sq_dists, 0.0)
    # Sum of upper triangle (avoid double counting)
    pair_sum = 0.5 * np.sum(sq_dists)
    
    # Boundary clearances squared
    bnd = CUBE_HALF - np.abs(c)
    bnd_sum = np.sum(bnd * bnd)
    
    # We want to maximize, so minimize negative
    return -(w_pair * pair_sum + w_bound * bnd_sum)


def _qp_gradient(x, w_pair=1.0, w_bound=1.0):
    """
    Gradient of the QP objective.
    d/d(c_i) [sum_{j!=i} |c_i - c_j|^2] = 2 * sum_{j!=i} (c_i - c_j)
    d/d(c_ik) [(0.5 - |c_ik|)^2] = -2*(0.5 - |c_ik|)*sign(c_ik)
    """
    c = x.reshape(N, 3)
    
    # Pairwise gradient: for each i, sum over j of (c_i - c_j)
    # diff[i,j] = c_i - c_j
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    # Sum over j axis for each i: sum_j (c_i - c_j) = N*c_i - sum_j c_j
    # But we need 2 * sum_j (c_i - c_j) for the gradient of sum |c_i-c_j|^2
    # Actually d/d(c_i) sum_j |c_i - c_j|^2 = 2 * sum_j (c_i - c_j)
    pair_grad = 2.0 * np.sum(diff, axis=1)  # shape (N, 3)
    
    # Boundary gradient
    bnd = CUBE_HALF - np.abs(c)
    signs = np.sign(c)
    # d/d(c_ik) (bnd_ik)^2 = 2*bnd_ik * d(bnd_ik)/d(c_ik) = 2*bnd_ik * (-sign(c_ik))
    bnd_grad = -2.0 * bnd * signs
    
    # Total gradient (we're minimizing the negative, so this is the gradient of the objective)
    grad = -(w_pair * pair_grad + w_bound * bnd_grad)
    
    return grad.flatten()


def _construct_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _construct_random():
    """Random initialization inside the cube."""
    return (np.random.rand(N, 3) - 0.5) * 0.99


def _local_optimize_qp(centers, w_pair=1.0, w_bound=1.0, maxiter=2000):
    """Run L-BFGS-B on the QP objective."""
    x0 = centers.flatten()
    result = minimize(
        _qp_objective,
        x0,
        jac=_qp_gradient,
        method='L-BFGS-B',
        bounds=[(-CUBE_HALF, CUBE_HALF)] * (N * 3),
        args=(w_pair, w_bound),
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.x.reshape(N, 3)


def _refine_qp(centers, time_limit):
    """
    Multi-stage QP refinement with annealed weights.
    Start with balanced weights, then emphasize boundary more to push spheres inward.
    """
    start_time = time.time()
    current = centers.copy()
    
    # Stage 1: Balanced optimization
    w_schedules = [
        (1.0, 1.0, 500),
        (1.0, 1.5, 500),
        (1.0, 2.0, 500),
        (1.5, 2.0, 500),
        (2.0, 3.0, 500),
        (3.0, 5.0, 1000),
    ]
    
    for w_pair, w_bound, maxiter in w_schedules:
        if time.time() - start_time > time_limit - 5:
            break
        try:
            current = _local_optimize_qp(current, w_pair, w_bound, maxiter)
            current = np.clip(current, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            if not np.all(np.isfinite(current)):
                break
        except Exception:
            break
    
    return current


def construct():
    """
    Construct centers of N equal spheres in the unit cube to maximize common radius.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    best_centers = _construct_grid()
    best_score = _score(best_centers)
    
    # Multiple initializations
    initializations = []
    
    # 1. Grid
    initializations.append(_construct_grid())
    
    # 2. Random starts
    for _ in range(5):
        initializations.append(_construct_random())
    
    # 3. Grid with perturbations
    grid = _construct_grid()
    for scale in [0.01, 0.02, 0.03]:
        perturbed = grid + np.random.randn(N, 3) * scale
        perturbed = np.clip(perturbed, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        initializations.append(perturbed)
    
    # 4. Layered structure (5x5x2)
    layered = []
    for x in np.linspace(-0.4, 0.4, 5):
        for y in np.linspace(-0.4, 0.4, 5):
            for z in [-0.25, 0.25]:
                layered.append([x, y, z])
    layered = np.array(layered[:N])
    initializations.append(layered)
    
    # 5. Face-centered arrangement
    fcc_like = []
    # 4x4x3 grid
    for x in np.linspace(-0.375, 0.375, 4):
        for y in np.linspace(-0.375, 0.375, 4):
            for z in np.linspace(-0.333, 0.333, 3):
                fcc_like.append([x, y, z])
    fcc_like = np.array(fcc_like[:N])
    initializations.append(fcc_like)
    
    # Run QP optimization on each initialization
    for i, init in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        init = np.clip(init, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        try:
            refined = _local_optimize_qp(init, w_pair=1.0, w_bound=1.0, maxiter=300)
            refined = np.clip(refined, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_centers = refined.copy()
        except Exception:
            continue
    
    # Final refinement of the best solution with more time
    if time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        try:
            final = _refine_qp(best_centers, time_limit=remaining)
            final = np.clip(final, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            s = _score(final)
            if s > best_score:
                best_score = s
                best_centers = final.copy()
        except Exception:
            pass
    
    # Final safety
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    if not np.all(np.isfinite(best_centers)):
        best_centers = _construct_grid()
    
    return best_centers
