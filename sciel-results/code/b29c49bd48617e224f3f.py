"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/d) and its gradient.
    
    dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)

    mask = dist_sq > 1e-30
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    # True gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j)/d^2
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        energy = 0.0
    else:
        energy = math.fsum(np.log(1.0 / d).tolist())

    return energy, grad


def _energy_only(pts):
    """Compute log energy only (faster, no gradient)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        return 0.0
    return math.fsum(np.log(1.0 / d).tolist())


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _tangent_project(grad, pts):
    """Project gradient to tangent space of sphere."""
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_sphere(n, seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _lbfgs_optimize(pts, time_limit=5.0, tight=False):
    """Run L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = _tangent_project(g, p)
        return g_tangent.flatten()

    x0 = pts.flatten()

    if tight:
        options = {
            'maxiter': 20000,
            'maxfun': 50000,
            'ftol': 1e-18,
            'gtol': 1e-16,
            'maxls': 300
        }
    else:
        options = {
            'maxiter': 5000,
            'maxfun': 15000,
            'ftol': 1e-15,
            'gtol': 1e-13,
            'maxls': 100
        }

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options=options
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _adam_optimize(pts, time_limit=10.0, lr=0.05):
    """Adam optimizer on the sphere with correct gradient."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    # Adam hyperparameters
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

        # Update momentum
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Project gradient to tangent space
        m_tangent = _tangent_project(m_hat, pts)

        # Update
        step = lr * m_tangent / (np.sqrt(v_hat) + eps)
        pts = pts - step
        pts = _project_to_sphere(pts)

        # Check convergence
        if t % 200 == 0:
            g_tangent = _tangent_project(grad, pts)
            if np.linalg.norm(g_tangent) < 1e-11:
                break

    return best_pts, best_energy


def _parallel_tempering(pts, time_limit=20.0, seed=42, n_replicas=4):
    """Parallel tempering (replica exchange) to escape local minima."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    # Initialize replicas with different temperatures - wider range
    base_T = 0.08
    T_list = [base_T * (0.25 ** i) for i in range(n_replicas)]
    T_min = 1e-8
    cooling = 0.999

    # Initialize replicas: start from same point, then perturb some more aggressively
    replicas = []
    energies = []
    for i in range(n_replicas):
        if i == 0:
            rep_pts = pts.copy()
        elif i == 1:
            rep_pts = pts + np.random.RandomState(seed + i).randn(n, 3) * 0.15
            rep_pts = _project_to_sphere(rep_pts)
        else:
            rep_pts = pts + np.random.RandomState(seed + i).randn(n, 3) * 0.3
            rep_pts = _project_to_sphere(rep_pts)
        replicas.append(rep_pts)
        e, _ = _energy_and_grad(rep_pts)
        energies.append(e)

    best_pts = pts.copy()
    best_energy = min(energies)
    best_idx = int(np.argmin(energies))
    best_pts = replicas[best_idx].copy()

    rng = np.random.RandomState(seed)
    exchange_interval = 5  # Try exchange every 5 steps

    step = 0
    while time.time() - start_time < time_limit:
        step += 1

        # Update each replica
        for i in range(n_replicas):
            T = T_list[i]
            if T < T_min:
                continue

            # Generate perturbation scaled by temperature
            r = rng.rand()
            if r < 0.1:
                noise_scale = T * 0.4
            else:
                noise_scale = T * 0.12
            perturbation = rng.randn(n, 3) * noise_scale
            new_pts = replicas[i] + perturbation
            new_pts = _project_to_sphere(new_pts)

            new_energy = _energy_only(new_pts)

            # Metropolis acceptance
            if new_energy < energies[i] or (T > T_min and rng.rand() < math.exp(-(new_energy - energies[i]) / T)):
                replicas[i] = new_pts
                energies[i] = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts.copy()

        # Replica exchange (swap adjacent replicas if it improves)
        if step % exchange_interval == 0:
            for i in range(n_replicas - 1):
                T_low = T_list[i]
                T_high = T_list[i + 1]
                E_low = energies[i]
                E_high = energies[i + 1]

                # Probability of accepting swap
                # Delta E = (E_high - E_low) * (1/T_low - 1/T_high)
                delta_E = (E_high - E_low) * (1.0 / T_low - 1.0 / T_high)
                if delta_E < 0 or (T_low > T_min and rng.rand() < math.exp(-delta_E)):
                    # Swap
                    replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]

        # Cool down all replicas
        for i in range(n_replicas):
            T_list[i] *= cooling

    return best_pts, best_energy


def _basin_hopping(pts, time_limit=20.0, seed=42):
    """Basin hopping: SA-like exploration with L-BFGS refinement after each perturbation."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.05
    T_min = 1e-6
    cooling = 0.998
    max_steps = 8000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        # Generate perturbation - mix of small and large kicks
        r = rng.rand()
        if r < 0.05:
            # Large kick: perturb many points significantly
            noise_scale = T * 0.5
            perturbation = rng.randn(n, 3) * noise_scale
        elif r < 0.15:
            # Medium kick: perturb a subset of points
            k = rng.randint(10, 30)
            idx = rng.choice(n, size=k, replace=False)
            perturbation = np.zeros((n, 3))
            perturbation[idx] = rng.randn(k, 3) * T * 0.3
        else:
            # Small kick
            noise_scale = T * 0.08
            perturbation = rng.randn(n, 3) * noise_scale
        
        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)

        # Quick L-BFGS polish from perturbed point (short time limit)
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining > 2.0:
            polish_limit = min(1.5, remaining * 0.25)
            polished_pts, polished_energy = _lbfgs_optimize(new_pts, time_limit=polish_limit)
        else:
            polished_pts = new_pts
            polished_energy = _energy_only(new_pts)

        # Accept or reject based on polished energy
        if polished_energy < current_energy or (T > T_min and rng.rand() < math.exp(-(polished_energy - current_energy) / T)):
            pts = polished_pts
            current_energy = polished_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        # Cool down
        T *= cooling
        if T < T_min:
            break

    return best_pts, best_energy


def _final_polish(pts, time_limit=5.0):
    """High-precision final polish with tight tolerances."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = _tangent_project(g, p)
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 30000,
            'maxfun': 80000,
            'ftol': 1e-20,
            'gtol': 1e-18,
            'maxls': 500
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (evaluator kills at 150s)

    best_pts = None
    best_energy = np.inf

    # Generate diverse starting points
    starts = []
    starts.append(_fibonacci_sphere(N))
    starts.append(_fibonacci_sphere(N, seed=1))
    starts.append(_fibonacci_sphere(N, seed=7))
    starts.append(_fibonacci_sphere(N, seed=13))
    for s in range(2, 10):
        starts.append(_random_sphere(N, seed=s * 100))

    # Phase 1: L-BFGS-B from each diverse start
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 35:
            break
        time_limit = min(5.0, deadline - elapsed - 30)
        if time_limit < 2.0:
            break
        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts.copy()

    # Phase 2: Adam optimization from best point
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 35:
            time_limit = min(10.0, deadline - elapsed - 30)
            candidate_pts, candidate_energy = _adam_optimize(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 3: Basin hopping to escape local minima (multiple seeds)
    for bh_seed in [42, 123, 777]:
        if best_pts is not None:
            elapsed = time.time() - start_time
            if elapsed < deadline - 25:
                time_limit = min(12.0, deadline - elapsed - 15)
                if time_limit < 3.0:
                    break
                candidate_pts, candidate_energy = _basin_hopping(
                    best_pts, time_limit=time_limit, seed=bh_seed
                )
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_pts = candidate_pts.copy()

    # Phase 4: Parallel tempering with different seed
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            time_limit = min(12.0, deadline - elapsed - 10)
            if time_limit < 3.0:
                elapsed = time.time() - start_time
                if elapsed < deadline - 8:
                    time_limit = min(5.0, deadline - elapsed - 3)
            if time_limit >= 3.0:
                candidate_pts, candidate_energy = _parallel_tempering(
                    best_pts, time_limit=time_limit, seed=123, n_replicas=5
                )
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_pts = candidate_pts.copy()

    # Phase 5: Parallel tempering for better exploration
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            time_limit = min(10.0, deadline - elapsed - 6)
            if time_limit >= 3.0:
                candidate_pts, candidate_energy = _parallel_tempering(
                    best_pts, time_limit=time_limit, seed=777, n_replicas=4
                )
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_pts = candidate_pts.copy()

    # Phase 6: L-BFGS-B polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            time_limit = min(6.0, deadline - elapsed - 5)
            if time_limit >= 2.0:
                candidate_pts, candidate_energy = _lbfgs_optimize(best_pts, time_limit=time_limit, tight=True)
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_pts = candidate_pts.copy()

    # Phase 7: High-precision final polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            time_limit = min(6.0, deadline - elapsed - 2)
            if time_limit >= 2.0:
                candidate_pts, candidate_energy = _final_polish(best_pts, time_limit=time_limit)
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_pts = candidate_pts.copy()

    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)

    if min_dist < 1e-9:
        # Points too close, use Fibonacci instead
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
