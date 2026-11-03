"""Logarithmic energy minimization for N points on the unit sphere.
Uses multi-start L-BFGS-B optimization with diverse initializations (Sobol,
Fibonacci, jittered, random), a perturbation-based polishing phase, and a
final BFGS high-precision polish on the absolute best. The gradient is the
true (tangent-projected) gradient of the energy, so all descent steps are
correct."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
# The evaluator kills the program after 150s. We use 145s as a hard deadline.
_DEADLINE = 145.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    """Project points onto the unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation for scoring initializations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and the TRUE tangent-projected gradient of the energy.

    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_k = - sum_{j != k} (x_k - x_j) / |x_k - x_j|^2

    The returned gradient is the true gradient projected onto the tangent
    space, so moving in the -gradient direction decreases the energy.
    """
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))

    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))

    # W_ij = 1/|x_i - x_j|^2 for i != j, 0 otherwise.
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)

    # sum_j W_kj * (x_k - x_j) = x_k * sum_j W_kj - sum_j W_kj * x_j
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw = sum_W * pts - Wx  # = sum_j (x_k - x_j)/d^2

    # True gradient = -raw
    grad = -raw

    # Project onto tangent space: g = grad - (grad . x_k) x_k
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts

    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for high accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    """Generate random points on the sphere."""
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence (low discrepancy)."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)

    # Map [0,1]^2 to sphere uniformly
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]

    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _optimize_lbfgsb(pts, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B optimization from a starting point (true gradient)."""
    n = pts.shape[0]

    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)

    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxls': 50,
                'maxcor': 20
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _simulated_annealing(pts, time_limit=10.0, T_start=0.1, T_end=1e-5, eta=0.01):
    """
    Run Simulated Annealing for a specified time limit.
    Returns the best points and energy found.
    """
    n = pts.shape[0]
    current_pts = pts.copy()
    current_energy = _energy(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(123)
    start_time = time.time()
    step = 0
    total_steps_estimate = 5000.0
    
    while time.time() - start_time < time_limit:
        for _ in range(100):
            if time.time() - start_time >= time_limit:
                break
                
            # Decay temperature
            alpha = min(step / total_steps_estimate, 1.0)
            T = T_start * (T_end / T_start) ** alpha
            
            e, g = _energy_grad(current_pts)
            
            # Langevin step
            noise_scale = np.sqrt(2.0 * eta * T)
            noise = noise_scale * rng.standard_normal((n, 3))
            
            proposed = current_pts - eta * g + noise
            proposed = _project(proposed)
            
            proposed_energy = _energy(proposed)
            
            if proposed_energy < current_energy:
                current_pts = proposed
                current_energy = proposed_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = proposed.copy()
            else:
                delta = proposed_energy - current_energy
                if delta < 0:
                    current_pts = proposed
                    current_energy = proposed_energy
                else:
                    if T > 1e-12:
                        accept_prob = math.exp(-delta / T)
                        if rng.random() < accept_prob:
                            current_pts = proposed
                            current_energy = proposed_energy
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_pts = proposed.copy()
            step += 1
            
    return best_pts, best_energy

def _optimize_bfgs(pts, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Run BFGS optimization (cheap, high-precision local polish)."""
    n = pts.shape[0]

    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)

    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='BFGS',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    rng = np.random.default_rng(42)

    # Generate diverse initializations
    initializations = []

    # 1. Sobol sequences (Low discrepancy) - Primary source of good starts
    for s in range(30):
        initializations.append(_sobol_sphere(N, seed=s))

    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))

    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(15):
        jitter_scale = 0.05 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))

    # 4. Random points with different seeds
    for s in range(20):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))

    best_pts = None
    best_energy = float('inf')

    # Score all initializations and sort
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))

    scored_inits.sort(key=lambda x: x[0])

    # Optimize top candidates
    top_candidates = scored_inits[:60]

    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 15.0:
            break

        # Run L-BFGS-B optimization (correct true gradient)
        optimized = _optimize_lbfgsb(init_pts, max_iter=1000, ftol=1e-15, gtol=1e-12)

        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()

    # Iterative Polish Phase
    # Take the best result, perturb it slightly, and re-optimize.
    # This can help escape shallow local minima or refine the solution further.
    if best_pts is not None:
        current_best = best_pts.copy()
        current_e = best_energy

        for polish_round in range(5):
            if time.time() > deadline - 5.0:
                break

            found_improvement = False
            for p in range(3):
                if time.time() > deadline - 5.0:
                    break

                # Small random perturbation
                noise = 0.01 * rng.standard_normal((N, 3))
                perturbed = current_best + noise
                perturbed = _project(perturbed)

                # Optimize the perturbed version
                optimized = _optimize_lbfgsb(perturbed, max_iter=500, ftol=1e-15, gtol=1e-12)
                e_precise = _precise_energy(optimized)

                if e_precise < current_e:
                    current_e = e_precise
                    current_best = optimized.copy()
                    found_improvement = True

            if not found_improvement:
                break

        if current_e < best_energy:
            best_energy = current_e
            best_pts = current_best

    # Final high-precision polish on the absolute best.
    # First a tight L-BFGS-B pass, then a BFGS pass (cheap, high-precision).
    if best_pts is not None and time.time() < deadline - 2.0:
        refined = _optimize_lbfgsb(best_pts, max_iter=2000, ftol=1e-16, gtol=1e-14)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined

        # BFGS final polish (cheap, can reach very tight tolerance)
        if time.time() < deadline - 1.0:
            bfgs_pts = _optimize_bfgs(best_pts, max_iter=500, ftol=1e-16, gtol=1e-14)
            e_precise = _precise_energy(bfgs_pts)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = bfgs_pts

    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)

    # Final normalization and validation
    best_pts = _project(best_pts)

    if not _valid(best_pts):
        # Fallback to a known valid configuration
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)

    return best_pts
