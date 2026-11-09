"""Morse cluster global minima search: combining structured initialization with basin hopping."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


# Global deadline management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
# We aim to finish by 290s to be safe, leaving 40s buffer for overhead/evaluation
_DEADLINE = _START_TIME + 290.0


def _time_remaining():
    return _DEADLINE - time.time()


def _morse_energy(rho, coords):
    """Compute Morse energy. Returns huge value if any pair is closer than 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()
    if min_d < 0.5:
        return 1e10
    x = rho * (1.0 - d)
    ex = np.exp(x)
    e_ij = ex * (ex - 2.0)
    return 0.5 * e_ij.sum()


def _morse_energy_grad(rho, coords):
    """Compute Morse energy and gradient."""
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros_like(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()
    if min_d < 0.5:
        return 1e10, np.zeros_like(coords)
    x = rho * (1.0 - d)
    ex = np.exp(x)
    e_ij = ex * (ex - 2.0)
    energy = 0.5 * e_ij.sum()

    # Gradient computation
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n,n,3)
    d_safe = np.where(d > 0, d, 1.0)
    # dE/dr = -2*rho*ex*(ex-1)
    # grad[i] = sum_j dE/dr_ij * (r_i - r_j)/r_ij
    coeff = -2.0 * rho * ex * (ex - 1.0) / d_safe
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijc->ic', coeff, diff)
    return energy, grad


def _local_minimize(coords, rho, time_limit=5.0):
    """Minimize energy from given starting point using L-BFGS-B."""
    n = len(coords)

    def objective(x):
        c = x.reshape(-1, 3)
        return _morse_energy(rho, c)

    def obj_grad(x):
        c = x.reshape(-1, 3)
        e, g = _morse_energy_grad(rho, c)
        return e, g

    x0 = coords.flatten()
    start_t = time.time()
    try:
        res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 300, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < 1e9:
            return res.fun, res.x.reshape(-1, 3)
        else:
            return objective(x0), coords
    except Exception:
        return objective(x0), coords


def _make_initial_config(n, rng, rho):
    """Create a reasonable initial configuration using FCC-like lattice."""
    # For smaller rho (long range), atoms spread out more
    # For larger rho (short range), atoms pack tighter
    # Rough radius estimate
    r_est = (n / 8.0) ** (1.0 / 3.0) * 1.5
    r_est = max(r_est, 3.0)

    a = 1.05  # lattice parameter
    positions = []
    max_idx = int(np.ceil(r_est / a)) + 1

    # Generate cubic grid points within sphere
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p = np.array([i * a, j * a, k * a])
                if np.linalg.norm(p) <= r_est:
                    positions.append(p)

    # Add FCC-like offsets
    for p in list(positions):
        positions.append(p + np.array([a * 0.5, a * 0.5, 0.0]))
        positions.append(p + np.array([a * 0.5, 0.0, a * 0.5]))
        positions.append(p + np.array([0.0, a * 0.5, a * 0.5]))

    # Shuffle and pick n points with minimum separation
    rng.shuffle(positions)
    chosen = []
    min_sep = 0.85
    for c in positions:
        if all(np.linalg.norm(c - ch) >= min_sep for ch in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break

    # If not enough, fill randomly
    while len(chosen) < n:
        p = rng.uniform(-r_est - 1, r_est + 1, 3)
        if all(np.linalg.norm(p - ch) >= 0.5 for ch in chosen):
            chosen.append(p)
        else:
            for _ in range(20):
                p2 = rng.uniform(-r_est - 2, r_est + 2, 3)
                if all(np.linalg.norm(p2 - ch) >= 0.5 for ch in chosen):
                    chosen.append(p2)
                    break
            else:
                chosen.append(p)

    coords = np.array(chosen[:n])
    # Small random perturbation to break symmetry
    coords += rng.normal(0, 0.02, (n, 3))
    return coords


def _basin_hopping_step(coords, rho, displacement, rng):
    """One basin hopping step: displace + minimize."""
    displaced = coords + rng.normal(0, displacement, coords.shape)
    e, new_coords = _local_minimize(displaced, rho)
    return e, new_coords


def _solve_one_instance(instance, time_budget):
    """Solve one instance within time budget."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))

    start_time = time.time()
    deadline = start_time + time_budget

    # Generate diverse initial configurations
    rng = np.random.default_rng(hash((n, rho)) % (2**32))

    best_energy = float('inf')
    best_coords = None

    # First: try a few local minimizations from structured starts
    n_starts = 3
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
        coords = _make_initial_config(n, rng, rho)
        e, coords = _local_minimize(coords, rho)
        if e < best_energy:
            best_energy = e
            best_coords = coords.copy()

    # Now basin hopping from best
    displacement = 0.4
    max_hopping_steps = 150

    current_energy = best_energy
    current_coords = best_coords.copy()

    for step in range(max_hopping_steps):
        if time.time() > deadline - 15:
            break

        # Adaptive displacement
        disp = displacement * (1.0 + 0.15 * np.sin(step * 0.2))

        e_new, c_new = _basin_hopping_step(current_coords, rho, disp, rng)

        # Metropolis accept
        if e_new < current_energy:
            current_energy = e_new
            current_coords = c_new
            if e_new < best_energy:
                best_energy = e_new
                best_coords = c_new.copy()
        else:
            # Accept with probability (Metropolis criterion)
            delta = e_new - current_energy
            if delta < 1.0:
                # Temperature scales with energy magnitude
                temp = max(0.05, abs(current_energy) * 0.02)
                if rng.random() < np.exp(-delta / temp):
                    current_coords = c_new

        # Periodically try a fresh start if stuck
        if step % 30 == 0 and step > 0:
            coords = _make_initial_config(n, rng, rho)
            e, coords = _local_minimize(coords, rho)
            if e < best_energy:
                best_energy = e
                best_coords = coords.copy()
                current_coords = coords.copy()
                current_energy = e

    return best_coords


def construct(instance):
    """Find the lowest-energy arrangement of n atoms bound by Morse potentials."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))

    # Allocate time budget for this instance
    # We have 6 instances total, so roughly 45s each with safety margin
    time_budget = min(45.0, _time_remaining() - 5.0)
    if time_budget < 5.0:
        # Not enough time, just do a quick local minimization
        rng = np.random.default_rng(n)
        coords = _make_initial_config(n, rng, rho)
        e, coords = _local_minimize(coords, rho)
        return coords

    result = _solve_one_instance(instance, time_budget)

    # Final validation: ensure minimum distance constraint
    if result is not None and len(result) >= 2:
        d = cdist(result, result)
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.5:
            # If constraint violated, try to fix by spreading out
            # Simple fix: scale coordinates
            center = result.mean(axis=0)
            result = (result - center) * 1.1 + center
            # Check again
            d = cdist(result, result)
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.5:
                # Last resort: use the initial configuration
                rng = np.random.default_rng(n)
                coords = _make_initial_config(n, rng, rho)
                e, coords = _local_minimize(coords, rho)
                result = coords

    return result
