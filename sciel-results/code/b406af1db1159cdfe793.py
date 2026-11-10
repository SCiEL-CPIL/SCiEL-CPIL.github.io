"""Morse cluster global minima: Simulated Annealing with diverse structural seeds."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time tracking
_TOTAL_DEADLINE = 300.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _remaining():
    return _TOTAL_DEADLINE - _elapsed()


def _morse_energy(positions, rho):
    """Compute Morse potential energy."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 0, d, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    return energy


def _morse_energy_grad(positions, rho):
    """Compute Morse potential energy and gradient (vectorized)."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 0, d, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Vectorized gradient
    dv_dr = -rho * ex * (2.0 * ex - 1.0)
    # r_vec[i,j] = positions[i] - positions[j]
    r_vec = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_mat = np.where(d > 1e-10, d, 1.0)
    unit_vec = r_vec / r_mat[:, :, np.newaxis]
    grad = np.einsum('ij,ijk->ik', dv_dr, unit_vec)
    return energy, grad


def _fcc_cluster(n, spacing=1.0):
    """Generate FCC-like cluster."""
    a = spacing
    points = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                p = np.array([i, j, k]) * a
                points.append(p)
                points.append(p + np.array([a/2, a/2, 0]))
                points.append(p + np.array([a/2, 0, a/2]))
                points.append(p + np.array([0, a/2, a/2]))
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _bcc_cluster(n, spacing=1.0):
    """Generate BCC-like cluster."""
    a = spacing
    points = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                p = np.array([i, j, k]) * a
                points.append(p)
                points.append(p + np.array([a/2, a/2, a/2]))
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _hcp_cluster(n, spacing=1.0):
    """Generate HCP-like cluster."""
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    points = []
    for layer in range(6):
        z = layer * c
        if layer % 2 == 0:
            for i in range(6):
                for j in range(6):
                    x = i * a + (j % 2) * a/2
                    y = j * a * np.sqrt(3.0)/2.0
                    points.append(np.array([x, y, z]))
        else:
            for i in range(6):
                for j in range(6):
                    x = i * a + a/2
                    y = j * a * np.sqrt(3.0)/2.0 + a * np.sqrt(3.0)/6.0
                    points.append(np.array([x, y, z]))
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _icosahedral_shell(n, spacing=1.0):
    """Generate icosahedral shell cluster."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts[0]) * spacing
    
    points = [np.zeros(3)]
    points.extend(verts)
    
    # Add second shell
    s = spacing * 1.8
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append(np.array([i, j, k]) * s)
    
    # Add more points if needed
    rng = np.random.default_rng(42)
    while len(points) < n:
        theta = rng.uniform(0, 2*np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1-z*z)
        p = np.array([r*np.cos(theta), r*np.sin(theta), z]) * spacing * 3.0
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _random_cluster(n, spacing=1.0, seed=None):
    """Generate random cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    return points


def _lbfgs_refine(positions, rho, maxiter=200, ftol=1e-10, gtol=1e-8):
    """Refine positions using L-BFGS-B with vectorized gradient."""
    n = positions.shape[0]
    
    def fun_and_grad(x):
        pos = x.reshape(n, 3)
        e, g = _morse_energy_grad(pos, rho)
        return e, g
    
    try:
        result = minimize(fun_and_grad, positions.ravel(), method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return result.x.reshape(n, 3), result.fun
    except:
        return positions, _morse_energy(positions, rho)


def _simulated_annealing(positions, rho, T_start=5.0, T_end=0.01, n_steps=2000, 
                          step_size=0.1, seed=None, time_limit=30.0):
    """Simulated annealing with Metropolis criterion."""
    rng = np.random.default_rng(seed)
    n = positions.shape[0]
    current_pos = positions.copy()
    current_energy, _ = _morse_energy_grad(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    start = time.time()
    cooling_rate = (T_end / T_start) ** (1.0 / n_steps)
    
    for i in range(n_steps):
        if _remaining() < 5.0:
            break
        if time.time() - start > time_limit:
            break
        
        T = T_start * (cooling_rate ** i)
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, size=current_pos.shape)
        
        # Relax
        relaxed, e = _lbfgs_refine(perturbed, rho, maxiter=50)
        
        # Metropolis accept
        delta = e - current_energy
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_pos = relaxed
            current_energy = e
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy


def _basin_hopping(positions, rho, n_iterations=50, step_size=0.3, seed=None, time_limit=30.0):
    """Basin hopping with Metropolis criterion."""
    rng = np.random.default_rng(seed)
    n = positions.shape[0]
    current_pos = positions.copy()
    current_energy, _ = _morse_energy_grad(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    start = time.time()
    for i in range(n_iterations):
        if _remaining() < 5.0:
            break
        if time.time() - start > time_limit:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, size=current_pos.shape)
        
        # Relax
        relaxed, e = _lbfgs_refine(perturbed, rho, maxiter=100)
        
        # Metropolis accept
        delta = e - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / 0.3):
            current_pos = relaxed
            current_energy = e
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy


def _check_valid(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    d = pdist(positions)
    return np.min(d) >= min_dist


def construct(instance):
    """Find global minimum for Morse cluster."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Generate diverse initial structures
    seeds = []
    seeds.append(_fcc_cluster(n, spacing=1.0))
    seeds.append(_bcc_cluster(n, spacing=1.0))
    seeds.append(_hcp_cluster(n, spacing=1.0))
    seeds.append(_icosahedral_shell(n, spacing=1.0))
    for i in range(6):
        seeds.append(_random_cluster(n, spacing=1.0, seed=42+i))
    
    best_energy = np.inf
    best_pos = None
    
    # First pass: relax all seeds
    for pos in seeds:
        if _remaining() < 10.0:
            break
        try:
            relaxed, e = _lbfgs_refine(pos, rho, maxiter=200)
            if _check_valid(relaxed, 0.5) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except:
            pass
    
    # Second pass: simulated annealing on best candidates
    if _remaining() > 20.0 and best_pos is not None:
        sa_starts = [best_pos]
        if _remaining() > 40.0:
            sa_starts.append(_fcc_cluster(n, spacing=1.0))
            sa_starts.append(_icosahedral_shell(n, spacing=1.0))
            sa_starts.append(_random_cluster(n, spacing=1.0, seed=99))
        
        for start_pos in sa_starts:
            if _remaining() < 15.0:
                break
            try:
                sa_pos, sa_e = _simulated_annealing(start_pos, rho, T_start=3.0, T_end=0.01, 
                                                     n_steps=1500, step_size=0.15, seed=0, time_limit=30.0)
                if _check_valid(sa_pos, 0.5) and sa_e < best_energy:
                    best_energy = sa_e
                    best_pos = sa_pos.copy()
            except:
                pass
    
    # Third pass: basin hopping on best
    if _remaining() > 15.0 and best_pos is not None:
        try:
            bh_pos, bh_e = _basin_hopping(best_pos, rho, n_iterations=40, step_size=0.25, seed=0, time_limit=25.0)
            if _check_valid(bh_pos, 0.5) and bh_e < best_energy:
                best_energy = bh_e
                best_pos = bh_pos.copy()
        except:
            pass
    
    # Final high-precision polish
    if best_pos is not None and _remaining() > 5.0:
        try:
            final_pos, final_e = _lbfgs_refine(best_pos, rho, maxiter=500, ftol=1e-12, gtol=1e-10)
            if _check_valid(final_pos, 0.5):
                best_pos = final_pos
                best_energy = final_e
        except:
            pass
    
    if best_pos is None:
        best_pos = _random_cluster(n, spacing=1.0, seed=123)
    
    return best_pos
