"""Soft-core annealing for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 285.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n <= 1:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.all(dists >= min_dist)

def _soft_core_energy_and_grad(positions, kind, sigma):
    """Compute Sutton-Chen energy + gradient plus soft-core repulsion."""
    n = positions.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Sutton-Chen energy and gradient
    sc_e, sc_g = sutton_chen_energy_gradient(positions, kind)
    if not np.isfinite(sc_e):
        return 1e10, np.zeros_like(positions)
    
    # Soft-core repulsion: sum_{i<j} 2 * (sigma^12 / r^12 - sigma^6 / r^6)
    # This is 2 * LJ_126(sigma, r) for each pair
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    
    # Avoid division by zero
    r_safe = np.where(r < 1e-10, 1e-10, r)
    
    sigma2 = sigma * sigma
    sigma6 = sigma2 ** 3
    sigma12 = sigma6 ** 2
    
    inv_r2 = 1.0 / r_safe ** 2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Pair energy: 2*(sigma12/r^12 - sigma6/r^6)
    pair_e = 2.0 * (sigma12 * inv_r12 - sigma6 * inv_r6)
    np.fill_diagonal(pair_e, 0.0)
    sc_e_total = sc_e + np.sum(pair_e)
    
    # Gradient of soft-core: d/dr [2*(sigma12/r^12 - sigma6/r^6)] = 2*(-12*sigma12/r^13 + 6*sigma6/r^7)
    # = -24*sigma12/r^13 + 12*sigma6/r^7
    # Force on i from j: -dV/dr * (r_vec / r) = (24*sigma12/r^13 - 12*sigma6/r^7) * (r_vec / r)
    # = (24*sigma12/r^14 - 12*sigma6/r^8) * r_vec
    # where r_vec = pos_i - pos_j
    inv_r4 = inv_r2 ** 2
    inv_r8 = inv_r4 ** 2
    inv_r14 = inv_r8 * inv_r6 * inv_r2  # r^-14
    # Actually: inv_r14 = inv_r2 * inv_r12 = inv_r2 * inv_r6^2
    inv_r14 = inv_r2 * inv_r12
    
    coeff = 24.0 * sigma12 * inv_r14 - 12.0 * sigma6 * inv_r8
    np.fill_diagonal(coeff, 0.0)
    
    # grad_soft[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # = sum_j coeff[i,j] * diff[i,j]
    soft_g = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    total_g = sc_g + soft_g
    
    return sc_e_total, total_g

def _soft_core_anneal(positions, kind, n_stages=25, sigma_max=None):
    """Anneal a cluster through decreasing soft-core sigma values."""
    n = positions.shape[0]
    if sigma_max is None:
        # Estimate based on cluster size: want initial sigma to cover the cluster
        sigma_max = 0.8 + 0.15 * np.sqrt(n)
    
    # Create geometric sequence of sigma values from sigma_max down to 0
    # Use more stages at the end for fine-tuning
    sigmas = np.geomspace(sigma_max, 1e-6, n_stages)
    
    pos = positions.copy()
    for i, sigma in enumerate(sigmas):
        if _time_left() < 15.0:
            break
        
        # For very small sigma, just do a regular relax
        if sigma < 0.01:
            try:
                energy, pos = relax(pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(pos):
                    return energy, pos
            except Exception:
                pass
            break
        
        def fun(x):
            try:
                p = x.reshape(-1, 3)
                e, g = _soft_core_energy_and_grad(p, kind, sigma)
                if not np.isfinite(e):
                    return e, np.zeros_like(x)
                return e, g
            except Exception:
                return 1e10, np.zeros_like(x)
        
        try:
            res = minimize(fun, pos.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if np.isfinite(res.fun):
                new_pos = res.x.reshape(-1, 3)
                if _is_valid(new_pos):
                    pos = new_pos
        except Exception:
            pass
    
    # Final polish with native relax
    try:
        energy, pos = relax(pos, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(pos):
            return energy, pos
    except Exception:
        pass
    
    # Fallback: compute energy directly
    try:
        e, _ = sutton_chen_energy_gradient(pos, kind)
        if np.isfinite(e) and _is_valid(pos):
            return e, pos
    except Exception:
        pass
    
    return 1e10, pos

def _generate_dense_random(n, seed=0):
    """Generate n atoms in a dense overlapping random sphere."""
    rng = np.random.default_rng(seed)
    # Place atoms in a sphere of radius ~1.5
    # Use random directions and radii
    pts = np.zeros((n, 3))
    for i in range(n):
        # Random direction
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(1 - z * z)
        direction = np.array([r * np.cos(theta), r * np.sin(theta), z])
        # Random radius (biased toward center for overlap)
        radius = rng.uniform(0, 1.5)
        pts[i] = direction * radius
    return pts

def _generate_fcc(n):
    """Generate FCC-like cluster."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_bcc(n):
    """Generate BCC-like cluster."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _worker_anneal(args):
    """Worker: run soft-core annealing from a given initial structure."""
    positions, kind, seed, n_stages, sigma_max = args
    try:
        energy, pos = _soft_core_anneal(positions, kind, n_stages=n_stages, sigma_max=sigma_max)
        if np.isfinite(energy) and _is_valid(pos):
            return (energy, pos)
    except Exception:
        pass
    return (np.inf, positions)

def _worker_relax(args):
    """Worker: just relax a given structure (for lattice starts)."""
    positions, kind = args
    try:
        energy, pos = relax(positions, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(pos):
            return (energy, pos)
    except Exception:
        pass
    return (np.inf, positions)

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Quick relax of lattice structures (fast, gives good baselines)
    lattice_starts = []
    try:
        fcc = _generate_fcc(n)
        if _is_valid(fcc):
            lattice_starts.append(fcc)
    except Exception:
        pass
    try:
        bcc = _generate_bcc(n)
        if _is_valid(bcc):
            lattice_starts.append(bcc)
    except Exception:
        pass
    
    for ls in lattice_starts:
        if _time_left() < 10.0:
            break
        try:
            e, p = relax(ls, 'sutton_chen', kind)
            if np.isfinite(e) and _is_valid(p) and e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass
    
    # Phase 2: Soft-core annealing from multiple dense random starts
    # Use parallel workers
    n_anneal_starts = 8
    n_workers = min(4, n_anneal_starts)
    
    anneal_tasks = []
    for i in range(n_anneal_starts):
        seed = 1000 + i * 31 + n
        try:
            init_pos = _generate_dense_random(n, seed=seed)
        except Exception:
            init_pos = _generate_fcc(n)
        
        # Vary sigma_max slightly for diversity
        sigma_max = 0.8 + 0.15 * np.sqrt(n) + 0.1 * i
        n_stages = 25
        
        anneal_tasks.append((init_pos, kind, seed, n_stages, sigma_max))
    
    if _time_left() > 30.0:
        try:
            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(_worker_anneal, anneal_tasks)
            for energy, pos in results:
                if np.isfinite(energy) and _is_valid(pos) and energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
        except Exception:
            # Fallback: sequential
            for task in anneal_tasks[:2]:
                if _time_left() < 15.0:
                    break
                try:
                    energy, pos = _soft_core_anneal(task[0], kind, n_stages=task[3], sigma_max=task[4])
                    if np.isfinite(energy) and _is_valid(pos) and energy < best_energy:
                        best_energy = energy
                        best_pos = pos.copy()
                except Exception:
                    pass
    
    # Phase 3: If time remains, run more annealing from perturbed best
    if best_pos is not None and _time_left() > 40.0:
        n_extra = min(4, int((_time_left() - 20.0) / 15.0))
        extra_tasks = []
        for i in range(n_extra):
            rng = np.random.default_rng(9000 + i * 7 + n)
            perturbed = best_pos + rng.normal(0, 0.3, best_pos.shape)
            if _is_valid(perturbed):
                sigma_max = 0.5 + 0.1 * np.sqrt(n)
                extra_tasks.append((perturbed, kind, 9000 + i * 7 + n, 20, sigma_max))
        
        if extra_tasks:
            n_workers_extra = min(4, len(extra_tasks))
            try:
                with mp.Pool(processes=n_workers_extra) as pool:
                    results = pool.map(_worker_anneal, extra_tasks)
                for energy, pos in results:
                    if np.isfinite(energy) and _is_valid(pos) and energy < best_energy:
                        best_energy = energy
                        best_pos = pos.copy()
            except Exception:
                for task in extra_tasks[:1]:
                    if _time_left() < 15.0:
                        break
                    try:
                        energy, pos = _soft_core_anneal(task[0], kind, n_stages=task[3], sigma_max=task[4])
                        if np.isfinite(energy) and _is_valid(pos) and energy < best_energy:
                            best_energy = energy
                            best_pos = pos.copy()
                    except Exception:
                        pass
    
    # Final polish
    if best_pos is not None and _time_left() > 5.0:
        try:
            e, p = relax(best_pos, 'sutton_chen', kind)
            if np.isfinite(e) and _is_valid(p) and e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass
    
    # Fallback
    if best_pos is None:
        for ls in lattice_starts:
            if _time_left() < 5.0:
                break
            try:
                e, p = relax(ls, 'sutton_chen', kind)
                if np.isfinite(e) and _is_valid(p):
                    best_energy = e
                    best_pos = p
                    break
            except Exception:
                continue
    
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            best_pos = np.random.default_rng(0).uniform(-2, 2, (n, 3))
    
    return best_pos

# EVOLVE-BLOCK-END
