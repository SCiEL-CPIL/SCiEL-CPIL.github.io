"""LJ38 cluster global minimum via FCC truncated-octahedron seed + basin hopping."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 38

def _lj_energy(pos):
    """Total Lennard-Jones energy 4*sum(r^-12 - r^-6) over all pairs."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    d2 = d * d
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    r2 = d2[iu]
    inv_r6 = 1.0 / (r2 * r2 * r2)
    return 4.0 * np.sum(inv_r6 * inv_r6 - inv_r6)

def _lj_energy_grad(pos):
    """Combined energy and gradient for efficiency."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    d2 = d * d
    iu = np.triu_indices(N, k=1)
    r2 = d2[iu]
    inv_r6 = 1.0 / (r2 * r2 * r2)
    energy = 4.0 * np.sum(inv_r6 * inv_r6 - inv_r6)
    
    grad = np.zeros_like(pos)
    for i in range(N):
        for j in range(i + 1, N):
            rij = pos[i] - pos[j]
            r2ij = np.dot(rij, rij)
            if r2ij < 1e-12:
                continue
            inv_r6ij = 1.0 / (r2ij * r2ij * r2ij)
            coeff = 4.0 * (12.0 * inv_r6ij * inv_r6ij - 6.0 * inv_r6ij)
            grad[i] += coeff * rij
            grad[j] -= coeff * rij
    return energy, grad

def _lj_grad(pos):
    """Gradient of LJ energy."""
    _, grad = _lj_energy_grad(pos)
    return grad

def _relax(pos):
    """Local L-BFGS relaxation."""
    try:
        res = minimize(_lj_energy, pos, jac=_lj_grad, method='L-BFGS-B',
                       options={'maxiter': 3000, 'gtol': 1e-10, 'ftol': 1e-12})
        return res.x
    except Exception:
        return pos

def _enforce_min_separation(pos, min_sep=0.5):
    """Push atoms apart if any pair is closer than min_sep."""
    for _ in range(50):
        moved = False
        d = cdist(pos, pos)
        np.fill_diagonal(d, np.inf)
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < min_sep:
                    rij = pos[i] - pos[j]
                    r = np.linalg.norm(rij)
                    if r < 1e-12:
                        rij = np.array([1.0, 0.0, 0.0])
                        r = 1.0
                    push = (min_sep - r) * 0.5
                    unit = rij / r
                    pos[i] += push * unit
                    pos[j] -= push * unit
                    moved = True
        if not moved:
            break
    return pos

def _fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron 38-atom cluster.
    
    For LJ with sigma=1, the equilibrium nearest-neighbor distance is 2^(1/6) ≈ 1.12246.
    The FCC lattice with conventional cube edge a has nearest-neighbor distance a/sqrt(2).
    So a = 2^(1/6) * sqrt(2) = 2^(2/3) ≈ 1.5874.
    """
    a = 2.0 ** (2.0 / 3.0)
    
    # Generate FCC lattice points in a range
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (i + bx) * a
                    y = (j + by) * a
                    z = (k + bz) * a
                    points.append([x, y, z])
    
    points = np.array(points)
    # Center the structure
    center = points.mean(axis=0)
    points -= center
    
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    # Take the 38 closest
    pts = points[:N].copy()
    
    return pts

def _random_cluster(seed=42):
    """Generate a random cluster with reasonable spacing."""
    rng = np.random.RandomState(seed)
    # Place atoms in a sphere of radius ~4
    pos = rng.randn(N, 3) * 1.5
    # Ensure minimum separation
    pos = _enforce_min_separation(pos)
    return pos

def construct():
    import time
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Start with the FCC truncated-octahedron seed
    pos = _fcc_truncated_octahedron()
    pos = _enforce_min_separation(pos)
    
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Basin hopping with multiple strategies
    np.random.seed(42)
    
    # First, relax the initial structure
    relaxed = _relax(pos)
    relaxed = _enforce_min_separation(relaxed)
    e = _lj_energy(relaxed)
    if e < best_energy:
        best_energy = e
        best_pos = relaxed.copy()
    
    # Multi-start basin hopping with varying perturbation sizes
    # Use different perturbation scales to explore different basins
    perturbation_scales = [0.3, 0.2, 0.15, 0.1, 0.25, 0.35, 0.2, 0.15]
    
    for scale_idx, scale in enumerate(perturbation_scales):
        if time.time() - start_time > time_limit:
            break
            
        # Do several perturbations at this scale
        n_perturb = 5 if scale >= 0.25 else 8
        for p in range(n_perturb):
            if time.time() - start_time > time_limit:
                break
            
            # Perturb the current best
            perturbation = np.random.randn(N, 3) * scale
            new_pos = best_pos + perturbation
            new_pos = _enforce_min_separation(new_pos)
            
            # Relax
            new_pos = _relax(new_pos)
            new_pos = _enforce_min_separation(new_pos)
            new_energy = _lj_energy(new_pos)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
    
    # Additional random restarts from different initial configurations
    for restart in range(3):
        if time.time() - start_time > time_limit:
            break
        
        # Random cluster
        rand_pos = _random_cluster(seed=42 + restart * 100)
        rand_relaxed = _relax(rand_pos)
        rand_relaxed = _enforce_min_separation(rand_relaxed)
        rand_energy = _lj_energy(rand_relaxed)
        
        if rand_energy < best_energy:
            best_energy = rand_energy
            best_pos = rand_relaxed.copy()
            
            # Basin hop from this new best
            for p in range(5):
                if time.time() - start_time > time_limit:
                    break
                perturbation = np.random.randn(N, 3) * 0.2
                new_pos = best_pos + perturbation
                new_pos = _enforce_min_separation(new_pos)
                new_pos = _relax(new_pos)
                new_pos = _enforce_min_separation(new_pos)
                new_energy = _lj_energy(new_pos)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
    
    return best_pos
