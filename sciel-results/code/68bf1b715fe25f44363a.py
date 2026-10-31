"""LJ38 cluster global minimum via FCC truncated-octahedron seed + basin hopping."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 38

def _lj_energy(pos):
    """Total Lennard-Jones energy 4*sum(r^-12 - r^-6) over all pairs."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    d2 = d * d
    iu = np.triu_indices(N, k=1)
    r2 = d2[iu]
    inv_r6 = 1.0 / (r2 * r2 * r2)
    return 4.0 * np.sum(inv_r6 * inv_r6 - inv_r6)

def _lj_grad(pos):
    """Vectorized gradient of LJ energy."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    d2 = d * d
    iu = np.triu_indices(N, k=1)
    r2 = d2[iu]
    inv_r6 = 1.0 / (r2 * r2 * r2)
    # coeff = 4*(12*r^-14 - 6*r^-8) = 4*(12*inv_r6/r2 - 6*inv_r6/r2) = 4*inv_r6*(12-6)/r2 = 24*inv_r6/r2
    coeff = 24.0 * inv_r6 / r2
    # Build gradient
    grad = np.zeros_like(pos)
    rij = pos[iu[0]] - pos[iu[1]]
    grad[iu[0]] += coeff[:, np.newaxis] * rij
    grad[iu[1]] -= coeff[:, np.newaxis] * rij
    return grad

def _relax(pos):
    """Local L-BFGS relaxation."""
    try:
        res = minimize(_lj_energy, pos, jac=_lj_grad, method='L-BFGS-B',
                       options={'maxiter': 3000, 'gtol': 1e-9, 'ftol': 1e-12})
        return res.x
    except Exception:
        return pos

def _enforce_min_separation(pos, min_sep=0.5):
    """Push atoms apart if any pair is closer than min_sep."""
    for _ in range(100):
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
    """
    a = 2.0 ** (2.0 / 3.0)
    
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
    center = points.mean(axis=0)
    points -= center
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    pts = points[:N].copy()
    return pts

def _random_cluster():
    """Generate a random cluster with minimum separation."""
    np.random.seed(int(time.time() * 1000) % 100000)
    pos = np.random.randn(N, 3) * 2.0
    pos = _enforce_min_separation(pos)
    return pos

def _basin_hop(start_pos, n_iter=50, step_scale=0.2):
    """Basin hopping from a starting position."""
    pos = start_pos.copy()
    pos = _enforce_min_separation(pos)
    
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Relax initial
    relaxed = _relax(pos)
    relaxed = _enforce_min_separation(relaxed)
    e = _lj_energy(relaxed)
    if e < best_energy:
        best_energy = e
        best_pos = relaxed.copy()
    
    # Adaptive step size
    step = step_scale
    for i in range(n_iter):
        # Perturb
        perturbation = np.random.randn(N, 3) * step
        new_pos = best_pos + perturbation
        new_pos = _enforce_min_separation(new_pos)
        
        # Relax
        new_pos = _relax(new_pos)
        new_pos = _enforce_min_separation(new_pos)
        new_energy = _lj_energy(new_pos)
        
        # Always keep best
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            step = max(0.05, step * 1.1)  # Increase step when improving
        else:
            step = max(0.05, step * 0.95)  # Decrease step when not improving
    
    return best_pos, best_energy

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s buffer
    
    np.random.seed(42)
    
    # Multiple starting structures
    starts = []
    
    # FCC truncated octahedron
    fcc = _fcc_truncated_octahedron()
    fcc = _enforce_min_separation(fcc)
    starts.append(fcc)
    
    # Random clusters with different seeds
    for seed in [123, 456, 789, 321, 654]:
        np.random.seed(seed)
        rand = np.random.randn(N, 3) * 2.5
        rand = _enforce_min_separation(rand)
        starts.append(rand)
    
    best_pos = None
    best_energy = np.inf
    
    # Basin hopping from each start
    for s, start in enumerate(starts):
        if time.time() - start_time > timeout:
            break
        
        # Determine iterations based on remaining time
        remaining = timeout - (time.time() - start_time)
        # Estimate ~1.5s per basin hop iteration
        n_iter = max(10, int(remaining / 2.0))
        
        pos, energy = _basin_hop(start, n_iter=n_iter, step_scale=0.25)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    if best_pos is None:
        best_pos = starts[0].copy()
    
    return best_pos
