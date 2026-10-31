"""LJ38 cluster global minimum via Basin Hopping with Metropolis Acceptance."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

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
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    coeff = 4.0 * (12.0 * inv_r6 * inv_r6 * inv_r2 - 6.0 * inv_r6 * inv_r2)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _relax(pos):
    """Local L-BFGS relaxation."""
    try:
        res = minimize(_lj_energy, pos, jac=_lj_grad, method='L-BFGS-B',
                       options={'maxiter': 2000, 'gtol': 1e-8, 'ftol': 1e-10})
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

def construct():
    np.random.seed(42)
    
    # Initial seed: FCC truncated octahedron
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
    pos = points[:N].copy()
    pos = _enforce_min_separation(pos)
    
    # Relax initial
    current_pos = _relax(pos)
    current_energy = _lj_energy(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Basin Hopping with Metropolis Acceptance
    T = 1.0
    T_min = 0.01
    steps = 150
    perturbation_scale = 0.5
    
    for i in range(steps):
        # Perturb
        perturbation = np.random.randn(N, 3) * perturbation_scale
        new_pos = current_pos + perturbation
        new_pos = _enforce_min_separation(new_pos)
        
        # Relax
        new_pos = _relax(new_pos)
        new_pos = _enforce_min_separation(new_pos)
        new_energy = _lj_energy(new_pos)
        
        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            current_pos = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        # Cool down
        T *= 0.99
        if T < T_min:
            T = T_min
            
    return best_pos
