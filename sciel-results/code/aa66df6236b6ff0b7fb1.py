"""LJ38 cluster global minimum via Multi-Start Basin Hopping with Metropolis acceptance."""
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
    """Vectorized Gradient of LJ energy."""
    # pos is (N, 3)
    R = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d2 = np.sum(R * R, axis=2)
    np.fill_diagonal(d2, 1.0)
    
    inv_r2 = 1.0 / d2
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    coeff = 4.0 * (12.0 * inv_r14 - 6.0 * inv_r8)
    np.fill_diagonal(coeff, 0.0)
    
    F = coeff[:, :, np.newaxis] * R
    grad = np.sum(F, axis=1)
    
    return grad

def _relax(pos):
    """Local L-BFGS relaxation."""
    try:
        res = minimize(_lj_energy, pos, jac=_lj_grad, method='L-BFGS-B',
                       options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10})
        return res.x
    except Exception:
        return pos

def _enforce_min_separation(pos, min_sep=0.5):
    """Push atoms apart if any pair is closer than min_sep."""
    for _ in range(20):
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
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy: Multi-start Basin Hopping
    # Use several random initial configurations to avoid the icosahedral trap.
    
    num_starts = 10
    steps_per_start = 30
    
    for start in range(num_starts):
        # Random initial positions in a box
        pos = np.random.uniform(-5, 5, (N, 3))
        pos = _enforce_min_separation(pos)
        
        current_pos = pos.copy()
        
        # Relax the start
        current_pos = _relax(current_pos)
        current_pos = _enforce_min_separation(current_pos)
        current_energy = _lj_energy(current_pos)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Basin Hopping from this start
        step_size = 0.5
        T = 1.0
        
        for step in range(steps_per_start):
            # Perturb
            perturbation = np.random.randn(N, 3) * step_size
            new_pos = current_pos + perturbation
            new_pos = _enforce_min_separation(new_pos)
            
            # Relax
            new_pos = _relax(new_pos)
            new_pos = _enforce_min_separation(new_pos)
            new_energy = _lj_energy(new_pos)
            
            # Metropolis acceptance
            if new_energy < current_energy:
                current_pos = new_pos
                current_energy = new_energy
            else:
                delta = new_energy - current_energy
                if np.random.rand() < np.exp(-delta / T):
                    current_pos = new_pos
                    current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                
            # Cool temperature and step size
            T *= 0.95
            step_size *= 0.98

    return best_pos
