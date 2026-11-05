"""Dzugutov cluster global minima via continuous annealing of softness parameters."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Dzugutov potential constants
_A = 5.82
_B = 1.28
_a = 1.87
_b = 1.94
_c_real = 1.10
_d_real = 0.27
_m = 16

def _energy_and_gradient_vec(pos, c_val, d_val):
    """
    Vectorized computation of Dzugutov energy and gradient.
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    r = np.sqrt(np.maximum(r2, 1e-20))
    
    # Mask diagonal
    mask_diag = np.eye(n, dtype=bool)
    r = np.where(mask_diag, np.inf, r)
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    # Region 1: r < a
    mask_a = r < _a
    if np.any(mask_a):
        r_a = r[mask_a]
        exp_term = np.exp(c_val / (r_a - _a))
        V = _A * (r_a**(-_m) - _B) * exp_term
        energy += np.sum(V)
        
        dV_dr = _A * (-_m * r_a**(-_m - 1) * exp_term +
                      (r_a**(-_m) - _B) * exp_term * (-c_val / (r_a - _a)**2))
        
        factor = dV_dr / r_a
        idx_a = np.where(mask_a)
        d_a = diff[idx_a[0], idx_a[1]]
        
        np.add.at(grad, idx_a[0], factor[:, np.newaxis] * d_a)
        np.add.at(grad, idx_a[1], -factor[:, np.newaxis] * d_a)
        
    # Region 2: a <= r < b
    mask_b = (r >= _a) & (r < _b)
    if np.any(mask_b):
        r_b = r[mask_b]
        exp_term = np.exp(d_val / (r_b - _b))
        V = _B * exp_term
        energy += np.sum(V)
        
        dV_dr = _B * exp_term * (-d_val / (r_b - _b)**2)
        
        factor = dV_dr / r_b
        idx_b = np.where(mask_b)
        d_b = diff[idx_b[0], idx_b[1]]
        
        np.add.at(grad, idx_b[0], factor[:, np.newaxis] * d_b)
        np.add.at(grad, idx_b[1], -factor[:, np.newaxis] * d_b)
        
    return float(energy), grad

def _build_seed(n_atoms, rng):
    """
    Build a dense random seed structure with minimum distance enforcement.
    """
    points = []
    min_d = 0.6
    max_attempts = 1000
    R = (n_atoms * 8.0) ** (1/3) * 1.5
    
    for _ in range(n_atoms):
        added = False
        for _ in range(max_attempts):
            u = rng.uniform(-1, 1)
            v = rng.uniform(-1, 1)
            w = rng.uniform(-1, 1)
            norm = np.sqrt(u*u + v*v + w*w)
            if norm > 1:
                u, v, w = u/norm, v/norm, w/norm
            r = R * rng.uniform(0.8, 1.0)
            p = np.array([u, v, w]) * r
            
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_d:
                    valid = False
                    break
            if valid:
                points.append(p)
                added = True
                break
        if not added:
            # Force add at larger radius
            u = rng.uniform(-1, 1)
            v = rng.uniform(-1, 1)
            w = rng.uniform(-1, 1)
            norm = np.sqrt(u*u + v*v + w*w)
            if norm > 1:
                u, v, w = u/norm, v/norm, w/norm
            r = R * rng.uniform(1.0, 1.2)
            p = np.array([u, v, w]) * r
            points.append(p)
            
    return np.array(points)

def _anneal_trajectory(pos, n_steps=40, c_start=10.0, d_start=2.0):
    """
    Anneal c and d from high to real values, optimizing at each step.
    """
    pos = pos.copy()
    c_end = _c_real
    d_end = _d_real
    
    for i in range(n_steps):
        t = i / (n_steps - 1)
        # Cosine easing
        ease = np.cos(t * np.pi / 2)
        c_val = c_end + (c_start - c_end) * ease
        d_val = d_end + (d_start - d_end) * ease
        
        def obj(x):
            p = x.reshape(-1, 3)
            e, _ = _energy_and_gradient_vec(p, c_val, d_val)
            return e
        
        def jac(x):
            p = x.reshape(-1, 3)
            _, g = _energy_and_gradient_vec(p, c_val, d_val)
            return g.flatten()
            
        # Optimize
        # Reduce maxiter for early steps where landscape is smooth
        max_iter = int(50 + 150 * (1 - ease))
        res = minimize(obj, pos.flatten(), jac=jac, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-8})
        pos = res.x.reshape(-1, 3)
        
    return pos

def construct(instance):
    n_atoms = instance["n"]
    rng = np.random.default_rng(n_atoms * 7919)
    
    start_time = time.time()
    
    # Build seed
    pos = _build_seed(n_atoms, rng)
    
    # Anneal
    best_pos = _anneal_trajectory(pos, n_steps=50, c_start=12.0, d_start=2.5)
    
    # Final refinement with real potential
    def obj_real(x):
        p = x.reshape(-1, 3)
        e, _ = _energy_and_gradient_vec(p, _c_real, _d_real)
        return e
        
    def jac_real(x):
        p = x.reshape(-1, 3)
        _, g = _energy_and_gradient_vec(p, _c_real, _d_real)
        return g.flatten()
        
    res = minimize(obj_real, best_pos.flatten(), jac=jac_real, method='L-BFGS-B', 
                   options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
    
    final_pos = res.x.reshape(-1, 3)
    
    return final_pos
# EVOLVE-BLOCK-END
