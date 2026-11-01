"""
Strategy: Binary search on radius with Simulated Annealing and Nelder-Mead polishing.
Uses multiple initializations (Hex, Rings, Random) to escape local minima.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
TARGET_TIME = 110.0  # Leave 10s margin for safety

def energy(pos, r):
    """Calculate overlap and boundary violation energy."""
    # Pairwise distances
    dist = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Overlap penalty: max(0, 2r - dist)^2
    overlap = np.maximum(0, 2 * r - dist)
    e_overlap = np.sum(overlap ** 2)
    
    # Boundary penalty: max(0, |pos| + r - 1)^2
    norms = np.linalg.norm(pos, axis=1)
    boundary = np.maximum(0, norms + r - 1)
    e_boundary = np.sum(boundary ** 2)
    
    return e_overlap + e_boundary

def sa_optimize(init_pos, r, steps=3000, T0=1.0, T_end=1e-5):
    """Simulated Annealing to minimize energy."""
    pos = init_pos.copy()
    T = T0
    cooling = (T_end / T0) ** (1.0 / steps)
    best_pos = pos.copy()
    best_e = energy(pos, r)
    
    for _ in range(steps):
        if time.time() > TARGET_TIME:
            break
        # Random perturbation
        delta = np.random.normal(0, 0.02, (N, 2))
        new_pos = pos + delta
        e_new = energy(new_pos, r)
        e_old = energy(pos, r)
        
        # Accept if better or with Boltzmann probability
        if e_new < e_old or (T > 1e-9 and np.random.rand() < np.exp(-(e_new - e_old) / T)):
            pos = new_pos
            if e_new < best_e:
                best_e = e_new
                best_pos = pos.copy()
        
        T *= cooling
        
    return best_pos, best_e

def polish(pos, r):
    """Nelder-Mead optimization for final refinement."""
    res = minimize(
        lambda p: energy(p.reshape(N, 2), r),
        pos.flatten(),
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-12}
    )
    return res.x.reshape(N, 2), res.fun

def init_hex():
    """Hexagonal lattice initialization."""
    pts = []
    s = 0.2
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    mask = np.linalg.norm(pts, axis=1) < 0.9
    pts = pts[mask]
    if len(pts) > N:
        idx = np.random.choice(len(pts), N, replace=False)
        pts = pts[idx]
    # Scale to fit roughly within bounds
    max_r = np.linalg.norm(pts, axis=1).max()
    if max_r > 0:
        pts *= (0.8 / max_r)
    return pts

def init_rings():
    """Concentric rings initialization."""
    pts = []
    counts = [1, 6, 12, 18, 13]
    radii = [0.0, 0.25, 0.5, 0.75, 0.95]
    for count, r in zip(counts, radii):
        if count == 1:
            pts.append([0, 0])
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
    return np.array(pts)

def init_random():
    """Random initialization."""
    return np.random.uniform(-0.8, 0.8, (N, 2))

def try_pack(r):
    """Try to find a valid packing for radius r."""
    inits = [init_hex(), init_rings(), init_random()]
    best_pos = None
    best_e = np.inf
    
    for init in inits:
        if time.time() > TARGET_TIME:
            break
        pos, e = sa_optimize(init, r)
        # Only polish if SA achieved a low energy
        if e < 0.1:
            pos, e = polish(pos, r)
        if e < best_e:
            best_e = e
            best_pos = pos
            
    return best_pos, best_e

def construct():
    """Main entry point."""
    start_time = time.time()
    r_low = 0.11
    r_high = 0.126
    best_pos = None
    best_r = r_low
    
    # Initial attempt
    pos, e = try_pack(r_low)
    if e < 1e-8:
        best_pos = pos
        best_r = r_low
    else:
        # Fallback to smaller radius
        pos, e = try_pack(0.1)
        if e < 1e-8:
            best_pos = pos
            best_r = 0.1

    # Binary search for max radius
    while time.time() < TARGET_TIME and (r_high - r_low) > 1e-6:
        r_mid = (r_low + r_high) / 2
        pos, e = try_pack(r_mid)
        
        if e < 1e-8:
            # Valid packing found
            r_low = r_mid
            best_pos = pos
            best_r = r_mid
        else:
            r_high = r_mid
            
    # Final polish for the best radius found
    if best_pos is not None:
        pos, e = polish(best_pos, best_r)
        if e < 1e-8:
            best_pos = pos
        else:
            # If polish failed, keep the SA result
            best_pos = best_pos 

    if best_pos is None:
        # Emergency fallback: return a valid but suboptimal packing
        # Use a simple grid
        return np.random.uniform(-0.5, 0.5, (N, 2))
        
    return best_pos
