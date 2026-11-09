"""Improved program for Sutton-Chen cluster global minima search using Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _initialize_dense_random(n, rng, spacing=1.0):
    """Initialize a cluster with dense random packing."""
    radius = spacing * (n / (4.0/3.0 * np.pi)) ** (1/3.0) * 1.1
    points = []
    min_dist = spacing * 0.9
    attempts = 0
    max_attempts = n * 1000
    while len(points) < n and attempts < max_attempts:
        # Sample in sphere
        u = rng.random(3)
        r = radius * (u[0] ** (1/3.0))
        theta = 2 * np.pi * u[1]
        phi = np.arccos(2 * u[2] - 1)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        p = np.array([x, y, z])
        
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        attempts += 1
        
    # If we didn't fill it, add random points (might be close, relax will fix or fail)
    while len(points) < n:
        u = rng.random(3)
        r = radius * (u[0] ** (1/3.0))
        theta = 2 * np.pi * u[1]
        phi = np.arccos(2 * u[2] - 1)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points.append(np.array([x, y, z]))
        
    return np.array(points)


def _basin_hopping(positions, potential, kind, n_iter=100, T0=1.0, Tf=0.01, kick_size=0.5, rng=None):
    """Perform Basin Hopping optimization."""
    if rng is None:
        rng = np.random.default_rng()
        
    # Initial minimization
    try:
        E_curr, x_curr = relax(positions, potential, kind)
    except Exception:
        # If initial relax fails, try to repair or return original
        E_curr, x_curr = sutton_chen_energy_gradient(positions, kind)
        
    E_best = E_curr
    x_best = x_curr.copy()
    
    T = T0
    cooling_factor = (Tf / T0) ** (1.0 / max(1, n_iter - 1))
    
    for i in range(n_iter):
        # Kick
        x_prop = x_curr + rng.normal(0, kick_size, x_curr.shape)
        
        # Minimize proposed
        try:
            E_prop, x_prop_min = relax(x_prop, potential, kind)
        except Exception:
            # If relaxation fails, reject
            E_prop = np.inf
            x_prop_min = x_prop
            
        if np.isfinite(E_prop):
            dE = E_prop - E_curr
            # Metropolis acceptance
            if dE < 0 or rng.random() < np.exp(-dE / T):
                x_curr = x_prop_min
                E_curr = E_prop
                if E_prop < E_best:
                    E_best = E_prop
                    x_best = x_prop_min.copy()
        else:
            # If proposed state is invalid, we might just stay or restart kick? 
            # Staying is safer for stability
            pass
            
        T *= cooling_factor
        
    return E_best, x_best


def construct(instance):
    """Basin Hopping optimization for Sutton-Chen cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Determine number of iterations and runs based on cluster size to stay within time limit
    if size < 60:
        n_runs = 4
        n_iter = 100
    else:
        n_runs = 2
        n_iter = 60
        
    # Initial temperature
    T0 = 1.0
    Tf = 0.01
    
    # Kick size
    kick_size = 0.5
    
    best_energy = np.inf
    best_positions = None
    
    for run in range(n_runs):
        # Initialize
        x0 = _initialize_dense_random(size, rng)
        
        # Run BH
        E, x = _basin_hopping(x0, potential, kind, n_iter=n_iter, T0=T0, Tf=Tf, kick_size=kick_size, rng=rng)
        
        if np.isfinite(E) and E < best_energy:
            best_energy = E
            best_positions = x
            
    # Check validity of best solution
    if best_positions is not None:
        min_d = np.inf
        for a in range(len(best_positions)):
            for b in range(a+1, len(best_positions)):
                d = np.linalg.norm(best_positions[a] - best_positions[b])
                if d < min_d:
                    min_d = d
        if min_d < 0.35:
            best_energy = np.inf
            best_positions = None
            
    # Fallback if all failed
    if best_positions is None:
        x0 = _initialize_dense_random(size, rng)
        try:
            E, x = relax(x0, potential, kind)
            if np.isfinite(E):
                best_energy = E
                best_positions = x
        except Exception:
            pass
            
    return best_positions
# EVOLVE-BLOCK-END
