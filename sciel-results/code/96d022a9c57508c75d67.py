"""
Portfolio-search program for Sutton-Chen TM cluster global minima.
Uses Basin Hopping with diverse initial structures (FCC, BCC, Random).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 295.0  # Leave 5s safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Vectorized distance check for efficiency
    # This is O(N^2) but N is small (<=80)
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n, scale=0.72):
    """Generate a compact FCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
    a = scale  # Lattice constant approx
    # FCC basis vectors
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                # FCC positions: (i,j,k), (i+0.5,j+0.5,k), (i+0.5,j,k+0.5), (i,j+0.5,k+0.5)
                # Actually, just generate a grid and take closest to origin.
                # Simpler: Generate random points in a sphere and keep closest?
                # Better: Generate FCC lattice points.
                # Point 1: (i, j, k)
                pts.append(np.array([i, j, k]) * a)
                # Point 2: (i+0.5, j+0.5, k)
                pts.append(np.array([i+0.5, j+0.5, k]) * a)
                # Point 3: (i+0.5, j, k+0.5)
                pts.append(np.array([i+0.5, j, k+0.5]) * a)
                # Point 4: (i, j+0.5, k+0.5)
                pts.append(np.array([i, j+0.5, k+0.5]) * a)
    pts = np.array(pts)
    # Calculate distance to origin
    dists = np.linalg.norm(pts, axis=1)
    # Get indices of n closest
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _make_bcc(n, scale=0.72):
    """Generate a compact BCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
    a = scale
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                # BCC positions: (i,j,k) and (i+0.5, j+0.5, k+0.5)
                pts.append(np.array([i, j, k]) * a)
                pts.append(np.array([i+0.5, j+0.5, k+0.5]) * a)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _make_random_compact(n, scale=0.72, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = scale * n ** (1.0 / 3.0) * 1.2
    points = []
    # Rejection sampling to ensure density
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            if points:
                d = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(d) < 0.5 * scale:
                    continue
            points.append(p)
    return np.array(points)


def _basin_hopping(initial_pos, potential, kind, time_budget):
    """
    Run Basin Hopping from initial_pos for the given time budget.
    Returns (best_energy, best_positions).
    """
    n = len(initial_pos)
    # Initial relaxation
    energy, positions = relax(initial_pos, potential, kind)
    if not _is_valid(positions):
        return np.inf, initial_pos
    
    best_energy = energy
    best_positions = positions.copy()
    
    T = 0.5  # Initial temperature
    perturbation_scale = 0.2
    rng = np.random.default_rng()
    
    while _time_left() > time_budget:
        # Perturb
        perturbation = rng.normal(0, perturbation_scale, positions.shape)
        new_positions = positions + perturbation
        
        # Relax
        try:
            new_energy, new_positions_relaxed = relax(new_positions, potential, kind)
        except Exception:
            continue
            
        if not np.all(np.isfinite(new_energy)):
            continue
            
        if not _is_valid(new_positions_relaxed):
            continue
            
        # Metropolis acceptance
        if new_energy < energy or rng.random() < np.exp(-(new_energy - energy) / T):
            energy = new_energy
            positions = new_positions_relaxed
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        
        # Cool
        T *= 0.995
        # Adjust perturbation scale slightly? Keep constant for simplicity.
        
    return best_energy, best_positions


def construct(instance):
    """
    Find the global minimum for the given instance.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    # Generate diverse initial structures
    inits = []
    
    # 1. FCC
    try:
        fcc_pos = _make_fcc(size)
        if _is_valid(fcc_pos):
            inits.append(fcc_pos)
    except Exception:
        pass
        
    # 2. BCC
    try:
        bcc_pos = _make_bcc(size)
        if _is_valid(bcc_pos):
            inits.append(bcc_pos)
    except Exception:
        pass
        
    # 3. Random Compact
    try:
        rand_pos = _make_random_compact(size, seed=size)
        if _is_valid(rand_pos):
            inits.append(rand_pos)
    except Exception:
        pass
        
    # 4. Another Random
    try:
        rand_pos2 = _make_random_compact(size, seed=size+123)
        if _is_valid(rand_pos2):
            inits.append(rand_pos2)
    except Exception:
        pass
        
    if not inits:
        # Fallback: simple sphere
        r = size ** (1/3)
        theta = np.linspace(0, 2*np.pi, size)
        phi = np.linspace(0, np.pi, size)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        inits.append(np.column_stack((x, y, z)))
        
    # Allocate time budget to each initial structure
    # We have ~295s. Let's give each a chunk, but ensure we finish.
    # If we have 4 inits, give each ~70s.
    # But we should check time left dynamically.
    
    best_energy = np.inf
    best_positions = None
    
    num_inits = len(inits)
    # Reserve 5s for final check/return
    total_time_available = _time_left()
    time_per_init = max(10, (total_time_available - 5) / num_inits)
    
    for i, init_pos in enumerate(inits):
        if _time_left() < 5:
            break
            
        # Run BH for this init
        try:
            e, pos = _basin_hopping(init_pos, potential, kind, time_per_init)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except Exception:
            continue
            
    # If we still have time, run more BH from the best found so far
    if best_positions is not None and _time_left() > 10:
        try:
            e, pos = _basin_hopping(best_positions, potential, kind, _time_left() - 5)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except Exception:
            pass
            
    if best_positions is None:
        # Absolute fallback
        best_positions = inits[0]
        
    return best_positions
# EVOLVE-BLOCK-END
