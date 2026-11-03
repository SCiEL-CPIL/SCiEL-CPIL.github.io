"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

# Best known packing radius
BEST_KNOWN_R = 0.1258254895304


def _compute_score(centers):
    """Compute the packing radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside unit circle
    dists = np.linalg.norm(centers, axis=1)
    if np.any(dists >= 1.0):
        return 0.0
    # Min distance from center to boundary
    r_boundary = np.min(1.0 - dists)
    # Min pairwise distance / 2
    if N > 1:
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        r_pair = np.min(d) / 2.0
    else:
        r_pair = r_boundary
    return min(r_boundary, r_pair)


def _energy(params, centers_init=None):
    """
    Energy function for optimization.
    params: array of shape (N*2,) representing 2D centers
    Returns a scalar energy to minimize (negative of the packing radius, with penalties).
    """
    centers = params.reshape(-1, 2)
    
    # Penalty for being outside the unit circle
    dists = np.linalg.norm(centers, axis=1)
    boundary_violation = np.maximum(0, dists - (1.0 - 1e-10))
    penalty_boundary = 100.0 * np.sum(boundary_violation**2)
    
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    
    # We want to maximize the minimum pairwise distance and min distance to boundary
    # Energy = -min(r_pair, r_boundary) + penalties
    r_boundary = np.min(1.0 - dists)
    r_pair = np.min(d) / 2.0
    r = min(r_boundary, r_pair)
    
    # If any violation, add large penalty
    if np.any(boundary_violation > 0):
        return -r + penalty_boundary
    
    return -r


def _energy_grad(params):
    """Gradient of the energy function (numerical approximation)."""
    eps = 1e-7
    grad = np.zeros_like(params)
    base = _energy(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        grad[i] = (_energy(params_plus) - base) / eps
    return grad


def _get_ring_initial():
    """
    Generate initial configuration based on concentric rings.
    Ring 0: center (1 circle)
    Ring 1: 6 circles
    Ring 2: 12 circles
    Ring 3: 15 circles (partial)
    Ring 4: 16 circles (partial)
    Total: 1 + 6 + 12 + 15 + 16 = 50
    """
    centers = np.zeros((N, 2))
    
    # Ring 0: center
    centers[0] = [0.0, 0.0]
    
    # Ring 1: 6 circles at radius ~0.25
    r1 = 0.25
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for i, a in enumerate(angles1):
        centers[1+i] = [r1 * np.cos(a), r1 * np.sin(a)]
    
    # Ring 2: 12 circles at radius ~0.5
    r2 = 0.5
    angles2 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for i, a in enumerate(angles2):
        centers[7+i] = [r2 * np.cos(a), r2 * np.sin(a)]
    
    # Ring 3: 15 circles at radius ~0.72
    r3 = 0.72
    angles3 = np.linspace(0, 2*np.pi, 15, endpoint=False)
    for i, a in enumerate(angles3):
        centers[19+i] = [r3 * np.cos(a), r3 * np.sin(a)]
    
    # Ring 4: 16 circles at radius ~0.88
    r4 = 0.88
    angles4 = np.linspace(0, 2*np.pi, 16, endpoint=False)
    for i, a in enumerate(angles4):
        centers[34+i] = [r4 * np.cos(a), r4 * np.sin(a)]
    
    return centers


def _optimize_from_start(centers_init, max_time=100.0):
    """
    Optimize from a given initial configuration using local optimization.
    Returns the best centers found.
    """
    start_time = time.time()
    best_centers = centers_init.copy()
    best_score = _compute_score(best_centers)
    
    params = centers_init.flatten().copy()
    
    # Use L-BFGS-B with bounds
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    def obj(x):
        return _energy(x)
    
    try:
        result = minimize(obj, params, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if time.time() - start_time > max_time:
            return best_centers
        new_centers = result.x.reshape(-1, 2)
        new_score = _compute_score(new_centers)
        if new_score > best_score:
            best_centers = new_centers
            best_score = new_score
    except Exception:
        pass
    
    # Polish with tighter tolerance
    start_time2 = time.time()
    try:
        result2 = minimize(obj, best_centers.flatten(), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
        if time.time() - start_time2 > max_time:
            return best_centers
        new_centers = result2.x.reshape(-1, 2)
        new_score = _compute_score(new_centers)
        if new_score > best_score:
            best_centers = new_centers
            best_score = new_score
    except Exception:
        pass
    
    return best_centers


def _hexagonal_lattice():
    """Generate a hexagonal lattice configuration as a starting point."""
    # Use hexagonal packing
    spacing = 0.26  # approximate spacing for r ~ 0.13
    centers_list = []
    
    # Generate points in hexagonal lattice
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                centers_list.append([x, y])
    
    centers = np.array(centers_list)
    if len(centers) >= N:
        # Sort by distance from center and take first N
        dists = np.linalg.norm(centers, axis=1)
        idx = np.argsort(dists)
        return centers[idx[:N]]
    else:
        # Fallback: ring configuration
        return _get_ring_initial()


def construct():
    """
    Main function to construct the packing of 50 circles in a unit circle.
    Returns an array of shape (N, 2) with the centers.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    best_centers = None
    best_score = 0.0
    
    # Strategy 1: Hexagonal lattice start
    centers_hex = _hexagonal_lattice()
    score_hex = _compute_score(centers_hex)
    if score_hex > best_score:
        best_score = score_hex
        best_centers = centers_hex
    
    if time.time() > deadline:
        return best_centers
    
    # Optimize from hexagonal start
    centers_opt1 = _optimize_from_start(centers_hex, max_time=min(60.0, deadline - time.time()))
    score_opt1 = _compute_score(centers_opt1)
    if score_opt1 > best_score:
        best_score = score_opt1
        best_centers = centers_opt1
    
    if time.time() > deadline:
        return best_centers
    
    # Strategy 2: Ring configuration start
    centers_ring = _get_ring_initial()
    score_ring = _compute_score(centers_ring)
    if score_ring > best_score:
        best_score = score_ring
        best_centers = centers_ring
    
    if time.time() > deadline:
        return best_centers
    
    # Optimize from ring start
    centers_opt2 = _optimize_from_start(centers_ring, max_time=min(40.0, deadline - time.time()))
    score_opt2 = _compute_score(centers_opt2)
    if score_opt2 > best_score:
        best_score = score_opt2
        best_centers = centers_opt2
    
    if time.time() > deadline:
        return best_centers
    
    # Strategy 3: Differential evolution on ring parameters
    if time.time() < deadline - 20.0:
        def ring_params_to_centers(params):
            """Convert ring parameters to centers."""
            # params: [r1, a1, r2, a2, r3, a3, r4, a4]
            # where ri is radius of ring i, ai is rotation angle
            r1, a1, r2, a2, r3, a3, r4, a4 = params
            
            centers = np.zeros((N, 2))
            centers[0] = [0.0, 0.0]
            
            angles1 = np.linspace(a1, a1 + 2*np.pi, 6, endpoint=False)
            for i, a in enumerate(angles1):
                centers[1+i] = [r1 * np.cos(a), r1 * np.sin(a)]
            
            angles2 = np.linspace(a2, a2 + 2*np.pi, 12, endpoint=False)
            for i, a in enumerate(angles2):
                centers[7+i] = [r2 * np.cos(a), r2 * np.sin(a)]
            
            angles3 = np.linspace(a3, a3 + 2*np.pi, 15, endpoint=False)
            for i, a in enumerate(angles3):
                centers[19+i] = [r3 * np.cos(a), r3 * np.sin(a)]
            
            angles4 = np.linspace(a4, a4 + 2*np.pi, 16, endpoint=False)
            for i, a in enumerate(angles4):
                centers[34+i] = [r4 * np.cos(a), r4 * np.sin(a)]
            
            return centers
        
        def ring_energy(params):
            centers = ring_params_to_centers(params)
            return -_compute_score(centers)
        
        bounds_de = [
            (0.1, 0.4),   # r1
            (0, 2*np.pi), # a1
            (0.3, 0.7),   # r2
            (0, 2*np.pi), # a2
            (0.5, 0.9),   # r3
            (0, 2*np.pi), # a3
            (0.7, 0.95),  # r4
            (0, 2*np.pi), # a4
        ]
        
        try:
            result_de = differential_evolution(ring_energy, bounds_de, maxiter=50, 
                                               popsize=15, seed=42, tol=1e-8,
                                               polish=True)
            if time.time() < deadline:
                centers_de = ring_params_to_centers(result_de.x)
                score_de = _compute_score(centers_de)
                if score_de > best_score:
                    best_score = score_de
                    best_centers = centers_de
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5.0 and best_centers is not None:
        try:
            bounds = [(-1.0, 1.0)] * (2 * N)
            
            def obj(x):
                return _energy(x)
            
            result_final = minimize(obj, best_centers.flatten(), method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            new_centers = result_final.x.reshape(-1, 2)
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_centers = new_centers
                best_score = new_score
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback
        best_centers = _hexagonal_lattice()
    
    return best_centers

# EVOLVE-BLOCK-END
