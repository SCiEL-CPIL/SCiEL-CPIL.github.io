"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(positions):
    """Compute the packing radius (minimum of pairwise distances/2 and distance to boundary)."""
    n = len(positions)
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    # Distance to boundary
    radii = np.hypot(positions[:, 0], positions[:, 1])
    min_boundary = 1.0 - np.max(radii)

    return min(min_pair, min_boundary)


def _negative_score(positions):
    """Negative of the packing radius for minimization."""
    return -_score(positions)


def _concentric_ring_init(n):
    """Initialize with concentric rings of circles."""
    positions = []
    # Center circle
    positions.append([0.0, 0.0])
    
    # Rings with increasing radius
    ring_radii = [0.25, 0.5, 0.75, 0.95]
    points_per_ring = [6, 12, 18, 13]  # Total: 1 + 6 + 12 + 18 + 13 = 50
    
    for r, count in zip(ring_radii, points_per_ring):
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            positions.append([r * np.cos(a), r * np.sin(a)])
    
    positions = np.array(positions[:n])
    return positions


def _random_init(n):
    """Random initialization inside the unit circle."""
    positions = np.zeros((n, 2))
    for i in range(n):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x * x + y * y <= 1.0:
                positions[i] = [x, y]
                break
    return positions


def _hex_init(n):
    """Hexagonal lattice initialization."""
    spacing = 0.22
    points = []
    k = 8
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r < 0.9:
                points.append((x, y))
    
    points = np.array(points)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n]
    return points[idx].copy()


def _simulated_annealing(positions, max_time=100.0, seed=42):
    """Simulated annealing optimization."""
    rng = np.random.RandomState(seed)
    positions = positions.copy()
    n = len(positions)
    
    current_score = _score(positions)
    best_score = current_score
    best_positions = positions.copy()
    
    # Annealing parameters
    T = 0.1
    T_min = 0.001
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        # Propose a move: perturb a random subset of positions
        num_moves = rng.randint(1, 4)
        indices = rng.choice(n, size=num_moves, replace=False)
        
        old_positions = positions[indices].copy()
        # Gaussian perturbation
        positions[indices] += rng.normal(0, T, size=(num_moves, 2))
        
        # Keep inside container
        radii = np.hypot(positions[:, 0], positions[:, 1])
        max_allowed = 1.0 - current_score * 0.99
        mask = radii > max_allowed
        if np.any(mask):
            positions[mask] *= max_allowed / radii[mask]
        
        new_score = _score(positions)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_positions = positions.copy()
        else:
            positions[indices] = old_positions
        
        T *= cooling_rate
    
    return best_positions, best_score


def _local_refine(positions, max_time=30.0):
    """Local refinement using scipy's L-BFGS-B."""
    positions = positions.copy()
    n = len(positions)
    
    # Flatten for scipy
    x0 = positions.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 2)
        # Penalty for going outside
        radii = np.hypot(pos[:, 0], pos[:, 1])
        penalty = np.sum(np.maximum(0, radii - (1.0 - 0.1)) ** 2) * 10
        return -_score(pos) + penalty
    
    # Bounds: keep inside [-1, 1] in each dimension
    bounds = [(-0.99, 0.99)] * (2 * n)
    
    start_time = time.time()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'maxfun': 10000,
                                   'ftol': 1e-10, 'gtol': 1e-8})
        
        refined_pos = result.x.reshape(-1, 2)
        
        # Ensure inside container
        radii = np.hypot(refined_pos[:, 0], refined_pos[:, 1])
        max_allowed = 1.0 - _score(refined_pos) * 0.999
        mask = radii > max_allowed
        if np.any(mask):
            refined_pos[mask] *= max_allowed / radii[mask]
        
        return refined_pos
    except Exception:
        return positions


def construct():
    """Construct positions for N circles in the unit disk."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_positions = None
    best_score = 0.0
    
    # Try multiple initialization strategies
    inits = [
        ("concentric", _concentric_ring_init(N)),
        ("hex", _hex_init(N)),
        ("random1", _random_init(N)),
        ("random2", _random_init(N)),
    ]
    
    for name, init_pos in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Simulated annealing
        sa_pos, sa_score = _simulated_annealing(init_pos, max_time=25.0, seed=hash(name) % 10000)
        
        if sa_score > best_score:
            best_score = sa_score
            best_positions = sa_pos.copy()
        
        # Local refinement if time permits
        if time.time() < deadline - 15.0:
            refined_pos = _local_refine(sa_pos, max_time=15.0)
            refined_score = _score(refined_pos)
            if refined_score > best_score:
                best_score = refined_score
                best_positions = refined_pos.copy()
    
    # Final validation
    if best_positions is None:
        best_positions = _concentric_ring_init(N)
    
    radii = np.hypot(best_positions[:, 0], best_positions[:, 1])
    max_allowed = 1.0 - _score(best_positions) * 0.999
    mask = radii > max_allowed
    if np.any(mask):
        best_positions[mask] *= max_allowed / radii[mask]
    
    return best_positions
# EVOLVE-BLOCK-END
