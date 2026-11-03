"""Packing 50 equal circles in a unit circle using rigid-body dynamics."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(points):
    """Compute the packing radius for a set of points."""
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside the unit circle
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii > 1.0):
        return 0.0
    # Compute minimum distance between pairs
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair_dist:
                min_pair_dist = d
    min_container = 1.0 - np.max(radii)
    return min(min_pair_dist / 2.0, min_container)


def _hexagonal_init(n_points, seed=42):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    # Start with a dense hexagonal lattice
    spacing = 0.15
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    points = np.array(points)
    # Add small random perturbation
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    # Sort by radius and take first n_points
    radii = np.linalg.norm(points, axis=1)
    idx = np.argsort(radii)[:n_points]
    return points[idx].copy()


def _random_init(n_points, seed=0):
    """Generate a random initialization inside the unit circle."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, size=n_points)
    radii = rng.uniform(0, 1, size=n_points) ** 0.5
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _soft_min_objective(points, beta=100.0):
    """
    Soft-min log-sum-exp objective for the packing radius.
    We want to maximize r = min(min_pair_dist/2, min_container).
    Equivalently, minimize -r.
    
    Using log-sum-exp: -min(a_i) ≈ (1/beta) * log(sum(exp(-beta * a_i)))
    """
    points = points.reshape(-1, 2)
    n = points.shape[0]
    
    # Pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = np.linalg.norm(points[i] - points[j])
    
    # Container distances
    radii = np.linalg.norm(points, axis=1)
    container_dists = 1.0 - radii
    
    # All "distances" that constrain the radius
    # Pair distances contribute dist/2, container contributes (1-|c|)
    all_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            all_dists.append(dists[i, j] / 2.0)
        all_dists.append(container_dists[i])
    all_dists = np.array(all_dists)
    
    # Soft min via log-sum-exp
    # -min(x) ≈ (1/beta) * log(sum(exp(-beta*x)))
    # To avoid overflow, use max trick
    max_val = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - max_val))
    soft_min = max_val - (1.0 / beta) * np.log(np.sum(exp_terms))
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min


def _soft_min_gradient(points, beta=100.0):
    """Compute gradient of the soft-min objective."""
    points = points.reshape(-1, 2)
    n = points.shape[0]
    
    # Pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = np.linalg.norm(points[i] - points[j])
    
    radii = np.linalg.norm(points, axis=1)
    container_dists = 1.0 - radii
    
    all_dists = []
    dist_indices = []  # (type, i, j) where type=0 for pair, 1 for container
    for i in range(n):
        for j in range(i + 1, n):
            all_dists.append(dists[i, j] / 2.0)
            dist_indices.append((0, i, j))
        all_dists.append(container_dists[i])
        dist_indices.append((1, i, -1))
    all_dists = np.array(all_dists)
    
    max_val = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - max_val))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(points)
    
    for idx, (typ, i, j) in enumerate(dist_indices):
        w = weights[idx]
        if typ == 0:  # pair distance
            diff = points[i] - points[j]
            dist = dists[i, j]
            if dist > 1e-12:
                # d/dp_i (dist/2) = diff / (2*dist) * ... 
                # Actually d(dist)/dp_i = diff/dist
                # d(dist/2)/dp_i = diff/(2*dist)
                # Objective is -soft_min, so grad = -w * beta * d(soft_min)/dp
                # d(soft_min)/dp_i = -d(dist/2)/dp_i = -diff/(2*dist)
                # So grad contribution = -w * beta * (-diff/(2*dist)) = w*beta*diff/(2*dist)
                grad[i] += w * beta * diff / (2.0 * dist)
                grad[j] -= w * beta * diff / (2.0 * dist)
        else:  # container
            r = radii[i]
            if r > 1e-12:
                # d(1-r)/dp_i = -p_i/r
                # d(soft_min)/dp_i = -d(1-r)/dp_i = p_i/r
                # grad contribution = -w * beta * (p_i/r)
                grad[i] -= w * beta * points[i] / r
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initializations with different strategies
    initializations = []
    
    # Hexagonal with different seeds
    for seed in [42, 123, 7, 99, 555]:
        initializations.append(_hexagonal_init(N, seed=seed))
    
    # Random initializations
    for seed in [0, 1, 2, 3, 4]:
        initializations.append(_random_init(N, seed=seed))
    
    # Ring-based initialization: concentric rings
    for seed in [10, 20, 30]:
        rng = np.random.RandomState(seed)
        points = []
        # Center
        points.append([0.0, 0.0])
        # Rings
        ring_radii = [0.25, 0.5, 0.75, 0.95]
        counts = [6, 12, 18, 13]
        for r, count in zip(ring_radii, counts):
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            angles += rng.uniform(0, 2 * np.pi / count, size=count)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
        points = np.array(points[:N])
        points += rng.uniform(-0.02, 0.02, size=points.shape)
        initializations.append(points)
    
    # Try each initialization with L-BFGS-B optimization
    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Multi-stage beta annealing
        betas = [10.0, 50.0, 100.0, 200.0, 500.0]
        
        current_points = init_points.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    _soft_min_objective,
                    current_points,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_soft_min_gradient,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_points = result.x
            except Exception:
                pass
            
            # Evaluate actual score
            score = _score(current_points)
            if score > best_score:
                best_score = score
                best_points = current_points.copy()
    
    # If we still have time, do a final refinement with a very high beta
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            result = minimize(
                _soft_min_objective,
                best_points,
                args=(1000.0,),
                method='L-BFGS-B',
                jac=_soft_min_gradient,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            refined = result.x
            score = _score(refined)
            if score > best_score:
                best_points = refined
                best_score = score
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to hexagonal
        best_points = _hexagonal_init(N, seed=42)
    
    return best_points
# EVOLVE-BLOCK-END
