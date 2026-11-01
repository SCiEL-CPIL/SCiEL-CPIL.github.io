"""Pack 50 equal circles in the unit circle using force-directed + soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import cdist

N = 50

def _compute_score(centers):
    """Compute the packing score (largest common radius)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    dists = np.sqrt(np.sum(centers**2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary < 0:
        return 0.0
    pairwise = cdist(centers, centers)
    np.fill_diagonal(pairwise, np.inf)
    min_pair = np.min(pairwise)
    score = min(min_boundary, min_pair / 2.0)
    return max(score, 0.0)

def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return 0.0
    m = np.min(v)
    return m - np.log(np.sum(np.exp(beta * (v - m)))) / beta



def _local_search_maximin(centers, time_limit=10.0, rng_seed=42):
    """
    Local search to maximize the minimum distance (and boundary distance).
    Uses random perturbations (Hill Climbing with kicks).
    """
    start_time = time.time()
    centers = centers.copy()
    current_score = _compute_score(centers)
    rng = np.random.RandomState(rng_seed)
    
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit:
        # Pick a random circle
        i = rng.randint(N)
        
        # Random direction
        dir_vec = rng.randn(2)
        dir_vec /= np.linalg.norm(dir_vec)
        
        # Try a few step sizes
        found_improvement = False
        for s in [step_size, step_size * 0.5, step_size * 2.0]:
            new_centers = centers.copy()
            new_centers[i] += s * dir_vec
            
            # Check bounds
            dist = np.linalg.norm(new_centers[i])
            if dist > 1.0:
                # Push back inside
                new_centers[i] = new_centers[i] * (1.0 / dist)
            
            new_score = _compute_score(new_centers)
            
            if new_score > current_score:
                centers = new_centers
                current_score = new_score
                no_improve_count = 0
                found_improvement = True
                break
        if not found_improvement:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reduce step size or kick
                step_size *= 0.9
                if step_size < 1e-5:
                    # Try a kick
                    for _ in range(5):
                        k = rng.randint(N)
                        kick_dir = rng.randn(2)
                        kick_dir /= np.linalg.norm(kick_dir)
                        new_centers = centers.copy()
                        new_centers[k] += 0.05 * kick_dir
                        if np.linalg.norm(new_centers[k]) > 1.0:
                            new_centers[k] = new_centers[k] * (1.0 / np.linalg.norm(new_centers[k]))
                        if _compute_score(new_centers) > current_score:
                            centers = new_centers
                            current_score = _compute_score(centers)
                            step_size = 0.01
                            no_improve_count = 0
                            break
                    else:
                        # If no kick worked, reset step size and continue with smaller steps
                        step_size = 0.01
                        no_improve_count = 0
                else:
                    no_improve_count = 0

    return centers, current_score

def _polish(centers, time_limit=5.0):
    """Very fine local search."""
    start_time = time.time()
    centers = centers.copy()
    current_score = _compute_score(centers)
    rng = np.random.RandomState(12345)
    step_size = 1e-4
    
    while time.time() - start_time < time_limit:
        i = rng.randint(N)
        dir_vec = rng.randn(2)
        dir_vec /= np.linalg.norm(dir_vec)
        
        new_centers = centers.copy()
        new_centers[i] += step_size * dir_vec
        
        if np.linalg.norm(new_centers[i]) > 1.0:
            new_centers[i] = new_centers[i] * (1.0 / np.linalg.norm(new_centers[i]))
            
        new_score = _compute_score(new_centers)
        if new_score > current_score:
            centers = new_centers
            current_score = new_score
    return centers, current_score

def _get_hex_lattice(n, radius, angle=0.0):
    """Generate hexagonal lattice points within unit circle."""
    spacing = 2 * radius
    sqrt3 = np.sqrt(3)
    points = []
    max_k = int(np.ceil(1.0 / spacing)) + 3
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            if np.sqrt(x**2 + y**2) <= 1.0 - radius + 1e-9:
                points.append([x, y])
    if abs(angle) > 1e-9:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        points = [np.dot(rot, p) for p in points]
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    return np.array(points[:n])

def _get_ring_initialization(n, radii=None):
    """Initialize with concentric rings."""
    if radii is None:
        # Approximate optimal ring radii for 50 circles
        # Ring 0: 1 circle at center
        # Ring 1: 6 circles
        # Ring 2: 12 circles
        # Ring 3: 18 circles
        # Ring 4: 13 circles
        # Total: 50
        rings = [1, 6, 12, 18, 13]
        ring_radii = [0.0, 0.25, 0.50, 0.75, 0.90]
    else:
        rings = [1, 6, 12, 18, 13]
        ring_radii = radii
    
    points = []
    for ring_idx, (count, r) in enumerate(zip(rings, ring_radii)):
        if count == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
    
    return np.array(points[:n])

def _get_fibonacci_initialization(n):
    """Initialize using Fibonacci spiral."""
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        r = 0.95 * np.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

def _get_random_initialization(n, seed):
    """Random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(n):
        r = np.sqrt(rng.random()) * 0.95
        theta = rng.random() * 2 * np.pi
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)



def construct():
    """Construct the best packing found within the time budget."""
    deadline = time.time() + 130.0  # Safety margin from 150s limit
    
    best_centers = None
    best_score = 0.0
    
    # Generate multiple initializations
    initializations = []
    
    # Hexagonal lattice with various angles
    for angle in [0.0, np.pi/6, np.pi/3, np.pi/4, np.pi/2, 2*np.pi/3]:
        for radius in [0.12, 0.125, 0.13]:
            try:
                init = _get_hex_lattice(N, radius, angle)
                if len(init) == N:
                    initializations.append(init)
            except Exception:
                pass
    
    # Ring initialization
    initializations.append(_get_ring_initialization(N))
    
    # Fibonacci spiral
    initializations.append(_get_fibonacci_initialization(N))
    
    # Random initializations
    for seed in [42, 123, 456, 789, 1000, 2000]:
        initializations.append(_get_random_initialization(N, seed))
    
    # Process each initialization
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # Local search
        try:
            optimized, opt_score = _local_search_maximin(init, time_limit=5.0, rng_seed=init_idx+1)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = optimized
        except Exception:
            pass
    
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 5.0:
        try:
            remaining = min(10.0, deadline - time.time() - 2.0)
            polished, pol_score = _polish(best_centers, time_limit=remaining)
            if pol_score > best_score:
                best_score = pol_score
                best_centers = polished
        except Exception:
            pass
    
    if best_centers is None:
        best_centers = _get_hex_lattice(N, 0.12)
    
    return best_centers
