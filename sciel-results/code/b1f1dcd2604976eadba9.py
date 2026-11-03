"""
Sphere packing: 100 equal spheres in a unit cube [-0.5, 0.5]^3.
Uses repulsive potential energy minimization with Lloyd-style iteration,
FCC lattice initialization, and multi-scale refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
CUBE_LIMIT = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        return float(np.min(CUBE_LIMIT - np.abs(centers)))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    min_wall_dist = np.min(CUBE_LIMIT - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _energy_and_gradient(centers, power=6, eps=1e-8):
    """
    Repulsive potential energy: E = sum_{i<j} 1/d^power + wall repulsion.
    Lower energy = more uniform spacing = larger minimum distance.
    """
    n = len(centers)
    grad = np.zeros_like(centers)
    energy = 0.0
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            d = centers[i] - centers[j]
            dist_sq = np.dot(d, d)
            if dist_sq < eps * eps:
                dist_sq = eps * eps
            dist = np.sqrt(dist_sq)
            inv_d_pow = dist ** (-power)
            energy += inv_d_pow
            # d/d(centers[i]) of 1/|c_i - c_j|^power = -power * (c_i - c_j) / |c_i - c_j|^(power+2)
            coeff = -power * inv_d_pow / dist_sq
            grad[i] += coeff * d
            grad[j] -= coeff * d
    
    # Wall repulsion: push away from walls
    for i in range(n):
        for k in range(3):
            wall_dist = CUBE_LIMIT - abs(centers[i, k])
            if wall_dist < 0.15:
                # Strong repulsion near walls
                rep = 1.0 / (wall_dist + eps) ** power
                energy += rep
                sign = 1.0 if centers[i, k] >= 0 else -1.0
                grad[i, k] += -power * rep / (wall_dist + eps) * sign
    
    return energy, grad

def _energy_and_gradient_fast(centers, power=6, eps=1e-8, cutoff=0.3):
    """
    Faster version using KD-tree to only compute interactions within cutoff.
    """
    n = len(centers)
    grad = np.zeros_like(centers)
    energy = 0.0
    
    tree = cKDTree(centers)
    pairs = list(tree.query_pairs(cutoff))
    
    for i, j in pairs:
        d = centers[i] - centers[j]
        dist_sq = np.dot(d, d)
        if dist_sq < eps * eps:
            dist_sq = eps * eps
        dist = np.sqrt(dist_sq)
        inv_d_pow = dist ** (-power)
        energy += inv_d_pow
        coeff = -power * inv_d_pow / dist_sq
        grad[i] += coeff * d
        grad[j] -= coeff * d
    
    # Wall repulsion
    for i in range(n):
        for k in range(3):
            wall_dist = CUBE_LIMIT - abs(centers[i, k])
            if wall_dist < 0.2:
                rep = 1.0 / (wall_dist + eps) ** power
                energy += rep
                sign = 1.0 if centers[i, k] >= 0 else -1.0
                grad[i, k] += -power * rep / (wall_dist + eps) * sign
    
    return energy, grad

def _lloyd_iteration(centers, power=6, eps=1e-8, max_iter=500, tol=1e-10, time_limit=10.0):
    """
    Lloyd-style iteration: at each point, compute the centroid of its Voronoi cell
    approximation (weighted by inverse distance), and move toward it.
    This is equivalent to minimizing the repulsive potential with adaptive step size.
    """
    start_time = time.time()
    centers = centers.copy()
    n = len(centers)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces (negative gradient of energy)
        energy, grad = _energy_and_gradient_fast(centers, power=power, eps=eps, cutoff=0.4)
        
        # Adaptive step size based on gradient magnitude
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Use a simple gradient descent step with damping
        step_size = 0.01 * (1.0 / (1.0 + grad_norm))
        
        new_centers = centers - step_size * grad
        
        # Project to cube
        new_centers = np.clip(new_centers, -CUBE_LIMIT + 1e-6, CUBE_LIMIT - 1e-6)
        
        # Check if radius improved
        old_radius = _compute_radius(centers)
        new_radius = _compute_radius(new_centers)
        
        if new_radius >= old_radius - 1e-12:
            centers = new_centers
        else:
            # Try smaller step
            step_size *= 0.1
            new_centers = centers - step_size * grad
            new_centers = np.clip(new_centers, -CUBE_LIMIT + 1e-6, CUBE_LIMIT - 1e-6)
            new_radius = _compute_radius(new_centers)
            if new_radius >= old_radius - 1e-12:
                centers = new_centers
            else:
                break
    
    return centers

def _lbfgs_minimize(centers, power=6, max_iter=200, time_limit=10.0):
    """
    Use L-BFGS-B to minimize the repulsive energy.
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    def obj_grad(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        c = x.reshape(N, 3)
        e, g = _energy_and_gradient_fast(c, power=power, eps=1e-8, cutoff=0.4)
        if not np.all(np.isfinite(g)):
            return 1e10, np.zeros_like(x)
        return e, g
    
    try:
        result = minimize(
            obj_grad, x0,
            method='L-BFGS-B',
            bounds=[(-CUBE_LIMIT, CUBE_LIMIT)] * (N * 3),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
        )
        if result.success or _compute_radius(result.x.reshape(N, 3)) > _compute_radius(centers):
            return result.x.reshape(N, 3)
    except Exception:
        pass
    
    return centers

def _fcc_init(n):
    """
    Generate FCC lattice initialization clipped to the cube.
    FCC has 4 atoms per unit cell, giving higher density than simple cubic.
    """
    # Generate FCC lattice points in a larger region
    a = 0.25  # lattice parameter
    points = []
    
    # Simple cubic sublattice
    range_val = int(np.ceil(1.0 / a)) + 2
    for ix in range(-range_val, range_val + 1):
        for iy in range(-range_val, range_val + 1):
            for iz in range(-range_val, range_val + 1):
                # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) in units of a
                for bx, by, bz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (ix + bx) * a
                    y = (iy + by) * a
                    z = (iz + bz) * a
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        points.append([x, y, z])
    
    points = np.array(points)
    
    # If we have more than n points, select the n with largest minimum distance
    if len(points) > n:
        # Use greedy selection
        selected = [0]  # Start with center
        while len(selected) < n:
            if len(points) <= 1:
                break
            # Compute min distance from each candidate to selected points
            min_dists = np.full(len(points), np.inf)
            for idx in selected:
                d = np.linalg.norm(points - points[idx], axis=1)
                min_dists = np.minimum(min_dists, d)
            
            # Mask already selected
            for idx in selected:
                min_dists[idx] = -1
            
            best_idx = np.argmax(min_dists)
            if min_dists[best_idx] <= 1e-10:
                break
            selected.append(best_idx)
        
        points = points[selected]
    elif len(points) < n:
        # Pad with random points if needed
        rng = np.random.RandomState(42)
        extra = rng.uniform(-0.45, 0.45, size=(n - len(points), 3))
        points = np.vstack([points, extra])
    
    return points[:n]

def _jitter_init(n, seed=0, jitter=0.015):
    """Jittered FCC initialization."""
    base = _fcc_init(n)
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-jitter, jitter, size=base.shape)
    return np.clip(base + noise, -0.49, 0.49)

def _random_init(n, seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _polish_critical_pairs(centers, time_limit=5.0):
    """
    Final polish: identify critical constraints and do local optimization.
    """
    start_time = time.time()
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    for _ in range(30):
        if time.time() - start_time > time_limit:
            break
        
        # Find all pairs within 2.1 * best_radius
        r = best_radius
        tree = cKDTree(centers)
        pairs = list(tree.query_pairs(2.2 * r + 0.01))
        
        if len(pairs) == 0:
            break
        
        improved = False
        for i, j in pairs[:30]:
            if time.time() - start_time > time_limit:
                break
            
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                continue
            
            direction = d / dist
            
            # Try moving apart
            for step in [0.003, 0.001, 0.0005]:
                new_centers = centers.copy()
                new_centers[i] += step * direction
                new_centers[j] -= step * direction
                new_centers = np.clip(new_centers, -0.499999, 0.499999)
                
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius + 1e-12:
                    centers = new_centers
                    best_radius = new_radius
                    improved = True
                    break
        
        if not improved:
            break
    
    return centers

def _anneal_restart(centers, time_limit=10.0):
    """
    Simulated annealing with restarts to escape local minima.
    Uses the energy function for acceptance criterion.
    """
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    rng = np.random.RandomState(7)
    
    T0 = 0.03
    Tend = 0.0005
    n_steps = 1500
    cooling_rate = (Tend / T0) ** (1.0 / n_steps)
    
    T = T0
    current_energy, _ = _energy_and_gradient_fast(centers, power=6, eps=1e-8, cutoff=0.4)
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randint(N)
        disp = rng.normal(0, 0.015, size=3)
        new_centers = centers.copy()
        new_centers[i] += disp
        new_centers = np.clip(new_centers, -0.499999, 0.499999)
        
        new_radius = _compute_radius(new_centers)
        delta_r = new_radius - _compute_radius(centers)
        
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            centers = new_centers
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        
        T *= cooling_rate
    
    return best_centers

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple initializations with FCC-based starts
    inits = []
    
    # 1. FCC lattice
    inits.append(_fcc_init(N))
    
    # 2. Jittered FCC
    for seed in range(6):
        inits.append(_jitter_init(N, seed=seed, jitter=0.015 + 0.01 * seed))
    
    # 3. Random inits
    for seed in range(4):
        inits.append(_random_init(N, seed=seed))
    
    # Optimize each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        try:
            time_remaining = deadline - time.time() - 8.0
            if time_remaining < 5.0:
                break
            
            centers = init.copy()
            
            # Phase 1: Lloyd-style iteration (fast, many iterations)
            lloyd_time = min(8.0, time_remaining * 0.4)
            centers = _lloyd_iteration(centers, power=6, max_iter=300, time_limit=lloyd_time)
            r_lloyd = _compute_radius(centers)
            
            # Phase 2: L-BFGS on energy (more precise)
            if time.time() < deadline - 15.0:
                lbfgs_time = min(8.0, deadline - time.time() - 12.0)
                centers = _lbfgs_minimize(centers, power=6, max_iter=150, time_limit=lbfgs_time)
                r_lbfgs = _compute_radius(centers)
                if r_lbfgs > r_lloyd:
                    r_lloyd = r_lbfgs
            
            # Phase 3: Annealing to escape local minima
            if time.time() < deadline - 10.0:
                sa_time = min(8.0, deadline - time.time() - 8.0)
                sa_centers = _anneal_restart(centers, time_limit=sa_time)
                r_sa = _compute_radius(sa_centers)
                if r_sa > r_lloyd:
                    centers = sa_centers
                    r_lloyd = r_sa
            
            # Phase 4: Final polish
            if time.time() < deadline - 5.0:
                polish_time = min(4.0, deadline - time.time() - 4.0)
                polished = _polish_critical_pairs(centers, time_limit=polish_time)
                r_polished = _compute_radius(polished)
                if r_polished > r_lloyd:
                    r_lloyd = r_polished
                    centers = polished
            
            if r_lloyd > best_radius:
                best_radius = r_lloyd
                best_centers = centers.copy()
                
        except Exception:
            continue
    
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 3.0:
        try:
            polish_time = min(4.0, deadline - time.time() - 2.0)
            polished = _polish_critical_pairs(best_centers, time_limit=polish_time)
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _fcc_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
