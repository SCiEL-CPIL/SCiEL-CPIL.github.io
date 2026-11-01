"""Pack 50 equal circles in the unit circle using Newton-Raphson active-set optimization."""
import numpy as np
import time
from scipy.optimize import minimize
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

def _get_active_contacts(centers, r, tol=1e-6):
    """
    Identify active constraints: pairs of circles in contact and circles touching the boundary.
    Returns (pair_contacts, boundary_contacts) where each is a list of indices.
    """
    n = len(centers)
    pair_contacts = []
    boundary_contacts = []
    
    for i in range(n):
        # Boundary contact
        dist_to_origin = np.linalg.norm(centers[i])
        if abs(dist_to_origin + r - 1.0) < tol:
            boundary_contacts.append(i)
        
        # Pair contacts
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if abs(d - 2 * r) < tol:
                pair_contacts.append((i, j))
    
    return pair_contacts, boundary_contacts

def _newton_raphson_packing(centers, r, max_iter=100, tol=1e-10):
    """
    Newton-Raphson solver for the active-set problem.
    Given an initial configuration and radius, solve for exact positions where
    all active contacts are satisfied exactly.
    
    The system of equations is:
    - For each pair contact (i,j): |c_i - c_j| = 2r
    - For each boundary contact i: |c_i| = 1 - r
    
    We solve this by minimizing the residual of these constraints.
    """
    centers = centers.copy()
    n = len(centers)
    
    # Get active contacts
    pair_contacts, boundary_contacts = _get_active_contacts(centers, r, tol=1e-4)
    
    if len(pair_contacts) == 0 and len(boundary_contacts) == 0:
        return centers, r, 0.0
    
    # Build the constraint system
    # Variables: 2n coordinates + 1 radius = 2n+1 variables
    # Constraints: len(pair_contacts) + len(boundary_contacts) equations
    
    def residual(x):
        """Residual of the constraint system."""
        coords = x[:2*n].reshape(n, 2)
        r_val = x[2*n]
        res = []
        
        for (i, j) in pair_contacts:
            d = np.linalg.norm(coords[i] - coords[j])
            res.append(d - 2 * r_val)
        
        for i in boundary_contacts:
            d = np.linalg.norm(coords[i])
            res.append(d - (1 - r_val))
        
        return np.array(res)
    
    def jacobian(x):
        """Jacobian of the residual."""
        coords = x[:2*n].reshape(n, 2)
        r_val = x[2*n]
        m = len(residual(x))
        J = np.zeros((m, 2*n + 1))
        
        row = 0
        for (i, j) in pair_contacts:
            d_vec = coords[i] - coords[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-12:
                d = 1e-12
            # d/dx_i |c_i - c_j| = (c_i - c_j) / |c_i - c_j|
            J[row, 2*i:2*i+2] = d_vec / d
            # d/dx_j |c_i - c_j| = -(c_i - c_j) / |c_i - c_j|
            J[row, 2*j:2*j+2] = -d_vec / d
            # d/dr |c_i - c_j| - 2r = -2
            J[row, 2*n] = -2.0
            row += 1
        
        for i in boundary_contacts:
            c = coords[i]
            d = np.linalg.norm(c)
            if d < 1e-12:
                d = 1e-12
            # d/dx_i |c_i| = c_i / |c_i|
            J[row, 2*i:2*i+2] = c / d
            # d/dr |c_i| - (1-r) = +1
            J[row, 2*n] = 1.0
            row += 1
        
        return J
    
    # Initial guess
    x0 = np.zeros(2*n + 1)
    x0[:2*n] = centers.flatten()
    x0[2*n] = r
    
    # Newton iterations
    for it in range(max_iter):
        res = residual(x0)
        if np.max(np.abs(res)) < tol:
            break
        
        J = jacobian(x0)
        
        # Solve J dx = -res
        try:
            # Use least squares if overdetermined
            if J.shape[0] >= J.shape[1]:
                dx, residuals, rank, sv = np.linalg.lstsq(J, -res, rcond=None)
            else:
                dx, residuals, rank, sv = np.linalg.lstsq(J, -res, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        # Damping
        alpha = 1.0
        for _ in range(20):
            x_new = x0 + alpha * dx
            # Ensure radius is positive
            if x_new[2*n] <= 0:
                alpha *= 0.5
                continue
            res_new = residual(x_new)
            if np.max(np.abs(res_new)) < np.max(np.abs(res)) * 0.9:
                x0 = x_new
                break
            alpha *= 0.5
        
        if alpha < 1e-10:
            break
    
    # Extract final positions
    final_centers = x0[:2*n].reshape(n, 2)
    final_r = x0[2*n]
    
    # Verify and adjust
    final_score = _compute_score(final_centers)
    
    return final_centers, final_r, final_score

def _soft_min_optimize(centers, time_limit=5.0, beta_start=50.0, beta_end=500.0):
    """
    Optimize using soft-min surrogate with L-BFGS.
    This handles the case where the active set is not well-defined.
    """
    start_time = time.time()
    centers = centers.copy()
    
    def objective_and_grad(x):
        coords = x.reshape(N, 2)
        
        # Pairwise distances
        pairwise = cdist(coords, coords)
        np.fill_diagonal(pairwise, np.inf)
        
        # Boundary distances
        boundary = 1.0 - np.linalg.norm(coords, axis=1)
        
        # Soft min of all distances/2 and boundary
        all_vals = np.concatenate([pairwise.flatten() / 2.0, boundary])
        
        # Adaptive beta
        elapsed = time.time() - start_time
        progress = min(elapsed / time_limit, 1.0)
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Soft min
        m = np.min(all_vals)
        soft_min_val = m - np.log(np.sum(np.exp(beta * (all_vals - m)))) / beta
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Boundary gradient
        norms = np.linalg.norm(coords, axis=1)
        norms_safe = np.where(norms > 1e-12, norms, 1e-12)
        boundary_grad = -coords / norms_safe[:, np.newaxis]
        
        # Weight for boundary
        exp_boundary = np.exp(beta * (boundary - m))
        w_boundary = exp_boundary / np.sum(np.exp(beta * (all_vals - m)))
        
        grad[:N*2] = (grad[:N*2].reshape(N, 2) + np.sum(w_boundary, axis=1)[:, np.newaxis] * boundary_grad).flatten()
        
        # Pairwise gradient
        exp_pair = np.exp(beta * (pairwise / 2.0 - m))
        w_pair = exp_pair / np.sum(np.exp(beta * (all_vals - m)))
        
        # For each pair (i,j), contribution to grad
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d_vec = coords[i] - coords[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-12:
                    continue
                grad[i*2:i*2+2] += w_pair[i,j] * d_vec / d
                grad[j*2:j*2+2] -= w_pair[i,j] * d_vec / d
        
        return soft_min_val, grad
    
    # Optimize
    result = minimize(objective_and_grad, centers.flatten(), 
                      method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
    
    optimized = result.x.reshape(N, 2)
    score = _compute_score(optimized)
    
    return optimized, score

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
    """Very fine local search with Newton-Raphson active-set refinement."""
    start_time = time.time()
    centers = centers.copy()
    current_score = _compute_score(centers)
    rng = np.random.RandomState(12345)
    step_size = 1e-4
    
    # First, try Newton-Raphson on the current active set
    r = current_score
    try:
        new_centers, new_r, new_score = _newton_raphson_packing(centers, r, max_iter=50, tol=1e-12)
        if new_score > current_score:
            centers = new_centers
            current_score = new_score
            r = new_score
    except Exception:
        pass
    
    # Then do fine local search
    while time.time() - start_time < time_limit:
        # Try Newton-Raphson refinement
        if rng.random() < 0.3:
            try:
                new_centers, new_r, new_score = _newton_raphson_packing(centers, current_score, max_iter=20, tol=1e-12)
                if new_score > current_score:
                    centers = new_centers
                    current_score = new_score
                    continue
            except Exception:
                pass
        
        # Random perturbation
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

def _get_hex_lattice_optimal_radius(n):
    """
    Find the optimal radius for a hexagonal lattice packing of n circles.
    Uses binary search on the radius.
    """
    def can_pack(radius):
        spacing = 2 * radius
        sqrt3 = np.sqrt(3)
        count = 0
        max_k = int(np.ceil(1.0 / spacing)) + 3
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * sqrt3 / 2.0
                if np.sqrt(x**2 + y**2) <= 1.0 - radius + 1e-9:
                    count += 1
                    if count >= n:
                        return True
        return False
    
    # Binary search
    lo, hi = 0.01, 0.2
    for _ in range(60):
        mid = (lo + hi) / 2
        if can_pack(mid):
            lo = mid
        else:
            hi = mid
    
    return (lo + hi) / 2

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
    
    # Hexagonal lattice with various angles and optimal radius
    hex_radius = _get_hex_lattice_optimal_radius(N)
    for angle in [0.0, np.pi/6, np.pi/3, np.pi/4, np.pi/2, 2*np.pi/3]:
        for radius in [hex_radius * 0.95, hex_radius, hex_radius * 1.05]:
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
    
    # Process each initialization with multiple strategies
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        # Strategy 1: Soft-min optimization
        try:
            remaining = min(8.0, deadline - time.time() - 35.0)
            if remaining > 2.0:
                optimized, opt_score = _soft_min_optimize(init, time_limit=remaining)
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = optimized
        except Exception:
            pass
        
        # Strategy 2: Newton-Raphson on the soft-min result
        if best_centers is not None and best_score > 0.1:
            try:
                new_centers, new_r, new_score = _newton_raphson_packing(
                    best_centers, best_score, max_iter=50, tol=1e-12)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers
            except Exception:
                pass
        
        # Strategy 3: Local search if time permits
        if time.time() < deadline - 25.0:
            try:
                remaining = min(5.0, deadline - time.time() - 30.0)
                if remaining > 1.0:
                    optimized, opt_score = _local_search_maximin(init, time_limit=remaining, rng_seed=init_idx+1)
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
