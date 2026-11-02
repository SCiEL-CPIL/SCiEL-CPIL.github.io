"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_initial_grid(nx, ny, nz, noise_scale=0.005, seed=42):
    """Build a grid initialization with small noise to break symmetry."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add small noise to break perfect symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init(n=100, noise_scale=0.005, seed=42):
    """
    Build a BCC (Body-Centered Cubic) inspired initialization.
    BCC is known to be a good lattice for sphere packing.
    """
    # Create a 5x5x4 BCC lattice
    nx, ny, nz = 5, 5, 4
    margin = 0.03
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    all_centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corner points
                all_centers.append([xs[i], ys[j], zs[k]])
                # Body center
                if i < nx - 1 and j < ny - 1 and k < nz - 1:
                    all_centers.append([(xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2, (zs[k] + zs[k + 1]) / 2])
    
    all_centers = np.array(all_centers)
    rng = np.random.default_rng(seed)
    
    if len(all_centers) >= N:
        idx = rng.choice(len(all_centers), N, replace=False)
        centers = all_centers[idx]
    else:
        centers = all_centers
        while len(centers) < N:
            centers = np.vstack([centers, centers[0]])
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n=100, seed=42):
    """
    Build a Halton sequence based initialization for good space-filling properties.
    """
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n + 1):
        centers[i - 1, 0] = -0.5 + halton(i, 2)
        centers[i - 1, 1] = -0.5 + halton(i, 3)
        centers[i - 1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=8):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    # Create beta schedule: geometric progression from beta_start to beta_end
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() > deadline - 10.0:  # Leave time for polish
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            # Check if we improved
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 15.0:
            break
    
    return best_centers, best_radius

def _anneal(centers, deadline, T_start=0.005, T_end=0.00005, n_steps=3000):
    """
    Simulated annealing to explore the configuration space.
    Uses Metropolis-Hastings with energy = -radius (we want to maximize radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_centers = centers.copy()
    
    rng = np.random.default_rng(42)
    
    n_done = 0
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Annealing schedule
        progress = step / max(1, n_steps - 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Random perturbation - move a random point
        idx = rng.integers(N)
        delta = rng.normal(0, 0.008, 3)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
        
        # Compute new radius
        new_radius = _compute_radius(new_centers)
        
        # Metropolis criterion
        if new_radius >= current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            delta_E = current_radius - new_radius  # positive means worse
            if delta_E < 1e-15:
                accept = True
            else:
                accept = np.log(rng.uniform()) > -delta_E / T
            if accept:
                current_centers = new_centers
                current_radius = new_radius
        
        n_done += 1
    
    return best_centers, best_radius

def _polish_high_beta(centers, deadline):
    """
    Final high-precision polish using very high beta L-BFGS-B.
    This tightens the solution by focusing on the most constraining pairs.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    iu = np.triu_indices(N, k=1)
    
    # Use very high beta values for polishing
    betas = [10000.0, 50000.0, 100000.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    # Use 145s deadline to leave safety margin for the 150s limit
    deadline = start_time + 145.0
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Try multiple different initializations
    initializers = [
        ("grid_554", lambda: _build_initial_grid(5, 5, 4, 0.005, 42)),
        ("grid_455", lambda: _build_initial_grid(4, 5, 5, 0.005, 42)),
        ("grid_545", lambda: _build_initial_grid(5, 4, 5, 0.005, 42)),
        ("bcc", lambda: _build_bcc_init(100, 0.005, 42)),
        ("halton", lambda: _build_halton_init(100, 42)),
    ]
    
    for name, init_func in initializers:
        if time.time() > deadline - 60.0:
            break
        
        centers = init_func()
        r_init = _compute_radius(centers)
        
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
        
        # Optimize from this initial
        opt_centers, opt_radius = _optimize_from_initial(
            centers, deadline,
            beta_start=50.0, beta_end=5000.0, n_beta_steps=6
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # Phase 2: Simulated annealing from best solution
    if best_centers is not None and time.time() < deadline - 20.0:
        anneal_deadline = deadline - 10.0
        annealed_centers, anneal_radius = _anneal(
            best_centers, anneal_deadline,
            T_start=0.005, T_end=0.00005, n_steps=3000
        )
        
        if anneal_radius > best_radius:
            best_radius = anneal_radius
            best_centers = annealed_centers
    
    # Phase 3: Final high-beta polish
    if best_centers is not None and time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        polished_centers, polish_radius = _polish_high_beta(best_centers, polish_deadline)
        
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polished_centers
    
    # If no valid result, fallback to simple grid
    if best_centers is None:
        best_centers = _build_initial_grid(5, 5, 4, 0.01)
    
    return best_centers
# EVOLVE-BLOCK-END
