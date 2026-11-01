"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
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

def _build_initial_grid(nx, ny, nz, noise_scale=0.005):
    """Build a grid initialization with small noise to break symmetry."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init(n=10, noise_scale=0.005):
    """
    Build a BCC (Body-Centered Cubic) inspired initialization.
    BCC is known to be a good lattice for sphere packing.
    We create a 10x10x1 grid of BCC unit cells, giving 2*10*10*1 = 200 points,
    then randomly select 100.
    """
    # BCC unit cell has points at corners and center
    # We'll create a grid of BCC cells
    nx, ny, nz = 10, 10, 1
    margin = 0.02
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
                if i < nx-1 and j < ny-1 and k < nz-1:
                    all_centers.append([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, (zs[k]+zs[k+1])/2])
                elif i < nx-1 and j < ny-1:
                    all_centers.append([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, zs[k]])
                elif i < nx-1:
                    all_centers.append([(xs[i]+xs[i+1])/2, ys[j], zs[k]])
                elif j < ny-1:
                    all_centers.append([xs[i], (ys[j]+ys[j+1])/2, zs[k]])
    
    all_centers = np.array(all_centers)
    rng = np.random.default_rng(42)
    
    if len(all_centers) >= N:
        idx = rng.choice(len(all_centers), N, replace=False)
        centers = all_centers[idx]
    else:
        centers = all_centers
        # Pad if needed
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
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _anneal(centers, deadline, T_start=0.01, T_end=0.0001, n_steps=5000):
    """
    Simulated annealing to explore the configuration space.
    Uses Metropolis-Hastings with energy = -radius (we want to maximize radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_centers = centers.copy()
    
    iu = np.triu_indices(N, k=1)
    
    # Precompute wall distances for efficiency
    rng = np.random.default_rng(42)
    
    n_done = 0
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Annealing schedule
        progress = step / max(1, n_steps - 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Random perturbation
        idx = rng.integers(N)
        delta = rng.normal(0, 0.01, 3)
        
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

def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=6):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
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
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final high-precision polish pass using tighter tolerances.
    """
    iu = np.triu_indices(N, k=1)
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Use very high beta for precise optimization
    betas = [20000.0, 50000.0, 100000.0]
    
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
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
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
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -1.0
    best_points = None
    
    # Multiple initializations
    inits = [
        _build_initial_grid(5, 5, 4),
        _build_bcc_init(),
        _build_halton_init(),
        _build_initial_grid(4, 5, 5),
        _build_initial_grid(5, 4, 5),
    ]
    
    # For each initialization, run a sequence of:
    # 1. Annealing (optional, if time permits)
    # 2. Optimization with increasing beta
    # 3. Polish
    
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        current_centers = init_centers.copy()
        
        # Annealing phase (shorter for later inits)
        if time.time() < deadline - 30.0:
            n_anneal_steps = 2000 if i == 0 else 1000
            current_centers, _ = _anneal(current_centers, deadline - 20.0, 
                                         n_steps=n_anneal_steps)
        
        # Optimization phase
        remaining_deadline = deadline - 10.0
        current_centers, r = _optimize_from_initial(current_centers, remaining_deadline)
        
        if r > best_score:
            best_score = r
            best_points = current_centers.copy()
        
        # Polish phase
        if time.time() < deadline - 5.0:
            current_centers, r = _polish(current_centers, deadline - 2.0)
            if r > best_score:
                best_score = r
                best_points = current_centers.copy()
    
    if best_points is None:
        best_points = _build_initial_grid(5, 5, 4)
    
    return best_points
# EVOLVE-BLOCK-END
