"""Combined program for 100 equal spheres in a cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

# Deadline guard
START_TIME = time.time()
DEADLINE = 145.0  # 5s safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _compute_radius(centres):
    """
    Compute the packing radius for a set of centres.
    Uses cKDTree for efficient nearest neighbor search.
    """
    if len(centres) < 2:
        return 0.5 - np.max(np.abs(centres))
    
    # Wall distance
    wall_dists = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distance using KDTree
    try:
        tree = cKDTree(centres)
        dists, _ = tree.query(centres, k=2)
        min_pair_dist = np.min(dists[:, 1])
    except Exception:
        d = cdist(centres, centres)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = d.min()
        
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_objective(x, beta, iu):
    """
    Smooth surrogate for the negative packing radius.
    Uses precomputed upper triangle indices for efficiency.
    """
    centres = x.reshape(N, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centres)
    wall_vals = wall_dists.flatten()
    
    # Pairwise distances / 2 using upper triangle indices
    d = cdist(centres, centres, metric='euclidean')
    pair_vals = d[iu] / 2.0
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft min using log-sum-exp trick
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _construct_bcc_init():
    """Body-centered cubic lattice initialization."""
    m = 5  # 5x5x5 = 125 positions, take 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add small random perturbations to break symmetry
    np.random.seed(42)
    perturbation = np.random.randn(len(grid), 3) * 0.01
    grid = grid + perturbation
    
    # Clip to stay within bounds
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid[:N]

def _construct_fcc_init():
    """Face-centered cubic lattice initialization."""
    n_cells = 3
    a = 1.0 / n_cells  # cell size
    
    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                base = np.array([ix, iy, iz]) * a
                # FCC positions in cell
                fcc_pos = [
                    np.array([0, 0, 0]),
                    np.array([0.5, 0.5, 0]) * a,
                    np.array([0.5, 0, 0.5]) * a,
                    np.array([0, 0.5, 0.5]) * a,
                ]
                for fp in fcc_pos:
                    positions.append(base + fp)
    
    positions = np.array(positions)
    
    # Center the whole thing
    positions = positions - 0.5  # shift to [-0.5, 0.5]
    
    # Add small perturbations
    np.random.seed(42)
    perturbation = np.random.randn(len(positions), 3) * 0.005
    positions = positions + perturbation
    
    # Clip
    positions = np.clip(positions, -0.49, 0.49)
    
    return positions[:N]

def _construct_grid_init():
    """Standard grid initialization."""
    # 5x5x4 = 100
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5, 0.5, nx)
    ys = np.linspace(-0.5, 0.5, ny)
    zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        # Adjust to get exactly N
        if len(centers) > N:
            centers = centers[:N]
        else:
            # Pad with random
            rng = np.random.default_rng(42)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
            
    # Add noise
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _construct_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _repulsion_search(centers, time_budget=10.0):
    """
    Local search using repulsion forces with KD-tree acceleration.
    This is the primary structural improvement: fast local refinement.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    # Parameters
    step_size = 0.003
    decay = 0.97
    min_step = 1e-7
    margin = 1.05
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 100
    
    while _time_left() > 1.0 and (time.time() - start_time) < time_budget and step_size > min_step and no_improve_count < max_no_improve:
        r = _compute_radius(centers)
        if r > best_r + 1e-12:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # 1. Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.01)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # 2. Pairwise forces using KD-tree
        tree = cKDTree(centers)
        search_radius = 2 * max(r_current * margin, 0.01)
        
        # Query neighbors
        neighbor_pairs = tree.query_pairs(r=search_radius)
        
        for i, j in neighbor_pairs:
            if i == j:
                continue
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9 and dist < search_radius:
                # Repulsion strength increases as they get closer
                strength = (search_radius - dist) / search_radius
                unit_diff = diff / dist
                forces[i] += strength * unit_diff
                forces[j] -= strength * unit_diff
        
        # Normalize forces to prevent huge jumps
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.4999, 0.4999)
        
        step_size *= decay
        r_current = r  # Update current radius for next loop's margin calc
        
    return best_centers, best_r

def _optimize_from_init(init_centres, time_budget=20.0, iu=None):
    """
    Run optimization from given initialization.
    Phase 1: Adaptive beta L-BFGS-B
    Phase 2: Repulsion Local Search (main work)
    Phase 3: Fine Polish
    """
    best_centres = init_centres.copy()
    best_r = _compute_radius(best_centres)
    
    if iu is None:
        # Precompute upper triangle indices
        iu = np.triu_indices(N, k=1)
    
    start_t = time.time()
    remaining = time_budget
    
    # Phase 1: Soft-min optimization with adaptive beta
    beta_start = 30.0
    beta_end = 3000.0
    n_beta_steps = 6
    
    if remaining > 8.0:
        for step in range(n_beta_steps):
            if _time_left() < 2.0 or (time.time() - start_t) > remaining - 2.0:
                break
                
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
            
            x0 = best_centres.flatten()
            
            try:
                max_iter = 150 if step < 3 else 80
                res = minimize(
                    _soft_min_objective,
                    x0,
                    args=(beta, iu),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (N * 3),
                    options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 20}
                )
                
                new_centres = res.x.reshape(N, 3)
                new_centres = np.clip(new_centres, -0.4999, 0.4999)
                
                new_r = _compute_radius(new_centres)
                if new_r > best_r:
                    best_r = new_r
                    best_centres = new_centres.copy()
            except Exception:
                pass

    # Phase 2: Repulsion Search
    if _time_left() > 5:
        rep_budget = min(15.0, max(0, remaining - (time.time() - start_t) - 3.0))
        if rep_budget > 2.0:
            rep_centres, rep_r = _repulsion_search(best_centres, time_budget=rep_budget)
            if rep_r > best_r:
                best_r = rep_r
                best_centres = rep_centres.copy()

    # Phase 3: Final Polish with high beta L-BFGS
    if _time_left() > 3:
        x0 = best_centres.flatten()
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                args=(2000.0, iu),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
            )
            new_centres = result.x.reshape(N, 3)
            new_centres = np.clip(new_centres, -0.4999, 0.4999)
            new_r = _compute_radius(new_centres)
            if new_r > best_r:
                best_r = new_r
                best_centres = new_centres.copy()
        except Exception:
            pass
            
        # One last quick repulsion pass
        if _time_left() > 1:
            rep_centres, rep_r = _repulsion_search(best_centres, time_budget=1.5)
            if rep_r > best_r:
                best_r = rep_r
                best_centres = rep_centres.copy()

    return best_centres, best_r

def construct():
    """Main entry point."""
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 145.0  # 5s safety margin
    
    # Precompute upper triangle indices for soft-min objective
    iu = np.triu_indices(N, k=1)
    
    best_centres = None
    best_r = -np.inf
    
    # Try multiple initializations
    inits = [
        _construct_bcc_init(),
        _construct_fcc_init(),
        _construct_grid_init(),
        _construct_halton_init(),
    ]
    
    # Also try a random initialization
    np.random.seed(123)
    random_init = np.random.uniform(-0.45, 0.45, (N, 3))
    inits.append(random_init)
    
    # Allocate time budget
    # We have ~145s. Give each init ~20-25s if it looks promising.
    
    for i, init in enumerate(inits):
        if _time_left() < 15:
            break
        
        # Give more time to structured inits, less to random
        budget = 25.0 if i < 4 else 10.0
        
        centres, r = _optimize_from_init(init, time_budget=budget, iu=iu)
        
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
            
    # Final global polish on the best candidate found
    if best_centres is not None and _time_left() > 10:
        polished_centres, polished_r = _optimize_from_init(best_centres, time_budget=_time_left() - 5, iu=iu)
        if polished_r > best_r:
            best_r = polished_r
            best_centres = polished_centres.copy()
            
    if best_centres is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N]
    
    # Ensure centres are strictly inside the cube
    best_centres = np.clip(best_centres, -0.4999, 0.4999)
    
    return best_centres
# EVOLVE-BLOCK-END
