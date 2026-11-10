"""
Strategy: Global Topological Optimization.
1. Model the packing as a periodic lattice defined by a small unit cell (e.g., 2x2x2 or 3x3x3).
2. Optimize the unit cell coordinates using BFGS on a smooth soft-min objective.
3. Use the optimized periodic structure to initialize a full 100-sphere configuration.
4. Refine the full configuration using Simulated Annealing (Metropolis-Hastings)
   to break periodicity constraints and fine-tune for the specific 100-sphere boundary conditions.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the valid radius for a set of centers."""
    if centers is None or centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    min_dist_wall = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(min_dist_wall)
    
    # Distance between spheres
    if N < 2:
        return r_wall
        
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist_pair = np.min(dist)
    r_pair = min_dist_pair / 2.0
    
    return min(r_wall, r_pair)

def _soft_min_obj_and_grad(flat_coords, beta, is_periodic=True, box_size=1.0):
    """
    Objective function and gradient for soft-min maximization.
    We minimize -soft_min(distances).
    """
    dim = 3
    n_atoms = len(flat_coords) // dim
    coords = flat_coords.reshape(n_atoms, dim)
    
    # For periodic case, we assume the unit cell is [0, box_size]^3 and atoms are inside.
    # However, for the full N=100 case, we handle boundaries explicitly.
    
    # Calculate all relevant distances
    if is_periodic:
        # This is only used for the small unit cell optimization where we assume periodicity
        # to find a good lattice.
        # For simplicity in the unit cell step, we just optimize coordinates in [0,1]^3
        # and calculate distances considering periodic boundary conditions.
        
        # Pair distances (periodic)
        # For a small cell (e.g. 4 atoms), this is cheap.
        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        # Minimum image convention
        diffs -= box_size * np.round(diffs / box_size)
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances are tricky in periodic model. 
        # Instead of walls, we just maximize the minimum pair distance in the periodic cell.
        # This finds a dense lattice.
        eff_dists = dists[np.triu_indices(n_atoms, k=1)]
        
        # Gradient of soft min
        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # d(soft_min)/d(x) = -1/beta * (1/sum) * sum(exp(-beta*d_i) * (-beta * dd_i/dx))
        #                   = 1/sum * sum(exp(-beta*d_i) * dd_i/dx)
        
        # We want to MINIMIZE -soft_min.
        # So gradient of objective = - gradient(soft_min)
        # grad_obj = - [ 1/sum * sum(exp(-beta*d_i) * dd_i/dx) ]
        
        # Let's compute weights w_i = exp(-beta * d_i)
        # To avoid overflow, subtract max
        m = np.max(eff_dists)
        weights = np.exp(-beta * (eff_dists - m))
        sum_weights = np.sum(weights)
        
        if sum_weights == 0:
            return 0.0, np.zeros_like(flat_coords)
        
        # Normalize weights
        w_norm = weights / sum_weights
        
        # We need to map back to pair indices
        iu = np.triu_indices(n_atoms, k=1)
        # Create a matrix of weights for all pairs (symmetric)
        w_mat = np.zeros((n_atoms, n_atoms))
        w_mat[iu] = w_norm
        w_mat[iu[1], iu[0]] = w_norm
        
        # Gradient of distance d_ij wrt coords
        # d_ij = ||c_i - c_j||
        # dd_ij/dc_i = (c_i - c_j) / d_ij
        # dd_ij/dc_j = (c_j - c_i) / d_ij = - dd_ij/dc_i
        
        # grad(soft_min) = sum_j w_ij * dd_ij/dc_i for all j
        # But we have w_mat.
        # grad_soft_min[i] = sum_j w_mat[i,j] * (c_i - c_j) / d_ij
        
        grad_soft_min = np.zeros_like(coords)
        # Efficiently:
        # diff = c_i - c_j
        # term = w * (c_i - c_j) / d
        # sum over j
        
        # Vectorized computation
        # dists_mat is n x n
        # We need to avoid division by zero on diagonal (w is 0 there anyway, but safe to mask)
        mask = dists > 1e-8
        inv_dists = np.zeros_like(dists)
        inv_dists[mask] = 1.0 / dists[mask]
        
        # w_mat * (c_i - c_j) / d_ij
        # shape: (n, n, 3)
        # diffs is (n, n, 3)
        terms = w_mat[:, :, np.newaxis] * diffs * inv_dists[:, :, np.newaxis]
        
        # grad_soft_min[i] = sum_j terms[i, j]
        grad_soft_min = np.sum(terms, axis=1)
        
        # Objective is -soft_min
        obj = -(-1.0 / beta * (m + np.log(sum_weights))) # -soft_min = 1/beta * log_sum_exp_term?
        # Wait. soft_min = -1/beta * log(sum(exp(-beta*d))).
        # min_d = soft_min approx.
        # We want to maximize min_d -> minimize -min_d.
        # obj = -soft_min = 1/beta * log(sum(exp(-beta*d)))
        # Let's re-calculate obj value carefully.
        log_sum_exp = m + np.log(sum_weights)
        soft_min_val = -log_sum_exp / beta
        obj_val = -soft_min_val
        
        # Gradient of obj = - Gradient of soft_min
        grad_flat = -grad_soft_min.flatten()
        
        return obj_val, grad_flat

    else:
        # Full N=100 case: Non-periodic, with walls.
        coords = flat_coords.reshape(n_atoms, dim)
        
        # Pair distances
        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        # w_i = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
        abs_coords = np.abs(coords)
        max_coord = np.max(abs_coords, axis=1)
        wall_dists = 0.5 - max_coord
        
        # Effective distances for soft min:
        # Pairs: d_ij
        # Walls: 2 * w_i (so that min(eff) / 2 is the radius)
        
        iu = np.triu_indices(n_atoms, k=1)
        pair_dists = dists[iu]
        
        eff_dists = np.concatenate([pair_dists, 2.0 * wall_dists])
        
        # Soft min
        m = np.max(eff_dists)
        weights = np.exp(-beta * (eff_dists - m))
        sum_weights = np.sum(weights)
        
        if sum_weights == 0:
            return 0.0, np.zeros_like(flat_coords)
            
        w_norm = weights / sum_weights
        
        log_sum_exp = m + np.log(sum_weights)
        soft_min_val = -log_sum_exp / beta
        obj_val = -soft_min_val
        
        # Gradient
        # grad_soft_min = sum_k w_k * d(eff_k)/d(x)
        # eff_k for pairs: d_ij. d(eff)/d(c_i) = (c_i - c_j)/d_ij.
        # eff_k for walls: 2*w_i. d(eff)/d(c_i) = -2 * sign(c_i) for the max coordinate.
        
        grad_soft_min = np.zeros_like(coords)
        
        # Pair part
        # w_pair is length n_pairs
        # Map to matrix
        w_pair_mat = np.zeros((n_atoms, n_atoms))
        w_pair_mat[iu] = w_norm[:len(pair_dists)]
        w_pair_mat[iu[1], iu[0]] = w_norm[:len(pair_dists)]
        
        mask = dists > 1e-8
        inv_dists = np.zeros_like(dists)
        inv_dists[mask] = 1.0 / dists[mask]
        
        terms_pair = w_pair_mat[:, :, np.newaxis] * diffs * inv_dists[:, :, np.newaxis]
        grad_soft_min += np.sum(terms_pair, axis=1)
        
        # Wall part
        w_wall = w_norm[len(pair_dists):]
        # For each atom i, if it's the max coordinate, derivative is -2 * sign(c_i, max_idx)
        # max_coord_idx = argmax(abs_coords, axis=1)
        max_idx = np.argmax(abs_coords, axis=1)
        signs = np.sign(coords[np.arange(n_atoms), max_idx])
        
        # grad_wall[i] = -2 * w_wall[i] * sign(c_i, max_idx) * e_max_idx
        grad_wall_vec = np.zeros_like(coords)
        grad_wall_vec[np.arange(n_atoms), max_idx] = -2.0 * w_wall * signs
        grad_soft_min += grad_wall_vec
        
        grad_flat = -grad_soft_min.flatten()
        
        return obj_val, grad_flat

def _optimize_unit_cell(cell_coords, beta=20.0):
    """Optimize a small periodic unit cell to find a dense lattice."""
    flat = cell_coords.flatten()
    
    def obj_grad(flat_coords):
        val, grad = _soft_min_obj_and_grad(flat_coords, beta, is_periodic=True, box_size=1.0)
        return val, grad
        
    res = minimize(obj_grad, flat, method='L-BFGS-B', 
                   bounds=[(0, 1)] * len(flat), 
                   options={'maxiter': 100, 'ftol': 1e-8})
    return res.x.reshape(cell_coords.shape)

def _generate_lattice_config(cell_coords, cell_size=2, target_N=N):
    """
    Expand a unit cell to a grid that fits ~N spheres.
    Returns centers in [-0.5, 0.5]^3.
    """
    # Determine grid dimensions
    # We want roughly N spheres.
    # If cell has k atoms, and we use a GxGxG grid of cells, total = k * G^3.
    # Let's pick G such that k * G^3 is close to N.
    k = len(cell_coords)
    # Estimate G
    G = int(np.round((target_N / k) ** (1.0/3.0)))
    if G < 1: G = 1
    
    # Generate grid points
    # Unit cell is [0,1)^3. We scale it to fit in [-0.5, 0.5]^3?
    # Actually, let's place the lattice in the cube.
    # The lattice period should be such that G periods fit in size 1.
    # So period = 1.0 / G.
    # Coordinates in unit cell are in [0,1].
    # Global coord = (cell_coord + offset) * period - 0.5 ?
    # No, let's just scale the unit cell to size 1/G and tile it.
    
    period = 1.0 / G
    
    centers = []
    for ix in range(G):
        for iy in range(G):
            for iz in range(G):
                # Offset for this cell
                offset = np.array([ix, iy, iz]) * period
                # Add cell coords scaled by period
                for c in cell_coords:
                    # c is in [0,1]
                    # Global pos = offset + c * period
                    pos = offset + c * period
                    # Center the whole thing in [-0.5, 0.5]
                    # The grid spans [0, 1]. Center is 0.5.
                    # So shift by -0.5
                    pos -= 0.5
                    centers.append(pos)
                    
    centers = np.array(centers)
    
    # We might have too many or too few.
    # If too many, remove random ones? Or just take first N?
    # Taking first N breaks symmetry at the end.
    # Better: if len > N, randomly subsample.
    if len(centers) > N:
        idx = np.random.choice(len(centers), N, replace=False)
        centers = centers[idx]
    elif len(centers) < N:
        # Pad with random points?
        # This is bad. We should choose G such that k*G^3 >= N and then subsample.
        # Let's ensure G is large enough.
        pass
        
    return centers

def _simulated_annealing(centers, t_start=0.01, t_end=0.001, max_iter=5000, deadline=None):
    """
    Refine centers using SA.
    """
    current_radius = _compute_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()
    
    t = t_start
    cooling_rate = (t_end / t_start) ** (1.0 / max_iter)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Random move
        idx = np.random.randint(N)
        delta = np.random.normal(0, t, 3) # Step size proportional to T? Or fixed?
        # Let's use fixed step size scaled by current radius?
        # Simple: step = 0.01 * t / t_start ? No, just use t as scale.
        new_centers = centers.copy()
        new_centers[idx] += delta
        
        # Clip to box
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_radius = _compute_radius(new_centers)
        
        # Acceptance criterion
        if new_radius > current_radius:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new - current) / T)
            # Note: we are maximizing, so delta_R = new - current < 0
            prob = np.exp((new_radius - current_radius) / t)
            if np.random.rand() < prob:
                centers = new_centers
                current_radius = new_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
                    
        t *= cooling_rate
        
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # 1. Generate diverse initial unit cells
    # FCC-like: 4 atoms
    # BCC-like: 2 atoms
    # Simple Cubic: 1 atom
    # Random: 4 atoms
    
    cell_templates = []
    
    # FCC
    fcc = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    cell_templates.append(fcc)
    
    # BCC
    bcc = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ])
    cell_templates.append(bcc)
    
    # Random 4
    np.random.seed(42)
    rand_cell = np.random.uniform(0, 1, (4, 3))
    cell_templates.append(rand_cell)
    
    # Random 8 (more freedom)
    rand_cell8 = np.random.uniform(0, 1, (8, 3))
    cell_templates.append(rand_cell8)
    
    # Optimize each unit cell
    optimized_cells = []
    for cell in cell_templates:
        if time.time() > deadline - 60.0:
            break
        opt_cell = _optimize_unit_cell(cell, beta=20.0)
        # Shift to ensure min coord > 0 and max < 1 to avoid boundary issues in tiling
        # Actually, _optimize_unit_cell uses bounds (0,1).
        # If atoms are at 0, they are on the boundary.
        # Let's nudge them slightly inside.
        opt_cell = np.clip(opt_cell, 0.01, 0.99)
        optimized_cells.append(opt_cell)
        
    # 2. Generate full configurations from optimized cells
    candidates = []
    for cell in optimized_cells:
        if time.time() > deadline - 30.0:
            break
        # Try a few grid sizes
        for G in [3, 4, 5]:
            config = _generate_lattice_config(cell, cell_size=G, target_N=N)
            if config is not None and len(config) == N:
                candidates.append(config)
                
    # Also add a random candidate
    candidates.append(np.random.uniform(-0.45, 0.45, (N, 3)))
    
    # 3. Refine candidates with SA
    for config in candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Pre-process: quick local push to fix obvious overlaps
        # Simple gradient step on soft min?
        # Let's just run SA. SA can handle overlaps by rejecting bad moves, 
        # but starting with overlaps is bad.
        # Let's do a quick "push" phase first.
        
        # Quick Push
        c = config.copy()
        for _ in range(100):
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair_idx = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = min_pair_idx
            d_pair = dist[i, j]
            if d_pair < 0.1: # Threshold
                dir_vec = c[i] - c[j]
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-6:
                    dir_vec /= norm
                    c[i] += 0.01 * dir_vec
                    c[j] -= 0.01 * dir_vec
            c = np.clip(c, -0.499, 0.499)
            
        # SA
        refined, r = _simulated_annealing(c, t_start=0.02, t_end=0.0005, max_iter=2000, deadline=deadline)
        
        if r > best_radius:
            best_radius = r
            best_centers = refined.copy()
            
    # Fallback
    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, (N, 3))
        
    return best_centers

# EVOLVE-BLOCK-END
