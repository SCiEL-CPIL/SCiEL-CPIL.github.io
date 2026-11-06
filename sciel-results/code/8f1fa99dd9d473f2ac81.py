"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_score(centers):
    """Compute the maximum common radius for given centers."""
    # Boundary constraint: min distance to each face
    # Cube is [-0.5, 0.5]^3
    dist_to_faces = np.min(np.abs(0.5 - np.abs(centers)), axis=1)
    min_boundary = np.min(dist_to_faces)
    
    # Pairwise distances
    if len(centers) < 2:
        min_pair = np.inf
    else:
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    
    return min(min_boundary, min_pair)

def _energy_and_grad(centers, beta=100.0):
    """
    Soft-min energy: -log(sum(exp(-beta * d_i))) approximates min(d_i) * log(num_terms)/beta
    for large beta. We minimize this (i.e., maximize the soft min).
    Returns (energy, gradient) where we want to MINIMIZE energy.
    """
    centers = np.asarray(centers).reshape(-1, 3)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Boundary terms: distance to each of the 6 faces
    # For center c, distance to face x=0.5 is 0.5 - c[0], etc.
    # We need min over all faces and all centers.
    # Soft min over boundary distances:
    # d_boundary_i = min over 6 faces of (0.5 - |c_ik|)
    # Actually, let's define boundary distance for sphere i as:
    #   min(0.5 - c_ix, c_ix + 0.5, 0.5 - c_iy, c_iy + 0.5, 0.5 - c_iz, c_iz + 0.5)
    
    # Let's compute all boundary distances
    # For each center, 6 face distances
    cx, cy, cz = centers[:, 0], centers[:, 1], centers[:, 2]
    bd = np.column_stack([
        0.5 - cx, cx + 0.5,
        0.5 - cy, cy + 0.5,
        0.5 - cz, cz + 0.5
    ])  # shape (n, 6)
    
    # Soft min over all boundary distances (n*6 terms)
    bd_flat = bd.ravel()
    # For numerical stability, subtract max
    bd_max = np.max(bd_flat)
    exp_bd = np.exp(-beta * (bd_flat - bd_max))
    log_sum_exp = -beta * bd_max + np.log(np.sum(exp_bd))
    # Energy contribution: we want to maximize min boundary, so minimize -soft_min
    # soft_min ≈ -log_sum_exp / beta ... let me re-derive.
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    # We want to MAXIMIZE soft_min, so MINIMIZE -soft_min = 1/beta * log(sum(exp(-beta * x_i)))
    
    energy_boundary = (1.0 / beta) * log_sum_exp
    energy += energy_boundary
    
    # Gradient of energy_boundary w.r.t. centers
    # d/dx_j [1/beta * log(sum_i exp(-beta * x_i))] = -sum_i exp(-beta*x_i) * (dx_i/dx_j) / sum_i exp(-beta*x_i)
    # = -sum_i w_i * (dx_i/dx_j) where w_i = exp(-beta*x_i) / sum(exp(-beta*x_i))
    
    w = exp_bd / np.sum(exp_bd)  # shape (n*6,)
    w = w.reshape(n, 6)
    
    # dx_i/dx_j for each face:
    # face 0: 0.5 - cx -> d/dc_x = -1
    # face 1: cx + 0.5 -> d/dc_x = +1
    # face 2: 0.5 - cy -> d/dc_y = -1
    # face 3: cy + 0.5 -> d/dc_y = +1
    # face 4: 0.5 - cz -> d/dc_z = -1
    # face 5: cz + 0.5 -> d/dc_z = +1
    
    grad[:, 0] -= w[:, 0] * (-1.0) + w[:, 1] * (1.0)
    grad[:, 1] -= w[:, 2] * (-1.0) + w[:, 3] * (1.0)
    grad[:, 2] -= w[:, 4] * (-1.0) + w[:, 5] * (1.0)
    
    # Pairwise distances
    if n >= 2:
        # Compute all pairwise distances
        # d_ij = ||c_i - c_j||
        # Soft min over all i<j
        # There are n*(n-1)/2 pairs
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, n, 3)
        dists = np.sqrt(np.sum(diffs**2, axis=2))  # (n, n)
        np.fill_diagonal(dists, np.inf)
        
        # Get upper triangle
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]  # (n*(n-1)/2,)
        
        if len(pair_dists) > 0:
            pd_max = np.max(pair_dists)
            exp_pd = np.exp(-beta * (pair_dists - pd_max))
            log_sum_exp_p = -beta * pd_max + np.log(np.sum(exp_pd))
            energy_pair = (1.0 / beta) * log_sum_exp_p
            energy += energy_pair
            
            w_p = exp_pd / np.sum(exp_pd)
            
            # Gradient: d/d(c_i) of energy_pair
            # energy_pair = 1/beta * log(sum_{i<j} exp(-beta * d_ij))
            # d/d(c_i) = -1/beta * sum_{j} exp(-beta*d_ij) * d(d_ij)/d(c_i) / sum(exp(-beta*d_ij))
            # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
            # So gradient contribution for center i: -sum_j w_ij * (c_i - c_j)/d_ij
            
            grad_pair = np.zeros_like(centers)
            for idx in range(len(pair_dists)):
                i, j = iu[0][idx], iu[1][idx]
                w_ij = w_p[idx]
                d_ij = pair_dists[idx]
                if d_ij > 1e-12:
                    diff = centers[i] - centers[j]
                    grad_pair[i] += w_ij * diff / d_ij
                    grad_pair[j] -= w_ij * diff / d_ij
            
            grad -= grad_pair
    
    return energy, grad

def _bcc_lattice_init(n_points):
    """Generate initial points on a BCC-like lattice, then randomly perturb."""
    # Start with a grid and add random perturbations
    m = int(np.ceil(n_points ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    points = grid[:n_points].copy()
    # Add small random perturbations
    points += np.random.randn(n_points, 3) * 0.01
    # Clip to stay inside
    points = np.clip(points, -0.49, 0.49)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # Try multiple random restarts with different initializations
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline - 30.0:
            break
        
        # Initialize
        if restart == 0:
            centers = _bcc_lattice_init(N)
        elif restart == 1:
            # Simple cubic grid
            m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N].copy()
            centers += rng.randn(N, 3) * 0.005
        else:
            # Random initialization
            centers = rng.uniform(-0.45, 0.45, size=(N, 3))
        
        # Optimize using L-BFGS-B with soft-min energy
        beta_schedule = [50.0, 100.0, 200.0, 400.0]
        
        for beta in beta_schedule:
            if time.time() > deadline - 20.0:
                break
            
            def fun(x):
                e, g = _energy_and_grad(x.reshape(-1, 3), beta=beta)
                return e, g
            
            try:
                res = minimize(fun, centers.ravel(), method='L-BFGS-B', jac=True,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})
                centers = res.x.reshape(-1, 3)
            except Exception:
                pass
            
            # Clip to valid range
            centers = np.clip(centers, -0.4999, 0.4999)
        
        # Evaluate
        score = _compute_score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
        
        # If we have a good score, do a final refinement with coordinate descent
        if best_score > 0.13:
            if time.time() > deadline - 15.0:
                break
            
            # Coordinate descent refinement
            cd_centers = best_centers.copy()
            for iteration in range(200):
                if time.time() > deadline - 10.0:
                    break
                improved = False
                for i in range(N):
                    for dim in range(3):
                        # Try small steps
                        for step in [0.001, -0.001, 0.0005, -0.0005, 0.002, -0.002]:
                            old = cd_centers[i, dim]
                            cd_centers[i, dim] = old + step
                            # Check validity
                            if np.any(cd_centers < -0.5) or np.any(cd_centers > 0.5):
                                cd_centers[i, dim] = old
                                continue
                            s = _compute_score(cd_centers)
                            if s > best_score + 1e-12:
                                best_score = s
                                best_centers = cd_centers.copy()
                                improved = True
                            else:
                                cd_centers[i, dim] = old
                if not improved:
                    break
    
    if best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
