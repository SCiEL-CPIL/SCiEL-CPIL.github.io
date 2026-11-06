"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _bcc_lattice_init(n_spheres=50):
    """Generate n_spheres centers on a distorted BCC superlattice scaled to fit in [-0.5,0.5]^3."""
    # BCC has 2 points per cubic cell: corners + body center
    # We want approximately n_spheres/2 cubic cells
    n_cells = max(1, int(round(n_spheres / 2.0)))
    # Arrange cells in a roughly cubic grid
    m = int(np.ceil(n_cells ** (1.0/3.0)))
    while m**3 < n_cells:
        m += 1
    # Generate all BCC points in m x m x m superlattice
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner point
                points.append([ix, iy, iz])
                # Body center point
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
    
    points = np.array(points)
    
    # Take first n_spheres points
    points = points[:n_spheres]
    
    # Scale to fit in [-0.5, 0.5]^3
    # The lattice spans [0, m-1] in each dimension for corners, [0.5, m-0.5] for centers
    # Overall span is [0, m-1] for the corner points
    span = m - 1 if m > 1 else 1.0
    # Scale so that the lattice fits with some margin
    # We want the max extent to be about 0.45 (leaving room for sphere radius)
    scale = 0.9 / span if span > 0 else 1.0
    points = points * scale - scale * (m - 1) / 2.0
    
    # Add small random perturbation to break symmetry
    rng = np.random.RandomState(42)
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    
    # Clip to ensure they're inside the cube
    points = np.clip(points, -0.45, 0.45)
    
    return points

def _repulsive_energy_and_grad(centers, wall_penalty=100.0, pair_penalty=1.0, softness=0.01):
    """
    Compute repulsive potential energy and gradient.
    Energy = sum over pairs of 1/(d_ij^2 + eps) + wall penalties.
    Lower energy = better packing (larger minimum distance).
    """
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pair interactions
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            d2 = np.dot(diff, diff) + 1e-12
            # Repulsive: 1/(d2 + eps) - encourages larger distances
            inv_d2 = 1.0 / (d2 + softness * softness)
            energy += pair_penalty * inv_d2
            # Gradient: d/dx_i [1/(d2+eps)] = -2*diff / (d2+eps)^2
            grad[i] += pair_penalty * 2.0 * diff * inv_d2 * inv_d2
            grad[j] -= pair_penalty * 2.0 * diff * inv_d2 * inv_d2
    
    # Wall penalties: push centers away from walls
    for i in range(n):
        for k in range(3):
            c = centers[i, k]
            # Penalty if too close to wall at -0.5 or +0.5
            dist_to_neg_wall = c + 0.5
            dist_to_pos_wall = 0.5 - c
            # Smooth penalty: 1/(dist + eps)^2
            if dist_to_neg_wall < 0.15:
                p = wall_penalty / (dist_to_neg_wall + softness * softness) ** 2
                energy += p
                grad[i, k] += wall_penalty * 2.0 / (dist_to_neg_wall + softness * softness) ** 3
            if dist_to_pos_wall < 0.15:
                p = wall_penalty / (dist_to_pos_wall + softness * softness) ** 2
                energy += p
                grad[i, k] -= wall_penalty * 2.0 / (dist_to_pos_wall + softness * softness) ** 3
    
    return energy, grad

def _soft_min_radius(centers, beta=50.0):
    """
    Soft minimum of all pairwise distances and wall distances using log-sum-exp.
    Returns a smooth approximation to the minimum radius.
    """
    n = len(centers)
    # Collect all constraint values (distances that define radius)
    # Pair distances / 2
    # Wall distances
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            vals.append(d)
        for k in range(3):
            wall_dist = 0.5 - abs(centers[i, k])
            vals.append(wall_dist)
    
    vals = np.array(vals)
    # Soft min via log-sum-exp: -1/beta * log(sum(exp(-beta * vals)))
    min_val = np.min(vals)
    exp_vals = np.exp(-beta * (vals - min_val))
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta
    return soft_min

def _soft_min_radius_grad(centers, beta=50.0):
    """Gradient of the soft minimum radius."""
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Collect all constraint values and their gradients
    vals = []
    grads = []
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                d = 1e-12
            r = d / 2.0
            vals.append(r)
            # d/dcenters[i] (|diff|/2) = diff / (2*d)
            g_i = diff / (2.0 * d)
            g_j = -diff / (2.0 * d)
            grads.append((i, g_i))
            grads.append((j, g_j))
        for k in range(3):
            c = centers[i, k]
            wall_dist = 0.5 - abs(c)
            vals.append(wall_dist)
            # d/dc (0.5 - |c|) = -sign(c)
            g = np.zeros(3)
            g[k] = -np.sign(c) if c != 0 else -1.0
            grads.append((i, g))
    
    vals = np.array(vals)
    min_val = np.min(vals)
    exp_vals = np.exp(-beta * (vals - min_val))
    sum_exp = np.sum(exp_vals)
    
    # Soft min = min_val - log(sum_exp) / beta
    # Gradient: sum_j [ -beta * exp(-beta*(v_j - min_val)) / sum_exp * grad(v_j) ] / beta
    #          = -sum_j [ exp(-beta*(v_j - min_val)) / sum_exp * grad(v_j) ]
    # But we also need to account for the derivative of min_val term... 
    # Actually, for the log-sum-exp soft min:
    # f = -1/beta * log(sum exp(-beta * v_j))
    # df/dx = -1/beta * [sum exp(-beta*v_j) * (-beta) * dv_j/dx] / sum exp(-beta*v_j)
    #        = sum_j [exp(-beta*v_j) / sum exp(-beta*v_j)] * dv_j/dx
    # Wait, let me redo this carefully.
    # f = -1/beta * log(S) where S = sum_j exp(-beta * v_j)
    # df/dx = -1/beta * (1/S) * dS/dx
    # dS/dx = sum_j exp(-beta*v_j) * (-beta) * dv_j/dx
    # df/dx = -1/beta * (1/S) * (-beta) * sum_j exp(-beta*v_j) * dv_j/dx
    #        = (1/S) * sum_j exp(-beta*v_j) * dv_j/dx
    #        = sum_j w_j * dv_j/dx where w_j = exp(-beta*v_j) / S
    
    weights = exp_vals / sum_exp
    
    for idx, (i, g) in enumerate(grads):
        grad[i] += weights[idx] * g
    
    return grad

def _exact_radius(centers):
    """Compute the exact (non-soft) minimum radius."""
    n = len(centers)
    min_r = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            if d < min_r:
                min_r = d
        for k in range(3):
            wall_dist = 0.5 - abs(centers[i, k])
            if wall_dist < min_r:
                min_r = wall_dist
    return min_r

def _polish(centers, max_time=5.0, beta=100.0):
    """Final polish pass with tight tolerances."""
    start = time.time()
    best = centers.copy()
    best_r = _exact_radius(best)
    
    for _ in range(20):
        if time.time() - start > max_time:
            break
        try:
            res = minimize(
                lambda x: -_soft_min_radius(x.reshape(-1, 3), beta),
                best.ravel(),
                jac=lambda x: -_soft_min_radius_grad(x.reshape(-1, 3), beta).ravel(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            new_centers = res.x.reshape(-1, 3)
            new_r = _exact_radius(new_centers)
            if new_r > best_r:
                best = new_centers
                best_r = new_r
        except Exception:
            break
    
    return best, best_r

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Leave 10s margin for the 150s kill
    
    # Step 1: BCC lattice initialization
    centers = _bcc_lattice_init(N)
    
    # Step 2: Repulsive potential energy minimization (coarse pass)
    t_coarse_start = time.time()
    for _ in range(5):
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                lambda x: _repulsive_energy_and_grad(x.reshape(-1, 3))[0],
                centers.ravel(),
                jac=lambda x: _repulsive_energy_and_grad(x.reshape(-1, 3))[1].ravel(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if res.success:
                new_centers = res.x.reshape(-1, 3)
                if _exact_radius(new_centers) > _exact_radius(centers):
                    centers = new_centers
        except Exception:
            break
    
    # Step 3: Soft-min radius maximization (fine pass)
    t_fine_start = time.time()
    for _ in range(3):
        if time.time() > deadline - 15.0:
            break
        try:
            res = minimize(
                lambda x: -_soft_min_radius(x.reshape(-1, 3), beta=50.0),
                centers.ravel(),
                jac=lambda x: -_soft_min_radius_grad(x.reshape(-1, 3), beta=50.0).ravel(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            new_centers = res.x.reshape(-1, 3)
            if _exact_radius(new_centers) > _exact_radius(centers):
                centers = new_centers
        except Exception:
            break
    
    # Step 4: Final polish
    if time.time() < deadline - 5.0:
        remaining = max(2.0, deadline - time.time() - 5.0)
        centers, _ = _polish(centers, max_time=remaining, beta=100.0)
    
    # Ensure centers are strictly inside the cube
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers
# EVOLVE-BLOCK-END
