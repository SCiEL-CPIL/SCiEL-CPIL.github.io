"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _min_radius(centers):
    """Compute the maximum common radius for given centers inside [-0.5, 0.5]^3."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    return min(min_pair / 2.0, np.min(wall))


def _rotation_matrix_from_angles(alpha, beta, gamma):
    """Create a rotation matrix from Euler angles (Z-X-Y convention)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    
    R = np.array([
        [cg * cb, cg * sb * sa - sg * ca, cg * sb * ca + sg * sa],
        [sg * cb, sg * sb * sa + cg * ca, sg * sb * ca - cg * sa],
        [-sb,    cb * sa,                 cb * ca]
    ])
    return R


def _lattice_packing(alpha, beta, gamma, scale, n_per_axis):
    """
    Generate a rotated, scaled lattice packing.
    Returns centers and the minimum radius.
    """
    R = _rotation_matrix_from_angles(alpha, beta, gamma)
    
    # Create a cubic lattice of n_per_axis^3 points
    ticks = (np.arange(n_per_axis) - (n_per_axis - 1) / 2.0) * scale
    lattice = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Rotate the lattice
    rotated = lattice @ R.T
    
    # Select the first N points (or all if fewer)
    if len(rotated) >= N:
        centers = rotated[:N]
    else:
        # Pad with random points if needed
        extra = np.random.randn(N - len(rotated), 3) * scale * 0.5
        centers = np.vstack([rotated, extra])
    
    # Center the configuration
    centers -= np.mean(centers, axis=0)
    
    # Scale to fit in cube
    max_coord = np.max(np.abs(centers))
    if max_coord > 0:
        centers = centers / max_coord * 0.48  # Leave margin for walls
    
    return centers


def _optimize_lattice_rotation(alpha_beta_gamma_scale, n_per_axis):
    """
    Objective function for optimizing rotation angles and scale.
    Returns negative of the minimum radius (to minimize).
    """
    alpha, beta, gamma, scale = alpha_beta_gamma_scale
    centers = _lattice_packing(alpha, beta, gamma, scale, n_per_axis)
    r = _min_radius(centers)
    return -r


def construct():
    """
    Build 100 equal-sphere packing in the unit cube [-0.5, 0.5]^3.
    Strategy:
      1. Start from a regular cubic lattice (5x5x5 = 125 sites, take 100).
      2. Optimize the global rotation matrix (Euler angles) and uniform scale
         to maximize the minimum clearance while keeping spheres inside the cube.
      3. Multiple random restarts with different initial rotations; keep the best.
    """
    t_start = time.time()
    time_limit = 115.0  # safety margin before 120s kill

    best_c = None
    best_r = 0.0
    rng = np.random.RandomState(42)

    # Try multiple random restarts with different initial rotations
    for restart in range(50):
        if time.time() - t_start > time_limit - 5:
            break

        # Random initial rotation angles
        alpha = rng.uniform(0, np.pi)
        beta = rng.uniform(0, np.pi)
        gamma = rng.uniform(0, 2 * np.pi)
        scale = rng.uniform(0.15, 0.25)

        x0 = np.array([alpha, beta, gamma, scale])
        
        # Define bounds for optimization
        bounds = [
            (0, np.pi),      # alpha
            (0, np.pi),      # beta
            (0, 2 * np.pi),  # gamma
            (0.1, 0.4)       # scale
        ]

        # Use Nelder-Mead for derivative-free optimization
        try:
            result = minimize(
                _optimize_lattice_rotation,
                x0,
                args=(5,),  # n_per_axis
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            if result.success or result.fun < -best_r:
                alpha_opt, beta_opt, gamma_opt, scale_opt = result.x
                centers = _lattice_packing(alpha_opt, beta_opt, gamma_opt, scale_opt, 5)
                r = _min_radius(centers)
                
                if r > best_r:
                    best_r = r
                    best_c = centers
        except Exception:
            continue

    if best_c is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    return best_c
# EVOLVE-BLOCK-END
