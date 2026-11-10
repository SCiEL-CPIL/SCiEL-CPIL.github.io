"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Fixed 5x5x4 grid lattice (100 points), centered at origin
# Coordinates in [-0.5, 0.5] range
_GRID_X = np.linspace(-0.5, 0.5, 5)
_GRID_Y = np.linspace(-0.5, 0.5, 5)
_GRID_Z = np.linspace(-0.5, 0.5, 4)
_BASE_LATTICE = np.array([(x, y, z) for z in _GRID_Z for y in _GRID_Y for x in _GRID_X])
# Center the lattice
_BASE_LATTICE = _BASE_LATTICE - _BASE_LATTICE.mean(axis=0)


def _compute_radius(centers):
    """Compute the radius for a set of centers."""
    # Wall clearance
    wall_clear = 0.5 - np.abs(centers).max(axis=0).max()
    # Inter-sphere clearance
    if len(centers) > 1:
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = dists.min()
        sphere_clear = min_dist / 2.0
    else:
        sphere_clear = 0.5
    return min(wall_clear, sphere_clear)


def _rotation_matrix_from_angles(a, b, c):
    """Create rotation matrix from Euler angles (ZYZ convention)."""
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    
    # Rz(a) * Ry(b) * Rz(c)
    R = np.array([
        [ca*cb*cc - sa*sc, -ca*cb*sc - sa*cc, ca*sb],
        [sa*cb*cc + ca*sc, -sa*cb*sc + ca*cc, sa*sb],
        [-sb*cc, sb*sc, cb]
    ])
    return R


def _evaluate_params(params, lattice):
    """Evaluate radius given rotation angles and scale."""
    a, b, c, s = params
    R = _rotation_matrix_from_angles(a, b, c)
    centers = (lattice @ R.T) * s
    return _compute_radius(centers)


def _objective(params, lattice):
    """Negative radius for minimization."""
    return -_evaluate_params(params, lattice)


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Start with identity rotation and scale to fit
    best_params = np.array([0.0, 0.0, 0.0, 1.0])
    best_radius = _evaluate_params(best_params, _BASE_LATTICE)
    
    # Multi-start optimization
    np.random.seed(42)
    n_starts = 20
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Random initial rotation and scale
        if i == 0:
            a0, b0, c0 = 0.0, 0.0, 0.0
            s0 = 1.0
        else:
            a0 = np.random.uniform(-np.pi, np.pi)
            b0 = np.random.uniform(-np.pi/2, np.pi/2)
            c0 = np.random.uniform(-np.pi, np.pi)
            s0 = np.random.uniform(0.8, 1.2)
        
        params0 = np.array([a0, b0, c0, s0])
        
        # Use L-BFGS-B with bounds
        bounds = [
            (-np.pi, np.pi),
            (-np.pi/2, np.pi/2),
            (-np.pi, np.pi),
            (0.5, 1.5)
        ]
        
        try:
            result = minimize(
                _objective, params0, args=(_BASE_LATTICE,),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < -best_radius:
                current_radius = -result.fun
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Apply best rotation and scale
    a, b, c, s = best_params
    R = _rotation_matrix_from_angles(a, b, c)
    centers = (_BASE_LATTICE @ R.T) * s
    
    # Final polish: local optimization of individual centers
    # This allows the grid to "breathe" for better packing
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Compute forces
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dists, 1e-10)
            
            # Wall forces
            wall_force = np.zeros_like(centers)
            for k in range(3):
                wall_force[:, k] += np.sign(centers[:, k]) * (0.5 - np.abs(centers[:, k]))
            
            # Sphere-sphere repulsion (only for close pairs)
            sphere_force = np.zeros_like(centers)
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    d = dists[i, j]
                    if d < 0.25:  # Only interact with nearby spheres
                        f = (0.2 - d) / d if d > 0 else 0
                        if f > 0:
                            sphere_force[i] += f * (centers[i] - centers[j])
                            sphere_force[j] -= f * (centers[i] - centers[j])
            
            centers += 0.01 * (wall_force + sphere_force)
            # Clamp to cube
            centers = np.clip(centers, -0.4999, 0.4999)
    
    # Ensure centers are strictly inside the cube
    centers = np.clip(centers, -0.499999, 0.499999)
    
    return centers
# EVOLVE-BLOCK-END
