"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _lj_energy_and_grad(coords):
    """Compute LJ energy and gradient for a set of coordinates."""
    n = len(coords)
    # Pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-10)
    r = np.sqrt(r2)
    
    # Energy: 4 * sum(r^-12 - r^-6) over i < j
    r6_inv = r2 ** (-3.0)  # 1/r^6
    r12_inv = r6_inv ** 2  # 1/r^12
    pair_energy = 4.0 * (r12_inv - r6_inv)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dr * dr/dx
    # d/dr [4(r^-12 - r^-6)] = 4(-12 r^-13 + 6 r^-7) = 24(1 - 2 r^6) / r^7
    # Actually: 4*(-12*r^-13 + 6*r^-7) = 4*6*r^-7*(1 - 2*r^6) = 24*r^-7*(1 - 2*r^6)
    # dr/dx_i = (x_i - x_j) / r
    # So dE/dx_i = sum_j 24*(1 - 2*r_ij^6) / r_ij^7 * (x_i - x_j) / r_ij
    #           = sum_j 24*(1 - 2*r_ij^6) * (x_i - x_j) / r_ij^8
    # But we need to be careful with the factor of 1/2 from symmetric sum.
    # Actually, let's derive more carefully.
    # E = sum_{i<j} 4*(r_ij^-12 - r_ij^-6)
    # dE/dx_i = sum_{j != i} 4*(-12*r_ij^-14 + 6*r_ij^-8) * (x_i - x_j) / r_ij
    #          = sum_{j != i} 4*(-12 + 6*r_ij^6) * (x_i - x_j) / r_ij^14
    #          = 24 * sum_{j != i} (r_ij^6 - 2) * (x_i - x_j) / r_ij^14
    
    r6 = r2 ** 3.0  # r^6
    r14 = r2 ** 7.0  # r^14
    # Factor for each pair
    factor = 24.0 * (r6 - 2.0) / r14
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = sum_j factor_ij * (x_i - x_j)
    # diff[i,j] = x_i - x_j
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _make_icosahedron_shell(radius):
    """Generate icosahedral vertices scaled to given radius."""
    a = radius
    # Icosahedron vertices: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    verts = []
    for sign1 in [1, -1]:
        for sign2 in [1, -1]:
            verts.append([0.0, sign1 * 1.0, sign2 * PHI])
            verts.append([sign1 * 1.0, sign2 * PHI, 0.0])
            verts.append([sign1 * PHI, 0.0, sign2 * 1.0])
    return np.array(verts)


def _initial_icosahedron():
    """Create initial two-shell Mackay icosahedron for 55 atoms."""
    # Inner shell: 12 atoms at radius ~1.12 (LJ minimum)
    # Outer shell: 42 atoms (20 on faces + 12 on edges + ... actually Mackay icosahedron has specific structure)
    # For simplicity, use inner 12 + outer 43, but let's use a reasonable starting point.
    
    # Standard Mackay icosahedron for 55: center 1, inner 12, outer 42
    # Let's create: 1 center + 12 inner shell + 42 outer shell
    
    coords = [np.zeros(3)]  # center
    
    # Inner shell: 12 vertices of icosahedron at radius 1.12
    inner_radius = 1.12
    inner_verts = _make_icosahedron_shell(inner_radius)
    coords.extend([v for v in inner_verts])
    
    # Outer shell: 42 atoms - use scaled icosahedron vertices plus additional points
    # For a rough start, scale the icosahedron to larger radius and add face centers
    outer_radius = 2.24
    outer_verts = _make_icosahedron_shell(outer_radius)
    coords.extend([v for v in outer_verts])
    
    # We have 1 + 12 + 12 = 25. Need 30 more for 55.
    # Add face centers of the outer icosahedron (20 faces) and edge centers (30 edges)
    # Actually, let's just create a reasonable starting configuration and let optimization handle it.
    
    # Let me reconsider: create 55 points more carefully
    coords = [np.zeros(3)]  # center atom
    
    # Inner shell: 12 atoms
    inner_radius = 1.12
    inner_verts = _make_icosahedron_shell(inner_radius)
    coords.extend([v for v in inner_verts])
    
    # Outer shell: 42 atoms
    # Use a larger icosahedron and add midpoints
    outer_radius = 2.3
    outer_verts = _make_icosahedron_shell(outer_radius)
    coords.extend([v for v in outer_verts])
    
    # Add 30 more atoms: face centers and some edge points
    # Compute face centers of icosahedron
    # Icosahedron has 20 faces. Let's add face centers.
    # This is getting complex. Let's just use a simple approach:
    # Create 55 points on a sphere with some randomness, then optimize
    
    # Actually, let me create a better initial guess
    coords = []
    # Center
    coords.append(np.zeros(3))
    # Inner shell: 12
    inner_r = 1.12
    for v in _make_icosahedron_shell(inner_r):
        coords.append(v)
    # Outer shell: 42 - use scaled icosahedron + perturbations
    outer_r = 2.24
    for v in _make_icosahedron_shell(outer_r):
        coords.append(v)
    
    # We have 25. Need 30 more. Add them at intermediate radii
    mid_r = 1.68
    for v in _make_icosahedron_shell(mid_r):
        coords.append(v)
    
    # Now we have 37. Need 18 more.
    # Add points along edges
    # For now, just duplicate some with small perturbations
    rng = np.random.default_rng(42)
    while len(coords) < 55:
        base = coords[rng.integers(1, len(coords))]
        perturbation = rng.normal(0, 0.1, 3)
        new_point = base + perturbation
        coords.append(new_point)
    
    return np.array(coords, dtype=float)


def _relax_from_start(coords, deadline_time, max_iters=5000):
    """Relax a starting configuration using L-BFGS."""
    def objective(x):
        coords = x.reshape(-1, 3)
        energy, grad = _lj_energy_and_grad(coords)
        return energy, grad.flatten()
    
    x0 = coords.flatten()
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
    
    return result.x.reshape(-1, 3), result.fun


def construct():
    deadline = time.time() + 130  # Safety margin
    
    best_coords = None
    best_energy = np.inf
    
    # Try multiple starting points
    starts = []
    
    # Start 1: Mackay icosahedron approximation
    starts.append(_initial_icosahedron())
    
    # Start 2: Cubic lattice (original)
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    starts.append(np.array(points[:N], dtype=float) * 1.12)
    
    # Start 3: Random sphere
    rng = np.random.default_rng(123)
    random_coords = rng.normal(0, 1.5, (N, 3))
    starts.append(random_coords)
    
    for idx, start_coords in enumerate(starts):
        if time.time() > deadline:
            break
        
        try:
            relaxed, energy = _relax_from_start(start_coords, deadline)
            
            # Check minimum distance constraint
            min_dist = np.inf
            for i in range(len(relaxed)):
                for j in range(i+1, len(relaxed)):
                    d = np.linalg.norm(relaxed[i] - relaxed[j])
                    if d < min_dist:
                        min_dist = d
            
            if min_dist >= 0.5 - 1e-6:
                if energy < best_energy:
                    best_energy = energy
                    best_coords = relaxed
        except Exception as e:
            continue
    
    if best_coords is None:
        # Fallback to cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_coords = np.array(points[:N], dtype=float) * 1.12
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 5 and best_coords is not None:
        try:
            def objective(x):
                coords = x.reshape(-1, 3)
                energy, grad = _lj_energy_and_grad(coords)
                return energy, grad.flatten()
            
            x0 = best_coords.flatten()
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            polished = result.x.reshape(-1, 3)
            
            # Verify constraint
            min_dist = np.inf
            for i in range(len(polished)):
                for j in range(i+1, len(polished)):
                    d = np.linalg.norm(polished[i] - polished[j])
                    if d < min_dist:
                        min_dist = d
            
            if min_dist >= 0.5 - 1e-6:
                best_coords = polished
        except Exception:
            pass
    
    return best_coords
# EVOLVE-BLOCK-END
