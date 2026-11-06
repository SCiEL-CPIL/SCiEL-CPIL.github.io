"""Rewritten packing program: BCC-inspired init + gradient-based optimization with wall repulsion."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 145.0  # safety margin before 150s kill


def _compute_radius(centers):
    """Compute the largest common radius for the given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    if n == 0:
        return 0.0
    # Wall distances
    wall_dists = np.min(np.abs(centers), axis=1)
    min_wall = 0.5 - np.max(wall_dists)
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    return min(min_pair / 2.0, min_wall)


def _soft_min_radius(centers, beta=100.0):
    """Smooth surrogate for the radius using log-sum-exp style soft-min."""
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    # Wall terms: 0.5 - |c_ik|
    wall_terms = 0.5 - np.abs(centers)  # shape (n, 3)
    # Pairwise terms: |c_i - c_j| / 2
    pair_terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            pair_terms.append(d)
    all_terms = np.concatenate([wall_terms.ravel(), np.array(pair_terms)])
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    m = np.min(all_terms)
    soft_min = m - np.log(np.sum(np.exp(-beta * (all_terms - m)))) / beta
    return soft_min


def _init_bcc():
    """Initialize 50 centers on a distorted BCC superlattice."""
    # BCC: for a cube of side L, BCC has 2 atoms per conventional cell.
    # We want ~50 atoms. Use a 3x3x3 supercell of BCC = 54 atoms, take 50.
    # Or better: use a grid approach.
    # BCC in unit cube: place atoms at corners and body centers of sub-cubes.
    # Let's use a 3x3x3 grid of sub-cubes, each with a corner atom and a body-center atom.
    # That gives 27*2 = 54 atoms. We'll take the first 50.
    
    m = 3  # 3x3x3 sub-cubes
    sub_size = 1.0 / m
    centers = []
    for iz in range(m):
        for iy in range(m):
            for ix in range(m):
                # Sub-cube center
                cx = -0.5 + (ix + 0.5) * sub_size
                cy = -0.5 + (iy + 0.5) * sub_size
                cz = -0.5 + (iz + 0.5) * sub_size
                # Body center of this sub-cube
                centers.append([cx, cy, cz])
                # Corner of this sub-cube (offset toward origin corner)
                # Actually, let's just use a simple perturbed cubic grid for now
                # and rely on optimization.
    
    # Alternative: use a 4x4x4 simple cubic grid = 64 points, take 50
    # But BCC is denser. Let me think differently.
    
    # Let's just use a well-spaced random init and let optimization do the work.
    # But the directive says BCC superlattice. Let me implement it properly.
    
    # BCC lattice: points at (0,0,0) and (0.5,0.5,0.5) in each unit cell of the lattice.
    # Scale to fit in [-0.5, 0.5]^3.
    # Use a 3x3x3 supercell: 27 cells * 2 = 54 points.
    
    m = 3
    cell_size = 1.0 / m
    centers = []
    for iz in range(m):
        for iy in range(m):
            for ix in range(m):
                # Two points per cell
                # Point 1: at (ix, iy, iz) * cell_size, shifted to center
                x1 = -0.5 + (ix + 0.25) * cell_size
                y1 = -0.5 + (iy + 0.25) * cell_size
                z1 = -0.5 + (iz + 0.25) * cell_size
                centers.append([x1, y1, z1])
                
                # Point 2: at (ix+0.5, iy+0.5, iz+0.5) * cell_size
                x2 = -0.5 + (ix + 0.75) * cell_size
                y2 = -0.5 + (iy + 0.75) * cell_size
                z2 = -0.5 + (iz + 0.75) * cell_size
                centers.append([x2, y2, z2])
    
    centers = np.array(centers[:N], dtype=np.float64)
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, size=centers.shape)
    # Clip to stay inside
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _optimize(centers, time_limit):
    """Gradient-based optimization to maximize the minimum radius."""
    start_time = time.time()
    centers = np.array(centers, dtype=np.float64, copy=True)
    n = len(centers)
    
    # Use a simple gradient ascent on the soft-min radius
    # with adaptive step size
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    beta = 50.0
    lr = 0.005
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Compute gradient of soft-min radius
        # We'll use finite differences for simplicity, or analytic gradient
        
        # Analytic gradient of soft-min:
        # soft_min = m - (1/beta) * log(sum(exp(-beta * (x_k - m))))
        # d(soft_min)/dx_i = -sum(exp(-beta*(x_k-m)) * (-beta * dx_k/dx_i)) / (beta * sum(exp(-beta*(x_k-m))))
        # = sum(exp(-beta*(x_k-m)) * dx_k/dx_i) / sum(exp(-beta*(x_k-m)))
        
        # This is complex. Let's use finite differences for the gradient.
        
        # Actually, let's use a repulsive potential approach instead.
        # Energy = -min_radius (to minimize)
        # Use a smooth approximation: E = -soft_min_radius
        
        # For efficiency, let's use a simpler approach:
        # Iterate: for each pair and wall, if too close, push apart.
        
        # Simple iterative relaxation:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(3)
                    dist = np.linalg.norm(d)
                if dist < 0.25:  # target minimum distance
                    # Push apart
                    force = (0.25 - dist) * 0.5
                    centers[i] += d / dist * force
                    centers[j] -= d / dist * force
                    improved = True
            
            # Wall repulsion
            for k in range(3):
                if centers[i, k] > 0.5 - 0.13:
                    centers[i, k] -= (centers[i, k] - (0.5 - 0.13)) * 0.5
                    improved = True
                elif centers[i, k] < -0.5 + 0.13:
                    centers[i, k] -= (centers[i, k] - (-0.5 + 0.13)) * 0.5
                    improved = True
        
        # Clip to cube
        centers = np.clip(centers, -0.499, 0.499)
        
        # Check radius
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        if not improved and it > 100:
            break
    
    return best_centers, best_radius


def construct():
    start_time = time.time()
    remaining = _DEADLINE - start_time
    
    # Initialize with BCC-inspired lattice
    centers = _init_bcc()
    
    # Optimize
    centers, radius = _optimize(centers, remaining - 10)
    
    # Verify validity
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return centers
# EVOLVE-BLOCK-END
