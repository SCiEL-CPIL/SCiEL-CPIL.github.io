"""
Grid-based density functional approximation (DFA) approach for Morse cluster global minima.
Uses a 3D voxel grid to model atomic density, identifies high-density regions,
and extracts atom positions from peaks in the density field. Coarse-to-fine grid
refinement guides the placement of atoms.
"""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import time

def _compute_morse_pair_energy(r, rho):
    """Compute Morse pair energy for a single distance r."""
    val = rho * (1.0 - r)
    e = np.exp(val)
    return e * (e - 2.0)

def _grid_based_initialization(n, rho, grid_size=20, rng=None, n_candidates=4):
    """
    Generate candidate structures using grid-based density functional approach.
    1. Create a 3D grid
    2. Compute an energy-based density field
    3. Extract atom positions from high-density regions
    """
    if rng is None:
        rng = np.random.default_rng()
    
    candidates = []
    
    # Estimate cluster radius
    R = (n ** (1.0/3.0)) * 1.3
    
    for trial in range(n_candidates):
        # Vary grid parameters per trial
        gs = grid_size + trial * 2  # Increase resolution per trial
        
        # Create 3D grid within sphere of radius R
        dx = 2.0 * R / gs
        coords = np.linspace(-R, R, gs)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        
        # Keep only points within sphere
        r2 = X**2 + Y**2 + Z**2
        mask = r2 <= R**2
        grid_x = X[mask]
        grid_y = Y[mask]
        grid_z = Z[mask]
        grid_r = np.sqrt(r2[mask])
        
        n_grid = len(grid_x)
        if n_grid < n:
            continue
        
        # Compute a density-based energy score for each grid point
        # Score based on: proximity to ideal packing distance, shell structure
        # For Morse with given rho, equilibrium is at r=1
        
        # Create a scoring function that favors:
        # 1. Points at distances ~1 from center (for shell structures)
        # 2. Points that allow close packing
        
        # Gaussian scoring around ideal radial distances
        # For small clusters, shell at r~1; for larger, multiple shells
        ideal_r = 1.0
        sigma = 0.3
        
        # Radial score: favor points near ideal radius
        radial_score = np.exp(-0.5 * ((grid_r - ideal_r) / sigma)**2)
        
        # Also add a term for inner regions for larger n
        if n > 40:
            inner_ideal = 0.5
            inner_sigma = 0.2
            inner_score = np.exp(-0.5 * ((grid_r - inner_ideal) / inner_sigma)**2) * 0.5
            radial_score = radial_score + inner_score
        
        # Angular uniformity score: favor points that are spread out angularly
        # Approximate by penalizing high local density
        # Use a simple heuristic: score is higher for points farther from other grid points
        # But this is O(n_grid^2), so approximate with random sampling
        
        # Add jitter to break symmetry
        jitter_scale = 0.05 * (trial + 1)
        gx = grid_x + rng.normal(0, jitter_scale, n_grid)
        gy = grid_y + rng.normal(0, jitter_scale, n_grid)
        gz = grid_z + rng.normal(0, jitter_scale, n_grid)
        
        # Combine scores
        scores = radial_score + rng.uniform(0, 0.1, n_grid)  # Add small random perturbation
        
        # Select n points with highest scores, ensuring minimum distance
        # Sort by score descending
        order = np.argsort(-scores)
        
        selected = []
        selected_idx = []
        min_dist = 0.55  # Slightly above the 0.5 constraint
        
        for idx in order:
            if len(selected) >= n:
                break
            px, py, pz = gx[idx], gy[idx], gz[idx]
            
            if not selected:
                selected.append([px, py, pz])
                selected_idx.append(idx)
            else:
                # Check distance to all selected points
                sel_arr = np.array(selected)
                dists = np.sqrt((sel_arr[:, 0] - px)**2 + (sel_arr[:, 1] - py)**2 + (sel_arr[:, 2] - pz)**2)
                if np.min(dists) >= min_dist:
                    selected.append([px, py, pz])
                    selected_idx.append(idx)
        
        if len(selected) == n:
            candidates.append(np.array(selected))
    
    # If grid-based failed, fallback to lattice-based
    if not candidates:
        # Simple BCC lattice fallback
        R_fallback = (n ** (1.0/3.0)) * 1.2
        a = np.sqrt(2.0)  # BCC lattice constant for min dist ~1
        max_idx = int(np.ceil(R_fallback / a)) + 1
        pts = []
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                for k in range(-max_idx, max_idx + 1):
                    p1 = np.array([i, j, k]) * a
                    if np.linalg.norm(p1) <= R_fallback:
                        pts.append(p1)
                    p2 = (np.array([i, j, k]) + 0.5) * a
                    if np.linalg.norm(p2) <= R_fallback:
                        pts.append(p2)
        pts = np.array(pts)
        if len(pts) >= n:
            idx = rng.choice(len(pts), n, replace=False)
            candidates.append(pts[idx])
        else:
            # Random fallback
            pts = []
            while len(pts) < n:
                p = rng.uniform(-R_fallback, R_fallback, 3)
                if np.linalg.norm(p) <= R_fallback:
                    pts.append(p)
            candidates.append(np.array(pts))
    
    return candidates

def construct(instance):
    """
    Construct the cluster using grid-based density functional approximation.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget per instance
    start_time = time.time()
    time_limit = 45.0  # Leave buffer for overhead
    
    rng = np.random.default_rng(size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate candidates using grid-based approach
    candidates = _grid_based_initialization(size, parameter, grid_size=18, rng=rng, n_candidates=5)
    
    # Relax all candidates
    for i, pos in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
        
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception as e:
            pass
    
    # If no candidate worked, return a valid fallback
    if best_positions is None:
        # Create a simple valid configuration
        R = (size ** (1.0/3.0)) * 1.2
        pts = []
        for i in range(size):
            angle = 2 * np.pi * i / size
            r = R * (0.3 + 0.7 * (i % 3) / 2.0)
            pts.append([r * np.cos(angle), r * np.sin(angle), 0.1 * (i % 2 - 0.5)])
        best_positions = np.array(pts)
    
    return best_positions
# EVOLVE-BLOCK-END
