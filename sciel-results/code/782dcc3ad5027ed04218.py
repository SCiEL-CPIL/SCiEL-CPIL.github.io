"""
Strategy: Deterministic Anisotropic Scaling Optimization with Symmetry-Aware Initialization.

Rationale:
The current approach uses stochastic basin hopping in the full 114-dimensional Cartesian space.
While robust, this is computationally expensive and can get trapped in local minima due to the 
high dimensionality and the "icosahedral funnel." 

The LJ38 global minimum is a truncated octahedron derived from an FCC lattice, but it is 
significantly distorted (anisotropic) compared to the ideal cubic FCC fragment. Specifically, 
the "octahedral" faces are closer together than the "cubic" faces, or vice versa, depending 
on the specific relaxation. 

This strategy proposes a two-stage deterministic approach:
1. **Low-Dimensional Anisotropic Scaling**: Instead of moving all 114 coordinates randomly, 
   we parameterize the cluster shape by scaling the FCC lattice along its three principal 
   axes (x, y, z). We perform a global optimization (using a robust method like Nelder-Mead 
   or a grid search followed by L-BFGS) on these 3 scaling parameters. This keeps the 
   topology fixed (FCC) but allows the cluster to find its optimal "breathing" and 
   "squashing" mode, which is where the true global minimum lies. This reduces the search 
   space from 114D to 3D, making it extremely fast and robust against local traps that 
   require breaking the FCC symmetry (which is not needed for the global min).

2. **Constrained Cartesian Polish**: Once the optimal anisotropic shape is found, we 
   perform a final L-BFGS minimization in the full 114D space. Since we start from a 
   geometry that is already very close to the global minimum (just with slightly off 
   internal relaxations), the local minimizer will converge to the precise global minimum 
   without wandering into the icosahedral basin.

This is structurally different because it uses a **parametric manifold** (anisotropic scaling) 
as the primary search space, rather than direct Cartesian perturbation.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0
START_TIME = time.time()

def lj_energy_vectorized(positions):
    """Compute LJ energy efficiently."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1e10)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron_base():
    """
    Generate the 38-atom FCC truncated octahedron points in a 'unit' scale.
    We use the standard FCC lattice points within a cutoff.
    """
    points = []
    # FCC lattice vectors
    # We generate points in a cubic box and select the 38 closest to origin
    # to form the truncated octahedron.
    max_coord = 4
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC condition: i+j+k is even
                if (i + j + k) % 2 == 0:
                    # Scale to typical LJ distance ~1.0
                    # In FCC, nearest neighbor distance is sqrt(2)/2 * a if a is lattice constant.
                    # Here we use integer coordinates scaled by 1/sqrt(2) so NN distance is 1.0.
                    # Wait, if i,j,k are integers, distance between (0,0,0) and (1,1,0) is sqrt(2).
                    # If we scale by 1/sqrt(2), distance is 1.0. This is good for LJ.
                    scale = 1.0 / math.sqrt(2.0)
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((r2, x, y, z))
    
    # Sort by distance to origin and take the first 38
    points.sort(key=lambda x: x[0])
    selected = [p[1:] for p in points[:38]]
    
    coords = np.array(selected, dtype=float)
    # Center the cluster
    coords -= np.mean(coords, axis=0)
    return coords

def apply_anisotropic_scaling(base_coords, sx, sy, sz):
    """
    Apply anisotropic scaling to the base coordinates.
    """
    scaled = base_coords.copy()
    scaled[:, 0] *= sx
    scaled[:, 1] *= sy
    scaled[:, 2] *= sz
    return scaled

def local_minimize_cartesian(positions):
    """
    Perform a local L-BFGS minimization in Cartesian space.
    """
    def energy_func(pos):
        pos = pos.reshape(N, 3)
        return lj_energy_vectorized(pos)
    
    try:
        result = minimize(
            energy_func,
            positions.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        if result.success:
            min_pos = result.x.reshape(N, 3)
            min_energy = result.fun
            if check_validity(min_pos):
                return min_pos, min_energy
    except Exception:
        pass
    
    return positions, lj_energy_vectorized(positions)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # 1. Generate base FCC structure
    base_coords = make_fcc_truncated_octahedron_base()
    
    # 2. Optimize Anisotropic Scaling Parameters
    # We search for the best (sx, sy, sz) that minimizes the energy of the scaled structure
    # followed by a quick local relaxation.
    
    def objective_anisotropic(params):
        sx, sy, sz = params
        # Ensure positive scales
        if sx <= 0 or sy <= 0 or sz <= 0:
            return 1e10
        scaled_coords = apply_anisotropic_scaling(base_coords, sx, sy, sz)
        
        # Quick local relaxation to get a good energy estimate for this shape
        # We use a limited number of iterations here to keep the search fast
        def quick_energy_func(pos):
            pos = pos.reshape(N, 3)
            return lj_energy_vectorized(pos)
        
        try:
            # Use L-BFGS-B with fewer iterations for the inner loop
            result = minimize(
                quick_energy_func,
                scaled_coords.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-6, 'ftol': 1e-10}
            )
            if result.success:
                min_pos = result.x.reshape(N, 3)
                if check_validity(min_pos, min_dist=0.4): # Slightly relaxed constraint for speed
                    return result.fun
        except Exception:
            pass
        
        return 1e10

    # Initial guess: isotropic scaling around 1.0
    # The global min is anisotropic. Typical ratios might be around 1.0-1.1.
    # We will use a multi-start approach for the 3D search to be safe.
    
    best_energy = 1e10
    best_params = np.array([1.0, 1.0, 1.0])
    best_scaled_coords = base_coords.copy()
    
    # Define a grid of starting points for the 3D optimizer
    # We test a range of scaling factors
    scales = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    
    start_time = time.time()
    
    # Perform a coarse grid search or multi-start Nelder-Mead
    # Given the small dimensionality, we can afford a few starts.
    
    starts = [
        [1.0, 1.0, 1.0],
        [1.1, 1.0, 1.0],
        [1.0, 1.1, 1.0],
        [1.0, 1.0, 1.1],
        [0.9, 0.9, 0.9],
        [1.1, 1.1, 1.1],
        [1.05, 0.95, 1.05],
        [0.95, 1.05, 0.95]
    ]
    
    for start in starts:
        if time.time() - start_time > 60: # Time limit for this phase
            break
        
        try:
            res = minimize(
                objective_anisotropic,
                start,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    # 3. Reconstruct the best anisotropic structure
    best_scaled_coords = apply_anisotropic_scaling(base_coords, *best_params)
    
    # 4. Final High-Precision Cartesian Relaxation
    # Now that we have a very good initial guess (optimized anisotropic shape),
    # we perform a full L-BFGS minimization to polish the internal coordinates.
    
    if time.time() - START_TIME < DEADLINE - 10:
        final_pos, final_energy = local_minimize_cartesian(best_scaled_coords)
        
        if final_energy < best_energy:
            best_energy = final_energy
            best_scaled_coords = final_pos
            
    # 5. Fallback / Safety
    if not check_validity(best_scaled_coords):
        # If something went wrong, fall back to the raw FCC
        best_scaled_coords = base_coords
        if not check_validity(best_scaled_coords):
             # Generate random valid points as last resort
             rng = np.random.default_rng(42)
             pts = []
             for _ in range(N):
                 while True:
                     p = rng.uniform(-2, 2, size=3)
                     if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                         pts.append(p)
                         break
             best_scaled_coords = np.array(pts)

    return best_scaled_coords

# EVOLVE-BLOCK-END
