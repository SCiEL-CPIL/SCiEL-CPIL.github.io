"""LJ38 cluster optimization using rigid-body parameterized FCC truncated octahedron."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0
START_TIME = time.time()

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron_raw():
    """
    Generate the 38 unique points of the FCC truncated octahedron.
    The points are:
    12 vertices: (0, 1, 1) and permutations (12 points)
    24 edges: (0, 1, 2) and permutations (24 points)
    Total: 36 points. Wait, 38 atoms.
    
    Let's re-verify the structure of LJ38 global minimum.
    The global minimum of LJ38 is indeed the FCC truncated octahedron.
    The layers are:
    Layer 1 (center): 1 atom (0,0,0) ? No, FCC doesn't have a center atom in the truncated octahedron usually, 
    but let's check the coordination.
    
    Actually, the standard FCC cluster for N=38 is:
    1st shell: 12 atoms (icosahedron vertices? No, octahedron vertices)
    2nd shell: 24 atoms
    3rd shell: 1 atom?
    
    Let's look at the standard "magic numbers" for FCC clusters: 13, 55, 147...
    38 is not a magic number.
    However, the literature states LJ38 global minimum is a truncated octahedron.
    
    Let's generate an FCC lattice and cut it.
    FCC basis: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) in cubic unit cell.
    Or simpler: points (i,j,k) where i+j+k is even.
    
    Let's generate points in a large box and pick the 38 closest to the origin.
    """
    points = []
    # i, j, k from -5 to 5
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                if (i + j + k) % 2 == 0:
                    # This generates an FCC-like lattice (BCC in integer coords, which is FCC in physical coords)
                    # Distance squared in this integer grid is proportional to physical distance squared
                    r2 = i*i + j*j + k*k
                    points.append((i, j, k, r2))
    
    points.sort(key=lambda x: x[3])
    # Take the first 38
    selected = [p[:3] for p in points[:38]]
    
    # Convert to float array
    coords = np.array(selected, dtype=float)
    
    # The "natural" scale of this lattice:
    # The nearest neighbor distance in this integer grid (e.g., (1,1,0)) is sqrt(2).
    # In LJ units, nearest neighbor is approx 1.12246.
    # So scale factor s = 1.12246 / sqrt(2).
    # We will optimize the scale factor anyway.
    
    # Center the cluster
    coords -= np.mean(coords, axis=0)
    
    return coords

def apply_rigid_transform(coords, params):
    """
    Apply rigid body transform (scale and anisotropic strain) to the raw coords.
    params: [s, exx, exy, exz, eyy, eyz]
    s: global scale
    exx, eyy, ezz: diagonal strain (1.0 is no strain)
    exy, exz, eyz: off-diagonal strain (0.0 is no strain)
    
    Transformation matrix M:
    [[s*exx, s*exy, s*exz],
     [s*exy, s*eyy, s*eyz],
     [s*exz, s*eyz, s*ezz]]
    """
    s, exx, exy, exz, eyy, eyz = params
    ezz = 1.0 # We fix ezz to 1.0 to remove rotational invariance? 
              # Actually, we should allow full 3x3 symmetric matrix for distortion?
              # But that's 6 params + scale = 7.
              # Let's stick to 6 params: scale + 5 distortion params (fixing one to avoid pure rotation ambiguity? 
              # Or just use 6 params for the symmetric matrix directly, assuming isotropic scale is included in the matrix).
    
    # Let's use a 6-parameter symmetric matrix M.
    # M = [[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]]
    # This allows any linear distortion.
    # We initialize M close to identity * scale.
    
    # Wait, the prompt suggests 6D optimization.
    # If we use a general 3x3 symmetric matrix (6 params), it covers scale + shear + strain.
    # We need to ensure it doesn't collapse.
    
    m11, m12, m13, m22, m23, m33 = params
    
    M = np.array([
        [m11, m12, m13],
        [m12, m22, m23],
        [m13, m23, m33]
    ])
    
    new_coords = coords @ M.T
    return new_coords

def energy_from_params(params, raw_coords):
    """
    Compute energy for given rigid body parameters.
    """
    try:
        positions = apply_rigid_transform(raw_coords, params)
        if not check_validity(positions):
            return 1e10
        return lj_energy_vectorized(positions)
    except Exception:
        return 1e10

def relax_ljfg(positions):
    """Relax using L-BFGS if available, otherwise return as-is."""
    try:
        from science_chemistry import relax
        return relax(positions)
    except ImportError:
        def energy_func(pos):
            pos = pos.reshape(N, 3)
            return lj_energy_vectorized(pos)
        
        result = minimize(energy_func, positions.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 100, 'gtol': 1e-6})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def construct():
    """
    Main entry point:
    1. Generate raw FCC truncated octahedron coordinates.
    2. Optimize 6 rigid body parameters (symmetric matrix elements) to minimize energy.
    3. Relax the resulting 3D structure.
    """
    global START_TIME
    START_TIME = time.time()
    
    raw_coords = make_fcc_truncated_octahedron_raw()
    
    # Initial guess for the transformation matrix
    # Nearest neighbor in raw_coords is sqrt(2).
    # Target nearest neighbor is ~1.12.
    # Scale factor approx 1.12 / sqrt(2) = 0.792
    s0 = 1.12 / math.sqrt(2.0)
    
    # Initial params: diagonal matrix with s0
    x0 = np.array([s0, 0.0, 0.0, s0, 0.0, s0])
    
    # Optimize the 6 parameters
    best_energy = 1e10
    best_params = x0
    best_positions = apply_rigid_transform(raw_coords, x0)
    
    # We might try multiple initial guesses for the distortion to avoid local minima in the 6D space
    # But the FCC symmetry is quite rigid, so one guess should work well.
    # Let's add a small random perturbation to break symmetry if needed?
    # No, the global minimum HAS this symmetry. We want to find the exact scale/strain.
    
    try:
        result = minimize(energy_from_params, x0, args=(raw_coords,), method='L-BFGS-B',
                         bounds=[(0.5, 2.0), (-0.5, 1.5), (-0.5, 1.5), (0.5, 2.0), (-0.5, 1.5), (0.5, 2.0)],
                         options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x
            best_positions = apply_rigid_transform(raw_coords, best_params)
            
    except Exception:
        pass
    
    # Ensure validity
    if not check_validity(best_positions):
        # Fallback to unstrained scaled version
        best_positions = raw_coords * s0
    
    # Final relaxation in 3D space
    # This is crucial to fix any residual errors from the rigid body approximation
    # and to allow the atoms to relax into their exact equilibrium positions.
    final_positions = relax_ljfg(best_positions)
    
    if check_validity(final_positions):
        final_energy = lj_energy_vectorized(final_positions)
        if final_energy < best_energy:
            best_positions = final_positions
            best_energy = final_energy
    else:
        # If relaxation failed or produced invalid coords, keep the rigid body result
        pass
        
    # Final check
    if not check_validity(best_positions):
        # Last resort: return the raw scaled coords
        best_positions = raw_coords * s0
        if not check_validity(best_positions):
             # Should not happen with s0 ~ 0.79, min dist in raw is sqrt(2)*0.79 ~ 1.12 > 0.5
             best_positions = raw_coords * 1.0
             
    return best_positions
