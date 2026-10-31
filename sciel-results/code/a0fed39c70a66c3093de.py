"""
LJ38 cluster global minimum search.

Strategy:
  1. Build the known FCC truncated-octahedron geometry (the putative global minimum).
  2. Use Basin Hopping to explore the energy landscape, escaping local minima 
     (like the icosahedral funnel) by combining random perturbations with local relaxation.
  3. Apply a high-precision L-BFGS-B polish to the best candidate.
  4. Return the polished coordinates.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 38
DEADLINE = 115.0  # Seconds, safety margin for 120s limit

def _lj_energy_and_grad(pos):
    """
    Total LJ energy and gradient for positions pos (N,3).
    """
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero on diagonal
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy: sum over all pairs
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    # dE/dx_i = 4 * sum_{j != i} (x_i - x_j) * (-12 / r^14 + 6 / r^8)
    # Using r2:
    # r^-14 = (r2)^-7
    # r^-8  = (r2)^-4
    # coeff = 4 * (6 * (r2)^-4 - 12 * (r2)^-7)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron geometry for LJ38.
    """
    # Generate FCC points
    max_coord = 5
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((float(i), float(j), float(k)))
    
    points = np.array(points)
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    
    # Sort by distance from origin
    dists = np.sum(points ** 2, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Re-center after selection
    points -= np.mean(points, axis=0)
    
    # Scale so that nearest-neighbor distance = 2^(1/6)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    if min_r > 1e-10:
        scale = 2.0 ** (1.0 / 6.0) / min_r
        points = points * scale
        
    return points

def _relax(pos, max_iter=1000, gtol=1e-8, ftol=1e-12):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        # This is a hard constraint violation. 
        # Scaling is a simple fix, but might be bad for energy.
        # However, the evaluator checks this.
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def construct():
    """
    Construct the 38-atom LJ cluster at (or very near) the global minimum.
    Returns array of shape (38, 3).
    """
    start_time = time.time()
    
    # 1. Initial Guess: FCC Truncated Octahedron
    coords = _make_fcc_truncated_octahedron()
    coords = _ensure_min_sep(coords)
    
    # Initial Relaxation
    best_coords, best_energy = _relax(coords, max_iter=1000, gtol=1e-8, ftol=1e-12)
    
    # 2. Basin Hopping
    # We use a simple basin hopping algorithm:
    # - Perturb the current best configuration with Gaussian noise
    # - Relax the perturbed configuration
    # - Accept if energy is lower (or with Metropolis probability)
    
    # Parameters for Basin Hopping
    # Step size for perturbation. 
    # Typical LJ equilibrium distance is ~1.12 (2^(1/6)).
    # A step of 0.5 is reasonable to explore neighbors without breaking the structure too much.
    step_size = 0.5 
    T = 1.0 # Temperature for Metropolis acceptance
    
    # We will run as many iterations as possible until the deadline.
    # Each iteration involves a relaxation which takes ~0.1-0.5s depending on convergence.
    
    while time.time() - start_time < DEADLINE - 10.0: # Leave 10s for final polish
        
        # Perturb
        noise = np.random.normal(0, step_size, size=best_coords.shape)
        perturbed = best_coords + noise
        
        # Relax
        new_coords, new_energy = _relax(perturbed, max_iter=500, gtol=1e-6, ftol=1e-10)
        
        # Check separation constraint
        new_coords = _ensure_min_sep(new_coords)
        # Re-evaluate energy after ensuring separation (scaling changes energy)
        new_energy, _ = _lj_energy_and_grad(new_coords)
        
        # Acceptance Criterion
        delta_e = new_energy - best_energy
        if delta_e < 0:
            # Better energy, accept
            best_coords = new_coords
            best_energy = new_energy
            # Optionally reduce step size or temperature if we want to converge faster
            # But for global search, keeping it constant is standard.
        else:
            # Worse energy, accept with probability exp(-delta_e / T)
            if np.random.random() < np.exp(-delta_e / T):
                best_coords = new_coords
                best_energy = new_energy
                # If we accept a bad move, we might want to increase step size to escape
                # or keep it as is. Standard BH often keeps step size constant or 
                # adapts it based on acceptance rate. Let's keep it simple.
                
        # Adaptive Step Size (Simple version):
        # If we just accepted a good move, maybe reduce step size slightly to refine?
        # If we are stuck, increase it?
        # For now, constant step size is robust enough for this short time budget.
        
    # 3. Final High-Precision Polish
    if time.time() - start_time < DEADLINE - 2.0:
        polished_coords, polished_energy = _relax(
            best_coords, 
            max_iter=5000, 
            gtol=1e-12, 
            ftol=1e-16
        )
        polished_coords = _ensure_min_sep(polished_coords)
        # Verify energy of polished coords
        final_energy, _ = _lj_energy_and_grad(polished_coords)
        
        if final_energy < best_energy:
            best_coords = polished_coords
            best_energy = final_energy

    return best_coords

# For testing
if __name__ == "__main__":
    import time
    start = time.time()
    pos = construct()
    end = time.time()
    e, g = _lj_energy_and_grad(pos)
    print(f"Time: {end-start:.2f}s")
    print(f"Energy: {e}")
    print(f"Min Dist: {np.min([np.linalg.norm(pos[i]-pos[j]) for i in range(N) for j in range(i+1,N)])}")
