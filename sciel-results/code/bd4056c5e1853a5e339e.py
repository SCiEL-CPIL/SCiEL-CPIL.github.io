"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _lj_energy_vec(positions):
    """Compute total Lennard-Jones energy for a set of positions. Vectorized."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Avoid division by zero by setting diagonal to a large number
    np.fill_diagonal(r2, 1e10)
    
    # Check for very small distances to avoid overflow/instability
    # If r2 < 1e-10, the energy is huge.
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy


def _local_minimize(positions):
    """Perform local minimization using L-BFGS-B."""
    def objective(x):
        pos = x.reshape(-1, 3)
        # Penalty for violating minimum distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, 1e10)
        min_r2 = np.min(r2)
        if min_r2 < 0.25:
            # Strong penalty
            return 1e10 + (0.25 - min_r2) * 1000
        return _lj_energy_vec(pos)

    x0 = positions.flatten()
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success:
            return result.x.reshape(-1, 3), result.fun
        else:
            return positions, _lj_energy_vec(positions)
    except Exception:
        return positions, _lj_energy_vec(positions)


def construct():
    """
    Use Basin Hopping to find the global minimum.
    This approach explores the energy landscape by combining local minimization
    with random perturbations, which is more robust than deterministic geometry
    generation for escaping local minima like the icosahedral funnel.
    """
    start_time = time.time()
    time_limit = 115  # seconds, leaving buffer for overhead
    
    # Initial guess: Random cluster or distorted FCC
    # Let's start with a few random restarts to ensure we don't get stuck.
    
    best_positions = None
    best_energy = np.inf
    
    # We will run basin hopping for a fixed number of iterations or until time limit
    # Basin Hopping parameters
    kick_size = 0.5  # Initial kick magnitude
    n_iterations = 5000  # Max iterations
    
    # Start with a random configuration to avoid bias
    # Or start with FCC and perturb? Random is safer for "new approach" claim, 
    # but FCC is a good prior. Let's mix: Start with FCC, then BH.
    
    # Generate a decent initial structure: FCC truncated octahedron
    a = 1.0
    v1 = np.array([a/2, a/2, 0.0])
    v2 = np.array([a/2, 0.0, a/2])
    v3 = np.array([0.0, a/2, a/2])
    
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                p = i * v1 + j * v2 + k * v3
                points.append(p)
    points = np.array(points)
    dists = np.sum(points ** 2, axis=1)
    order = np.argsort(dists)
    points = points[order][:N]
    
    # Scale to reasonable distance (~1.0)
    # Nearest neighbor in this grid is ~0.707. Scale by 1.4 to get ~1.0
    current_positions = points * 1.4
    
    # Local minimize the initial guess
    current_positions, current_energy = _local_minimize(current_positions)
    
    if current_energy < best_energy:
        best_energy = current_energy
        best_positions = current_positions.copy()
    
    # Basin Hopping Loop
    temp = 1.0  # Initial temperature
    cooling = 0.99  # Cooling factor
    
    for i in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random kick
        kick = np.random.normal(0, kick_size, size=current_positions.shape)
        new_positions = current_positions + kick
        
        # Local minimize the kicked position
        new_positions, new_energy = _local_minimize(new_positions)
        
        # Check distance constraint for new_positions
        diff = new_positions[:, np.newaxis, :] - new_positions[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, 1e10)
        if np.min(r2) < 0.25:
            continue # Reject if invalid
        
        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / temp):
            current_positions = new_positions
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        # Cool down
        temp *= cooling
        
        # Adaptive kick size: reduce if we are finding better solutions
        if i > 0 and i % 100 == 0:
            kick_size *= 0.95
    
    # Final local minimize on the best found to ensure precision
    if best_positions is not None:
        final_positions, final_energy = _local_minimize(best_positions)
        # Verify constraint
        diff = final_positions[:, np.newaxis, :] - final_positions[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, 1e10)
        if np.min(r2) >= 0.25:
            best_positions = final_positions
            best_energy = final_energy
        else:
            # Fallback to previous valid best
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
