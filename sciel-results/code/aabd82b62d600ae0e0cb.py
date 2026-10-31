"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient analytically."""
    pos_in = np.asarray(pos, dtype=np.float64)
    if pos_in.ndim == 1:
        pos = pos_in.reshape(N, 3)
    else:
        pos = pos_in
        
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 * r6
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Using full matrix, each pair counted twice
    e_mat = 4.0 * (r12 - r6)
    np.fill_diagonal(e_mat, 0.0)
    energy = 0.5 * np.sum(e_mat)
    # Gradient: dE/dr_i = 4 * sum_{j != i} (-12 r^-14 + 6 r^-8) * (r_i - r_j) / r
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    # factor = 4 * (-12 * r^-14 + 6 * r^-8) / r = 4 * (-12 * r^-15 + 6 * r^-9)
    factor = 4.0 * (-12.0 * r ** (-15) + 6.0 * r ** (-9))
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    if pos_in.ndim == 1:
        grad = grad.flatten()
        
    return energy, grad

def _check_validity(pos):
    """Check if positions are valid (atoms at least 0.5 apart)."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= 0.25  # (0.5)^2

def _make_fcc_initial_guess():
    """
    Create an initial guess based on FCC truncated octahedron structure.
    The LJ38 global minimum is a distorted FCC structure.
    """
    # FCC lattice vectors
    a = 1.12  # approximately the equilibrium distance
    # Generate FCC points in a cubic cell
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # In reduced units with lattice parameter a
    
    # For a truncated octahedron with FCC symmetry, we can generate points
    # Let's use a more direct approach: generate FCC points and select 38 closest to center
    
    # FCC points in a larger supercell
    n_cell = 4
    points = []
    for i in range(n_cell):
        for j in range(n_cell):
            for k in range(n_cell):
                # FCC lattice points
                points.append([i * a, j * a, k * a])
                points.append([i * a, (j + 0.5) * a, (k + 0.5) * a])
                points.append([(i + 0.5) * a, j * a, (k + 0.5) * a])
                points.append([(i + 0.5) * a, (j + 0.5) * a, k * a])
    
    points = np.array(points)
    # Center the cluster
    center = points.mean(axis=0)
    points -= center
    
    # Select 38 points closest to center
    dists = np.sum(points ** 2, axis=1)
    idx = np.argsort(dists)[:N]
    selected = points[idx]
    
    # Center again
    selected -= selected.mean(axis=0)
    
    return selected

def _relax_lj(pos, max_iter=200, tol=1e-10):
    """Relax using L-BFGS with analytic gradients."""
    pos_flat = pos.flatten()
    
    def objective(x):
        e, g = _lj_energy_and_grad(x)
        return e, g
    
    res = minimize(objective, pos_flat, method='L-BFGS-B', jac=True,
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8,
                            'maxls': 50})
    return res.x.reshape(N, 3), res.fun

def _basin_hop(pos, n_hops=50, step_size=0.3, max_iter_per_hop=100, tol=1e-10):
    """
    Basin hopping algorithm with L-BFGS relaxation.
    """
    best_pos = pos.copy()
    best_energy = _lj_energy_and_grad(pos)[0]
    rng = np.random.default_rng(42)
    
    current_pos = pos.copy()
    current_energy = best_energy
    
    for hop in range(n_hops):
        # Perturb
        perturbation = rng.normal(0, step_size, size=current_pos.shape)
        new_pos = current_pos + perturbation
        
        # Check validity
        if not _check_validity(new_pos):
            continue
        
        # Relax
        try:
            relaxed_pos, relaxed_energy = _relax_lj(new_pos, max_iter=max_iter_per_hop, tol=tol)
        except:
            continue
        
        if not _check_validity(relaxed_pos):
            continue
        
        # Metropolis acceptance
        if relaxed_energy < current_energy:
            current_pos = relaxed_pos
            current_energy = relaxed_energy
            if relaxed_energy < best_energy:
                best_energy = relaxed_energy
                best_pos = relaxed_pos.copy()
        else:
            # Accept with Metropolis criterion (simple: always accept if energy doesn't increase too much)
            delta_e = relaxed_energy - current_energy
            if delta_e < 0.1:  # small increase
                current_pos = relaxed_pos
                current_energy = relaxed_energy
    
    return best_pos, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate initial guess
    pos0 = _make_fcc_initial_guess()
    
    # First relaxation
    pos, energy = _relax_lj(pos0, max_iter=300, tol=1e-12)
    
    best_pos = pos.copy()
    best_energy = energy
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_pos
    
    # Basin hopping from multiple starting points
    n_attempts = 8
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Different initial guesses
        if attempt == 0:
            start_pos = pos0.copy()
        elif attempt == 1:
            # Icosahedral-like structure
            # Create an icosahedron with 12 vertices
            phi = (1 + np.sqrt(5)) / 2
            ico_points = np.array([
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                [0, phi, 1/phi], [0, phi, -1/phi],
                [0, -phi, 1/phi], [0, -phi, -1/phi],
                [phi, 1/phi, 0], [phi, -1/phi, 0],
                [-phi, 1/phi, 0], [-phi, -1/phi, 0],
                [1/phi, 0, phi], [1/phi, 0, -phi],
                [-1/phi, 0, phi], [-1/phi, 0, -phi],
            ], dtype=float)
            # Scale to appropriate size
            ico_points *= 0.55
            # Add more points to get 38
            rng = np.random.default_rng(attempt)
            extra = rng.normal(0, 0.2, size=(N - ico_points.shape[0], 3))
            start_pos = np.vstack([ico_points, extra])
            start_pos -= start_pos.mean(axis=0)
        else:
            # Perturbed FCC
            rng = np.random.default_rng(attempt * 100)
            start_pos = pos0 + rng.normal(0, 0.5, size=(N, 3))
            start_pos -= start_pos.mean(axis=0)
        
        # Relax
        try:
            pos, energy = _relax_lj(start_pos, max_iter=200, tol=1e-10)
        except:
            continue
        
        if not _check_validity(pos):
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
        
        # Basin hopping from this point if time permits
        if time.time() < deadline - 20.0:
            n_hops = max(1, int((deadline - time.time()) / 5.0))
            bh_pos, bh_energy = _basin_hop(pos, n_hops=n_hops, step_size=0.25, 
                                           max_iter_per_hop=80, tol=1e-10)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            final_pos, final_energy = _relax_lj(best_pos, max_iter=500, tol=1e-14)
            if _check_validity(final_pos) and final_energy < best_energy:
                best_pos = final_pos
                best_energy = final_energy
        except:
            pass
    
    # Ensure validity
    if not _check_validity(best_pos):
        # Fallback: return the original FCC guess
        best_pos = _make_fcc_initial_guess()
    
    return best_pos
# EVOLVE-BLOCK-END
