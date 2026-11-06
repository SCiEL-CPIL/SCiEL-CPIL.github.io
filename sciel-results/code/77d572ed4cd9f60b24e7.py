"""Morse cluster global minima search with structured initialization and multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse potential energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    # Sum over i < j
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    return energy


def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    # Energy
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    # Gradient: dE/dr_ij = rho * ex * (2*ex - 2) / r_ij * (pos_i - pos_j)
    # More precisely: dE/dr = rho * exp(rho*(1-r)) * (2*exp(rho*(1-r)) - 2)
    # Then grad_i = sum_{j != i} dE/dr_ij * (pos_i - pos_j) / r_ij
    dE_dr = rho * ex * (2.0 * ex - 2.0)
    # Set diagonal to 0
    np.fill_diagonal(dE_dr, 0.0)
    # grad_i = sum_j dE_dr_ij * (pos_i - pos_j) / r_ij
    # = sum_j dE_dr_ij * diff_ij / r_ij
    # diff_ij = pos_i - pos_j
    inv_r = np.zeros_like(dists)
    mask = dists > 1e-10
    inv_r[mask] = 1.0 / dists[mask]
    # weight matrix: dE_dr * inv_r
    w = dE_dr * inv_r
    # grad = -sum_j w_ij * diff_ij ... let me be careful
    # E = sum_{i<j} V(r_ij), V(r) = exp(rho(1-r))(exp(rho(1-r))-2)
    # dV/dr = -rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1) * ... 
    # Let me redo: V(r) = e^x (e^x - 2), x = rho(1-r)
    # dV/dr = dV/dx * dx/dr = (2e^x - 2)e^x * (-rho) = -rho * e^x(2e^x - 2)
    # So dE/dpos_i = sum_{j != i} dV/dr_ij * (pos_i - pos_j)/r_ij
    # = sum_{j != i} [-rho * e^{x_ij}(2e^{x_ij}-2)] * (pos_i - pos_j)/r_ij
    # = sum_{j != i} rho * e^{x_ij}(2e^{x_ij}-2) * (pos_j - pos_i)/r_ij
    # Let's use: grad_i = sum_{j != i} dE_dr_ij * (pos_i - pos_j) / r_ij
    # where dE_dr_ij = dV/dr = -rho * ex * (2*ex - 2)
    # So grad_i = sum_j [-rho * ex_ij * (2*ex_ij - 2)] * (pos_i - pos_j) / r_ij
    # = -sum_j rho * ex_ij * (2*ex_ij - 2) * diff_ij / r_ij
    # = -sum_j w_ij * diff_ij where w_ij = rho * ex_ij * (2*ex_ij - 2) / r_ij
    # Using matrix: grad = - (W @ diff) summed over j for each i
    # diff is (n,n,3), W is (n,n)
    # grad_i = -sum_j W_ij * diff_ij
    # In numpy: grad = -np.einsum('ij,ijk->ik', w, diff)
    grad = -np.einsum('ij,ijk->ik', w, diff)
    return energy, grad


def _lattice_init(n, rng):
    """Generate a compact lattice-like initial configuration."""
    # Try to form a roughly spherical cluster
    # Use a simple approach: place atoms in a grid and keep those within radius
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                p = np.array([i, j, k], dtype=float)
                if len(points) < n:
                    points.append(p)
    if len(points) < n:
        # Add more points with smaller spacing
        side2 = int(np.ceil(n ** (1/3))) + 4
        for i in range(-side2, side2 + 1):
            for j in range(-side2, side2 + 1):
                for k in range(-side2, side2 + 1):
                    p = np.array([i * 0.5, j * 0.5, k * 0.5], dtype=float)
                    if len(points) < n:
                        points.append(p)
    points = np.array(points[:n])
    # Center
    points -= np.mean(points, axis=0)
    # Add small random perturbation
    points += rng.normal(0, 0.05, points.shape)
    return points


def _random_packed_init(n, rng, spacing=0.9):
    """Generate a random packed initial configuration."""
    radius = spacing * n ** (1/3) * 0.9
    points = []
    attempts = 0
    max_attempts = 10000
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < spacing:
                    ok = False
                    break
            if ok:
                points.append(p)
    if len(points) < n:
        # Fill remaining with random positions
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    points = np.array(points)
    return points


def _fcc_like_init(n, rng):
    """Generate an FCC-like initial configuration."""
    points = []
    # FCC lattice points
    a = 1.0
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                # FCC basis
                basis = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5, 0.5]),
                    np.array([0.5, 0, 0.5]),
                    np.array([0.5, 0.5, 0]),
                ]
                for b in basis:
                    p = a * (np.array([i, j, k]) + b)
                    if len(points) < n:
                        points.append(p)
    if len(points) < n:
        # Add random points
        while len(points) < n:
            p = rng.uniform(-3, 3, 3)
            points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.03, points.shape)
    return points


def _hcp_like_init(n, rng):
    """Generate an HCP-like initial configuration."""
    points = []
    a = 1.0
    c = 1.633 * a  # HCP c/a ratio
    for layer in range(-5, 6):
        z = layer * c / 2
        for i in range(-5, 6):
            for j in range(-5, 6):
                offset = 0.0 if layer % 2 == 0 else 0.5
                p = np.array([a * (i + offset), a * j * np.sqrt(3)/2, z])
                if len(points) < n:
                    points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-3, 3, 3)
            points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.03, points.shape)
    return points


def construct(instance):
    """Find global minima of Morse potential clusters."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s margin for 330s limit
    
    rng = np.random.default_rng(42 + n)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse initial configurations
    init_generators = [
        _lattice_init,
        _random_packed_init,
        _fcc_like_init,
        _hcp_like_init,
    ]
    
    # Try multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Choose initialization method
        gen_func = init_generators[restart % len(init_generators)]
        pos0 = gen_func(n, rng)
        
        # Optimize using L-BFGS-B with gradient
        try:
            result = minimize(
                fun=lambda x: _morse_energy_grad(x.reshape(-1, 3), rho)[0],
                x0=pos0.flatten(),
                jac=lambda x: _morse_energy_grad(x.reshape(-1, 3), rho)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-6, 'maxfun': 5000}
            )
            
            pos = result.x.reshape(-1, 3)
            energy = result.fun
            
            # Check minimum distance constraint
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist >= 0.5:
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
        except Exception:
            continue
    
    # If no valid solution found, use a simple fallback
    if best_pos is None:
        pos0 = _random_packed_init(n, rng)
        try:
            result = minimize(
                fun=lambda x: _morse_energy_grad(x.reshape(-1, 3), rho)[0],
                x0=pos0.flatten(),
                jac=lambda x: _morse_energy_grad(x.reshape(-1, 3), rho)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-5}
            )
            best_pos = result.x.reshape(-1, 3)
        except Exception:
            best_pos = pos0
    
    return best_pos
# EVOLVE-BLOCK-END
