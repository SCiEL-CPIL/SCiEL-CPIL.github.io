"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse potential energy for a configuration."""
    n = len(pos)
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                continue
            u = rho * (1.0 - r)
            eu = np.exp(u)
            e += eu * (eu - 2.0)
    return e


def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient for a configuration."""
    try:
        return morse_energy_gradient(pos, rho)
    except Exception:
        # Fallback: numerical gradient
        n = len(pos)
        e = _morse_energy(pos, rho)
        grad = np.zeros_like(pos)
        eps = 1e-6
        for i in range(n):
            for d in range(3):
                orig = pos[i, d]
                pos[i, d] = orig + eps
                ep = _morse_energy(pos, rho)
                pos[i, d] = orig - eps
                em = _morse_energy(pos, rho)
                pos[i, d] = orig
                grad[i, d] = (ep - em) / (2.0 * eps)
        return e, grad


def _lattice_init(n, rng, spacing=1.0):
    """Generate lattice-based initial configuration."""
    # Use a simple cubic lattice with some perturbation
    shell = int(np.ceil(n ** (1.0 / 3.0)))
    points = []
    for ix in range(shell):
        for iy in range(shell):
            for iz in range(shell):
                if len(points) < n:
                    points.append(np.array([ix, iy, iz]) * spacing)
    
    if len(points) < n:
        # Fill remaining with random points near the lattice
        for _ in range(n - len(points)):
            p = rng.uniform(-0.5, shell * spacing + 0.5, 3)
            points.append(p)
    
    points = np.array(points[:n])
    # Center the configuration
    points -= points.mean(axis=0)
    # Add small random perturbation to break symmetry
    points += rng.normal(0, 0.05, points.shape)
    return points


def _random_packed_init(n, rng, spacing=1.0):
    """Generate random packed initial configuration."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.6 * spacing
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, add remaining with minimal spacing
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def construct(instance):
    """Best of multiple initial structures, each optimized with L-BFGS-B."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if potential == "morse" else 1.0
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}[potential]
    rng = np.random.default_rng(hash((size, rho)) % (2**32))
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    # Generate diverse initial configurations
    inits = []
    # Lattice-based
    inits.append(_lattice_init(size, rng, spacing))
    # Random packed
    inits.append(_random_packed_init(size, rng, spacing))
    # More lattice variants with different perturbations
    for _ in range(2):
        inits.append(_lattice_init(size, rng, spacing))
    # Random variant
    inits.append(_random_packed_init(size, rng, spacing))
    
    best_energy = float('inf')
    best_positions = None
    
    for idx, init_pos in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        # First do a quick relax to get a reasonable starting point
        try:
            energy_relaxed, pos_relaxed = relax(init_pos.copy(), potential, parameter)
        except Exception:
            energy_relaxed = _morse_energy(init_pos, rho)
            pos_relaxed = init_pos.copy()
        
        # Now do L-BFGS-B optimization with gradient
        try:
            def fun_and_grad(x):
                pos = x.reshape(-1, 3)
                e, g = _morse_energy_grad(pos, rho)
                return e, g.reshape(-1)
            
            x0 = pos_relaxed.flatten()
            result = minimize(
                fun_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
            )
            
            if result.success or result.fun < best_energy:
                e_opt = result.fun
                pos_opt = result.x.reshape(-1, 3)
                
                # Verify minimum distance constraint
                min_d = float('inf')
                for i in range(len(pos_opt)):
                    for j in range(i+1, len(pos_opt)):
                        d = np.linalg.norm(pos_opt[i] - pos_opt[j])
                        min_d = min(min_d, d)
                
                if min_d >= 0.5:
                    if e_opt < best_energy:
                        best_energy = e_opt
                        best_positions = pos_opt
        except Exception:
            # If L-BFGS fails, use the relaxed position
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_positions = pos_relaxed
    
    # Final safety check
    if time.time() > deadline - 10.0 and best_positions is not None:
        # Do a final polish if time permits
        pass
    
    if best_positions is None:
        # Fallback: return the first relaxed configuration
        energy, positions = relax(inits[0], potential, parameter)
        return positions
    
    return best_positions
# EVOLVE-BLOCK-END
