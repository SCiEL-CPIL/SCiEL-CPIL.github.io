"""Dzugutov cluster global minima search using continuous parameter annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

# Dzugutov potential parameters
A_const = 5.82
B_const = 1.28
a_const = 1.87
b_const = 1.94
c_const = 1.10
d_const = 0.27
m_const = 16.0


def _dzugutov_energy_annealed(positions, c_anneal, d_anneal):
    """Compute Dzugutov potential energy with annealed softness parameters."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a_const:
                energy += A_const * (r ** (-m_const) - B_const) * np.exp(c_anneal / (r - a_const))
            elif r < b_const:
                energy += B_const * np.exp(d_anneal / (r - b_const))
    return energy


def _dzugutov_energy_and_grad_annealed(positions, c_anneal, d_anneal):
    """Compute Dzugutov potential energy and gradient with annealed parameters."""
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            if r < 1e-12:
                continue
            if r < a_const:
                # V(r) = A*(r^-m - B)*exp(c/(r-a))
                # dV/dr = A*(-m*r^(-m-1) - B*c/(r-a)^2 * exp(c/(r-a)) + (r^-m - B)*exp(c/(r-a))*c/(r-a)^2)
                #      = A*exp(c/(r-a))*[-m*r^(-m-1) - B*c/(r-a)^2 + (r^-m - B)*c/(r-a)^2]
                #      = A*exp(c/(r-a))*[-m*r^(-m-1) + c/(r-a)^2 * (-B + r^-m - B)]
                #      = A*exp(c/(r-a))*[-m*r^(-m-1) + c/(r-a)^2 * (r^-m - 2B)]
                exp_term = np.exp(c_anneal / (r - a_const))
                dVdr = A_const * exp_term * (-m_const * r ** (-m_const - 1) + c_anneal / (r - a_const)**2 * (r ** (-m_const) - 2 * B_const))
            elif r < b_const:
                # V(r) = B*exp(d/(r-b))
                # dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
                exp_term = np.exp(d_anneal / (r - b_const))
                dVdr = B_const * exp_term * (-d_anneal / (r - b_const)**2)
            else:
                continue
            
            # grad_i += dV/dr * (r_ij / r), grad_j -= dV/dr * (r_ij / r)
            factor = dVdr / r
            grad[i] += factor * diff
            grad[j] -= factor * diff
            energy += A_const * (r ** (-m_const) - B_const) * np.exp(c_anneal / (r - a_const)) if r < a_const else B_const * np.exp(d_anneal / (r - b_const))
    
    return energy, grad


def _generate_icosahedron(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _build_polytetrahedral_seed(n, rng):
    """Build a polytetrahedral seed structure."""
    if n <= 13:
        core = np.vstack([np.zeros(3), _generate_icosahedron(1.0)])
        return core[:n]
    
    # Start with 13-atom icosahedron
    core = np.vstack([np.zeros(3), _generate_icosahedron(1.0)])
    points = list(core)
    
    remaining = n - len(points)
    shell_radius_base = 1.5
    
    for _ in range(remaining):
        shell_idx = min(3, (remaining // 12) + 1)
        shell_radius = shell_radius_base + shell_idx * 0.4
        
        u = rng.uniform(-1, 1)
        v = rng.uniform(-1, 1)
        theta = np.arccos(u)
        phi_angle = 2 * np.pi * v
        p = np.array([
            shell_radius * np.sin(theta) * np.cos(phi_angle),
            shell_radius * np.sin(theta) * np.sin(phi_angle),
            shell_radius * np.cos(theta)
        ])
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        
        if valid:
            points.append(p)
        else:
            for attempt in range(10):
                shell_radius_try = shell_radius + attempt * 0.15
                u = rng.uniform(-1, 1)
                v = rng.uniform(-1, 1)
                theta = np.arccos(u)
                phi_angle = 2 * np.pi * v
                p_try = np.array([
                    shell_radius_try * np.sin(theta) * np.cos(phi_angle),
                    shell_radius_try * np.sin(theta) * np.sin(phi_angle),
                    shell_radius_try * np.cos(theta)
                ])
                valid = True
                for q in points:
                    if np.linalg.norm(p_try - q) < 0.55:
                        valid = False
                        break
                if valid:
                    points.append(p_try)
                    break
            else:
                points.append(p)
    
    return np.array(points[:n])


def _anneal_and_optimize(positions, c_start, d_start, c_end, d_end, n_steps, deadline):
    """
    Perform continuous annealing of c and d parameters while optimizing positions.
    Returns (final_positions, final_energy).
    """
    current_pos = positions.copy()
    current_c = c_start
    current_d = d_start
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Linear annealing schedule
        t = step / max(1, n_steps - 1)
        current_c = c_start + (c_end - c_start) * t
        current_d = d_start + (d_end - d_start) * t
        
        # Optimize positions at current annealing parameters
        def objective_flat(x):
            pos = x.reshape(-1, 3)
            e, _ = _dzugutov_energy_and_grad_annealed(pos, current_c, current_d)
            return e
        
        def gradient_flat(x):
            pos = x.reshape(-1, 3)
            _, g = _dzugutov_energy_and_grad_annealed(pos, current_c, current_d)
            return g.ravel()
        
        x0 = current_pos.ravel()
        try:
            result = minimize(objective_flat, x0, jac=gradient_flat, method='L-BFGS-B',
                             options={'maxiter': 50, 'maxfun': 100, 'gtol': 1e-4})
            current_pos = result.x.reshape(-1, 3)
        except Exception:
            pass
        
        # Enforce minimum distance constraint
        for i in range(len(current_pos)):
            for j in range(i + 1, len(current_pos)):
                diff = current_pos[i] - current_pos[j]
                r = np.linalg.norm(diff)
                if r < 0.5:
                    if r < 1e-10:
                        # Random small displacement
                        current_pos[i] += rng.uniform(-0.1, 0.1, 3)
                    else:
                        correction = (0.5 - r) / r * diff
                        current_pos[i] += 0.5 * correction
                        current_pos[j] -= 0.5 * correction
    
    # Final energy with true parameters
    final_energy = _dzugutov_energy_annealed(current_pos, c_const, d_const)
    return current_pos, final_energy


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate multiple diverse seeds
    n_seeds = 4
    seeds = []
    for i in range(n_seeds):
        seed_rng = np.random.default_rng(size * 1000 + i)
        seed = _build_polytetrahedral_seed(size, seed_rng)
        seeds.append(seed)
    
    # Annealing parameters: start with very soft (large c, d) and anneal to true values
    c_start = 5.0  # Much softer than true value 1.10
    d_start = 2.0  # Much softer than true value 0.27
    c_end = c_const
    d_end = d_const
    
    n_anneal_steps = 20
    
    # Try annealing from each seed
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline:
            break
        
        try:
            final_pos, final_e = _anneal_and_optimize(seed, c_start, d_start, c_end, d_end, 
                                                      n_anneal_steps, deadline)
            if final_e < best_energy:
                best_energy = final_e
                best_positions = final_pos
        except Exception:
            pass
    
    # If we have time, try a few more seeds with different annealing schedules
    if time.time() < deadline - 20:
        for extra_idx in range(3):
            if time.time() > deadline:
                break
            
            seed_rng = np.random.default_rng(size * 10000 + extra_idx)
            seed = _build_polytetrahedral_seed(size, seed_rng)
            
            # Different annealing schedule
            c_start_alt = 3.0 + extra_idx * 0.5
            d_start_alt = 1.5 + extra_idx * 0.3
            n_steps_alt = 15
            
            try:
                final_pos, final_e = _anneal_and_optimize(seed, c_start_alt, d_start_alt, c_end, d_end,
                                                          n_steps_alt, deadline)
                if final_e < best_energy:
                    best_energy = final_e
                    best_positions = final_pos
            except Exception:
                pass
    
    # Final relaxation with the provided relax function
    if best_positions is not None:
        try:
            final_energy, final_positions = relax(best_positions, potential, None)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_positions
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback
        seed_rng = np.random.default_rng(size)
        best_positions = _build_polytetrahedral_seed(size, seed_rng)
        try:
            _, best_positions = relax(best_positions, potential, None)
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
