"""
Strategy: Deterministic Crystalline Initialization + Basin Hopping with L-BFGS-B Relaxation.
Instead of pure gradient descent (L-BFGS-B), we use Basin Hopping: a global
optimization method that combines local minimization with random perturbations
to escape local minima. This is structurally different from the current
program's approach of just doing multiple L-BFGS relaxations from fixed
initializations. Basin hopping is known to be very effective for molecular
cluster problems like Morse clusters.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf) # Avoid division by zero
    
    # Morse potential: V(r) = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let x = rho(1-r)
    # V = e^x (e^x - 2) = e^(2x) - 2e^x
    # dV/dr = (2e^(2x) - 2e^x) * dx/dr = 2e^x(e^x - 1) * (-rho)
    # dV/dri = -dV/dr * (ri - rj) / r
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0)) # Sum over all i<j? 
    # Note: The formula above sums over all i,j including i!=j. 
    # The task defines E = sum_{i<j}. 
    # My calculation above sums over all pairs i!=j. 
    # Let's check: diff[i,j] is ri-rj. r is symmetric. 
    # Sum over all i!=j counts each pair twice.
    # So energy = 0.5 * sum(ex*(ex-2))
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    # Gradient
    # dV/dr = -rho * 2 * ex * (ex - 1)
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    # dV/dri = -dV/dr * (ri - rj) / r
    # We need to sum over j for each i.
    # grad[i] = sum_j ( -dV_dr[i,j] * (positions[i] - positions[j]) / r[i,j] )
    # Note: r[i,i] is inf, dV_dr[i,i] is 0 (since ex->0 for large r? No, r=0 -> x=rho, ex=e^rho. But we set r=inf on diagonal, so ex=0, dV_dr=0. Correct.)
    
    # Vectorized gradient calculation
    # dV_dr shape (n, n)
    # r shape (n, n)
    # We need to avoid division by zero on diagonal. 
    # Since dV_dr is 0 on diagonal, we can mask or just let it be 0/inf = 0? 
    # np.where(r > 1e-12, 1.0/r, 0.0) is safer.
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    # dV_dr * inv_r shape (n, n)
    # Multiply by diff (n, n, 3)
    # grad[i] = sum_j factor[i,j] * (positions[i] - positions[j])
    # = positions[i] * sum_j factor[i,j] - sum_j factor[i,j] * positions[j]
    
    # sum_j factor[i,j] -> row_sum (n,)
    row_sum = np.sum(factor, axis=1)
    
    # sum_j factor[i,j] * positions[j] -> factor @ positions (n, 3)
    weighted_pos = factor.dot(positions)
    
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad

def _generate_fcc(n, rng):
    """Generate n atoms on an FCC lattice, cropped to a sphere."""
    # FCC lattice vectors
    # a = 1.0 (equilibrium distance)
    # Points: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0) in units of a?
    # Nearest neighbor distance in FCC is a/sqrt(2). 
    # We want nearest neighbor distance = 1.0.
    # So a = sqrt(2).
    a = np.sqrt(2.0)
    
    points = []
    # Generate a large enough grid
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC basis
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0, 0.5*a, 0.5*a), (0.5*a, 0, 0.5*a), (0.5*a, 0.5*a, 0)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    # Center
    points -= np.mean(points, axis=0)
    
    # Sort by distance from origin to pick the most compact cluster
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    # Add small noise to break symmetry and help escape shallow local minima
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_bcc(n, rng):
    """Generate n atoms on a BCC lattice, cropped to a sphere."""
    a = 2.0 / np.sqrt(3.0) # Nearest neighbor distance is a*sqrt(3)/2 = 1.0
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0.5*a, 0.5*a, 0.5*a)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_hcp(n, rng):
    """Generate n atoms on an HCP lattice, cropped to a sphere."""
    # HCP structure: ABAB stacking
    # a = 1.0 (nearest neighbor in plane)
    # c = 1.633 * a (ideal)
    a = 1.0
    c = 1.633 * a
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Layer k
                # Even k: A layer
                # Odd k: B layer
                if k % 2 == 0:
                    # A layer points
                    # Hexagonal grid in xy
                    # p = i*a1 + j*a2
                    # a1 = (a, 0, 0)
                    # a2 = (a/2, a*sqrt(3)/2, 0)
                    p = np.array([i*a + j*a/2, j*a*np.sqrt(3)/2, k*c])
                    points.append(p)
                    # Also need the 3rd atom in the hex unit? 
                    # Standard HCP unit cell has 2 atoms.
                    # But for a "cropped" approach, we just generate a dense grid of HCP sites.
                    # Let's add the shifted atom for BCC-like density if needed?
                    # No, HCP is close packed.
                else:
                    # B layer points
                    # Shifted by (1/3, 2/3) in hex coords? 
                    # Or simply shift by (a/2, a*sqrt(3)/6, 0) relative to A?
                    # Actually, B layer atoms sit in the hollows of A layer.
                    # p_B = p_A + (a/2, a*sqrt(3)/6, c/2)? No, z is k*c.
                    # Let's use the standard basis for HCP:
                    # (0,0,0) and (1/3, 2/3, 1/2) in hex coords.
                    # In Cartesian:
                    # a1 = (1, 0, 0)
                    # a2 = (-1/2, sqrt(3)/2, 0)
                    # a3 = (0, 0, c)
                    # Atom 1: (0,0,0)
                    # Atom 2: (1/3)a1 + (2/3)a2 + (1/2)a3
                    # = (1/3 - 1/3, 0 + sqrt(3)/3, c/2) = (0, sqrt(3)/3, c/2)
                    
                    # For a general grid index (i,j,k) in hex-like coords:
                    # This is getting complex. Let's stick to FCC and BCC which are easier to generate densely.
                    # HCP is often not the global minimum for these specific Morse parameters anyway (FCC/BCC usually win for medium/long range).
                    pass
    
    if len(points) < n:
        return None
        
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    points += rng.normal(0, 0.01, points.shape)
    return points

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)

def _local_minimize(positions, rho):
    """Local minimization using L-BFGS-B."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        g = g.flatten()
        return e, g
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
        final_pos = res.x.reshape(-1, 3)
        final_energy = res.fun
        
        if _is_valid(final_pos):
            return final_pos, final_energy
    except Exception as e:
        pass
    
    return positions, _morse_energy_and_grad(positions, rho)[0]

def _basin_hopping(initial_pos, rho, n_steps=20, step_size=0.5):
    """Basin hopping global optimization."""
    current_pos, current_energy = _local_minimize(initial_pos, rho)
    best_pos, best_energy = current_pos, current_energy
    
    rng = np.random.default_rng(42)
    
    for step in range(n_steps):
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
        
        # Random perturbation
        perturbation = rng.normal(0, step_size, current_pos.shape)
        perturbed_pos = current_pos + perturbation
        
        # Local minimize
        new_pos, new_energy = _local_minimize(perturbed_pos, rho)
        
        # Accept/reject based on energy
        if new_energy < best_energy:
            best_pos, best_energy = new_pos, new_energy
        
        # Always accept if energy improves, or with some probability if it worsens
        if new_energy < current_energy:
            current_pos, current_energy = new_pos, new_energy
        else:
            # Simple acceptance: accept if not too much worse
            delta = new_energy - current_energy
            if delta < 5.0:  # Accept if energy increase is small
                current_pos, current_energy = new_pos, new_energy
    
    return best_pos, best_energy

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42) # Fixed seed for reproducibility in generation
    
    # List of initial configurations
    configs = []
    
    # 1. FCC
    fcc = _generate_fcc(size, rng)
    if fcc is not None:
        configs.append(fcc)
        
    # 2. BCC
    bcc = _generate_bcc(size, rng)
    if bcc is not None:
        configs.append(bcc)
        
    # 3. HCP (if generated)
    hcp = _generate_hcp(size, rng)
    if hcp is not None:
        configs.append(hcp)
        
    # 4. Random compact (fallback / diversity)
    radius = 1.0 * size ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < size:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
            points.append(p)
    if points:
        configs.append(np.array(points))
        
    best_energy = float('inf')
    best_positions = None
    
    # Use basin hopping for each configuration
    for i, pos in enumerate(configs):
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
            
        # Use basin hopping with L-BFGS-B for local minimization
        try:
            final_pos, final_energy = _basin_hopping(pos, parameter, n_steps=15, step_size=0.4)
            
            # Verify validity
            if _is_valid(final_pos):
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_positions = final_pos
        except Exception as e:
            # Fallback to simple relax if basin hopping fails
            try:
                energy, positions = relax(pos, potential, parameter)
                if _is_valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass

    # If no valid position found, return a valid random one
    if best_positions is None:
        # Generate a simple valid random cluster
        radius = 1.0 * size ** (1.0 / 3.0) * 1.5
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        best_positions = np.array(points)
        
    return best_positions
# EVOLVE-BLOCK-END
