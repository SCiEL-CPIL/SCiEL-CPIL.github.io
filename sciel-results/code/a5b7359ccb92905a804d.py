"""Morse cluster global minima search with multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse potential energy: sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    # Sum over all pairs, divide by 2 since we count each pair twice
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    return energy


def _lattice_init(n, lattice_type='fcc', spacing=1.0):
    """Generate a cluster from a lattice fragment."""
    points = []
    if lattice_type == 'fcc':
        # FCC lattice vectors
        a = spacing
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    # FCC: corners + face centers
                    for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        p = np.array([i, j, k]) * a + np.array(offset) * a
                        points.append(p)
    elif lattice_type == 'bcc':
        a = spacing
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    for offset in [(0,0,0), (0.5,0.5,0.5)]:
                        p = np.array([i, j, k]) * a + np.array(offset) * a
                        points.append(p)
    else:  # simple cubic
        a = spacing
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    p = np.array([i, j, k]) * a
                    points.append(p)
    
    points = np.array(points)
    # Center
    points -= points.mean(axis=0)
    
    # Sort by distance from center and take first n
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    return points


def _random_packed_init(n, spacing=1.0, rng=None):
    """Generate random packed sphere configuration."""
    if rng is None:
        rng = np.random.default_rng()
    
    points = []
    radius = spacing * n ** (1.0/3.0) * 0.9
    min_dist = 0.55 * spacing  # Slightly tighter than 0.5 to allow relaxation
    
    attempts = 0
    max_attempts = 50000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't pack enough, add random points with smaller min_dist
    if len(points) < n:
        min_dist2 = 0.51 * spacing
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist2 for q in points):
                    points.append(p)
            attempts += 1
    
    # If still not enough, just add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _optimize(pos, rho, method='relax', maxiter=2000):
    """Optimize a configuration using the specified method."""
    try:
        if method == 'relax':
            energy, positions = relax(pos, 'morse', rho)
            return energy, positions
        elif method == 'lfgs':
            # Use scipy L-BFGS-B with gradient
            def fun_and_grad(x):
                pos_2d = x.reshape(-1, 3)
                e, g = morse_energy_gradient(pos_2d, rho)
                return e, g.reshape(-1)
            
            result = minimize(fun_and_grad, pos.flatten(), method='L-BFGS-B',
                             jac=True, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
            positions = result.x.reshape(-1, 3)
            return result.fun, positions
        elif method == 'bfgs':
            def fun_and_grad(x):
                pos_2d = x.reshape(-1, 3)
                e, g = morse_energy_gradient(pos_2d, rho)
                return e, g.reshape(-1)
            
            result = minimize(fun_and_grad, pos.flatten(), method='BFGS',
                             jac=True, options={'maxiter': maxiter, 'gtol': 1e-10})
            positions = result.x.reshape(-1, 3)
            return result.fun, positions
        else:
            return _morse_energy(pos, rho), pos
    except Exception:
        return _morse_energy(pos, rho), pos


def construct(instance):
    """Find global minimum of Morse cluster."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(size)
    inits = []
    
    # Lattice-based inits (FCC, BCC, SC)
    for lt in ['fcc', 'bcc', 'sc']:
        try:
            init = _lattice_init(size, lt, spacing=1.0)
            # Add small random perturbation
            init = init + rng.normal(0, 0.02, init.shape)
            inits.append(init)
        except Exception:
            pass
    
    # Random packed inits
    for _ in range(6):
        try:
            init = _random_packed_init(size, spacing=1.0, rng=rng)
            inits.append(init)
        except Exception:
            pass
    
    # Compact random inits
    radius = 1.0 * size ** (1.0/3.0) * 0.7
    for _ in range(4):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        if len(points) == size:
            inits.append(np.array(points))
    
    # Optimize each initial configuration with multiple methods
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Method 1: built-in relax
        e1, p1 = _optimize(init, rho, method='relax')
        if e1 < best_energy:
            best_energy = e1
            best_positions = p1
        
        # Method 2: L-BFGS-B
        if time.time() > deadline:
            break
        e2, p2 = _optimize(init, rho, method='lfgs', maxiter=1000)
        if e2 < best_energy:
            best_energy = e2
            best_positions = p2
        
        # Method 3: BFGS
        if time.time() > deadline:
            break
        e3, p3 = _optimize(init, rho, method='bfgs', maxiter=1000)
        if e3 < best_energy:
            best_energy = e3
            best_positions = p3
    
    # Final polish: take best and refine with tighter tolerance
    if best_positions is not None and time.time() < deadline - 10:
        # Re-relax with built-in
        e, p = _optimize(best_positions, rho, method='relax')
        if e < best_energy:
            best_energy = e
            best_positions = p
        
        # L-BFGS-B polish
        if time.time() < deadline - 5:
            e, p = _optimize(best_positions, rho, method='lfgs', maxiter=3000)
            if e < best_energy:
                best_energy = e
                best_positions = p
        
        # BFGS polish
        if time.time() < deadline - 2:
            e, p = _optimize(best_positions, rho, method='bfgs', maxiter=3000)
            if e < best_energy:
                best_energy = e
                best_positions = p
    
    # Ensure minimum distance constraint
    if best_positions is not None:
        # Verify and fix any atoms too close
        for _ in range(10):
            fixed = False
            for i in range(len(best_positions)):
                for j in range(i+1, len(best_positions)):
                    d = np.linalg.norm(best_positions[i] - best_positions[j])
                    if d < 0.5:
                        # Push apart
                        mid = (best_positions[i] + best_positions[j]) / 2
                        dir_vec = best_positions[i] - best_positions[j]
                        if np.linalg.norm(dir_vec) < 1e-10:
                            dir_vec = np.array([1.0, 0.0, 0.0])
                        dir_vec = dir_vec / np.linalg.norm(dir_vec)
                        shift = (0.5 - d) / 2 * 1.01
                        best_positions[i] += dir_vec * shift
                        best_positions[j] -= dir_vec * shift
                        fixed = True
            if not fixed:
                break
        # Final energy check
        best_energy = _morse_energy(best_positions, rho)
    
    if best_positions is None:
        # Fallback: simple lattice
        best_positions = _lattice_init(size, 'fcc', spacing=1.0)
    
    return best_positions
# EVOLVE-BLOCK-END
