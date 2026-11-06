"""Morse cluster global minima search with basin hopping and multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
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
        eps = 1e-7
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


def _min_distance(pos):
    """Compute minimum pairwise distance in a configuration."""
    n = len(pos)
    if n < 2:
        return float('inf')
    tree = cKDTree(pos)
    pairs = tree.query_pairs(0.5 + 1e-6)
    min_d = float('inf')
    for i, j in pairs:
        d = np.linalg.norm(pos[i] - pos[j])
        min_d = min(min_d, d)
    if min_d == float('inf'):
        # No pairs within 0.5, check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                min_d = min(min_d, d)
    return min_d


def _fix_min_distance(pos, min_dist=0.5, max_iter=100):
    """Iteratively separate atoms that are too close."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(min_dist + 1e-6)
        moved = False
        for i, j in pairs:
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                # Push apart
                direction = pos[i] - pos[j]
                if np.linalg.norm(direction) < 1e-10:
                    direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction /= np.linalg.norm(direction)
                shift = (min_dist - d + 1e-3) / 2.0
                pos[i] += shift * direction
                pos[j] -= shift * direction
                moved = True
        if not moved:
            break
    return pos


def _lattice_init(n, rng, spacing=1.0):
    """Generate lattice-based initial configuration."""
    shell = int(np.ceil(n ** (1.0 / 3.0)))
    points = []
    for ix in range(shell):
        for iy in range(shell):
            for iz in range(shell):
                if len(points) < n:
                    points.append(np.array([ix, iy, iz]) * spacing)
    
    if len(points) < n:
        for _ in range(n - len(points)):
            p = rng.uniform(-0.5, shell * spacing + 0.5, 3)
            points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    points += rng.normal(0, 0.05, points.shape)
    return _fix_min_distance(points)


def _random_packed_init(n, rng, spacing=1.0):
    """Generate random packed initial configuration."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.55 * spacing
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
    
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return _fix_min_distance(points)


def _optimize_local(pos, rho, deadline, max_iter=300):
    """Run L-BFGS-B optimization on a single configuration."""
    try:
        def fun_and_grad(x):
            if time.time() > deadline - 5:
                raise ValueError("deadline")
            p = x.reshape(-1, 3)
            e, g = _morse_energy_grad(p, rho)
            return e, g.reshape(-1)
        
        x0 = pos.flatten()
        result = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1500}
        )
        
        if result.fun < float('inf'):
            pos_opt = result.x.reshape(-1, 3)
            pos_opt = _fix_min_distance(pos_opt)
            e_opt = _morse_energy(pos_opt, rho)
            if _min_distance(pos_opt) >= 0.5:
                return e_opt, pos_opt
    except Exception:
        pass
    
    # Fallback: use relaxed position
    try:
        e, p = relax(pos.copy(), 'morse', rho)
        p = _fix_min_distance(p)
        if _min_distance(p) >= 0.5:
            return _morse_energy(p, rho), p
    except Exception:
        pass
    
    return float('inf'), None


def _basin_hop(pos, rho, deadline, n_iter=10, step_size=0.1):
    """Simple basin hopping algorithm."""
    best_e, best_pos = _optimize_local(pos, rho, deadline, max_iter=200)
    
    if best_pos is None:
        return float('inf'), None
    
    for i in range(n_iter):
        if time.time() > deadline - 10:
            break
        
        # Perturb
        perturb = np.random.normal(0, step_size, best_pos.shape)
        new_pos = best_pos + perturb
        new_pos = _fix_min_distance(new_pos)
        
        # Optimize
        e_new, pos_new = _optimize_local(new_pos, rho, deadline, max_iter=150)
        
        if pos_new is not None and e_new < best_e:
            best_e, best_pos = e_new, pos_new
    
    return best_e, best_pos


def construct(instance):
    """Best of multiple initial structures with basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if potential == "morse" else 1.0
    
    rng = np.random.default_rng(hash((size, rho)) % (2**32))
    
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_lattice_init(size, rng, 1.0))
    inits.append(_random_packed_init(size, rng, 1.0))
    for _ in range(3):
        inits.append(_lattice_init(size, rng, 1.0))
    for _ in range(2):
        inits.append(_random_packed_init(size, rng, 1.0))
    
    best_energy = float('inf')
    best_positions = None
    
    # Phase 1: Quick local optimization on all inits
    for idx, init_pos in enumerate(inits):
        if time.time() > deadline - 60:
            break
        
        e, pos = _optimize_local(init_pos, rho, deadline, max_iter=200)
        if pos is not None and e < best_energy:
            best_energy = e
            best_positions = pos
    
    # Phase 2: Basin hopping from best configurations
    if best_positions is not None and time.time() < deadline - 30:
        # Basin hop from the best
        e_bh, pos_bh = _basin_hop(best_positions.copy(), rho, deadline, 
                                   n_iter=8, step_size=0.15)
        if pos_bh is not None and e_bh < best_energy:
            best_energy = e_bh
            best_positions = pos_bh
        
        # Also try basin hopping from second-best init
        if len(inits) > 1 and time.time() < deadline - 15:
            e_bh2, pos_bh2 = _basin_hop(inits[1].copy(), rho, deadline,
                                         n_iter=5, step_size=0.12)
            if pos_bh2 is not None and e_bh2 < best_energy:
                best_energy = e_bh2
                best_positions = pos_bh2
    
    # Final polish
    if best_positions is not None and time.time() < deadline - 5:
        e_final, pos_final = _optimize_local(best_positions.copy(), rho, deadline, max_iter=300)
        if pos_final is not None and e_final < best_energy:
            best_energy = e_final
            best_positions = pos_final
    
    if best_positions is None:
        # Fallback
        try:
            _, positions = relax(inits[0], potential, parameter)
            positions = _fix_min_distance(positions)
            return positions
        except Exception:
            return inits[0]
    
    return best_positions
# EVOLVE-BLOCK-END
