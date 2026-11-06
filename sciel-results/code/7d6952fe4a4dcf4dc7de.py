"""Morse cluster global minima finder using structural initialization and L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient for all pairs."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r_sq = np.sum(diff**2, axis=-1)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r_sq, 1.0)
    r = np.sqrt(r_sq)
    
    # Morse potential: V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    
    # Gradient: dV/dr = -2*rho*ex*(ex-1)
    dvdr = -2.0 * rho * ex * (ex - 1.0)
    
    # Only compute for i < j to avoid double counting
    iu = np.triu_indices(n, k=1)
    dvdr_iu = dvdr[iu]
    r_iu = r[iu]
    diff_iu = diff[iu]
    
    # Gradient contribution: dvdr * (pos_i - pos_j) / r
    grad = np.zeros_like(pos)
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        factor = dvdr_iu[idx] / r_iu[idx]
        grad[i] += factor * diff_iu[idx]
        grad[j] -= factor * diff_iu[idx]
    
    energy = np.sum(v[iu])
    return energy, grad

def _generate_fcc(n):
    """Generate FCC lattice points centered at origin."""
    points = []
    # Generate a large enough cubic grid
    side = int(np.ceil(n ** (1.0/3.0))) + 4
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                # FCC basis
                points.append([i, j, k])
                points.append([i+0.5, j+0.5, k])
                points.append([i+0.5, j, k+0.5])
                points.append([i, j+0.5, k+0.5])
    
    points = np.array(points)
    # Center
    points -= np.mean(points, axis=0)
    # Sort by distance from center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _generate_hcp(n):
    """Generate HCP lattice points centered at origin."""
    points = []
    a = 1.0
    c = a * np.sqrt(8.0/3.0)
    side = int(np.ceil(n ** (1.0/3.0))) + 4
    for l in range(-side, side+1):
        z = l * c
        for i in range(-side, side+1):
            for j in range(-side, side+1):
                if l % 2 == 0:
                    x = i * a
                    y = j * a * np.sqrt(3.0)/2.0
                else:
                    x = (i + 0.5) * a
                    y = j * a * np.sqrt(3.0)/2.0
                points.append([x, y, z])
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _generate_bcc(n):
    """Generate BCC lattice points centered at origin."""
    points = []
    side = int(np.ceil(n ** (1.0/3.0))) + 4
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                points.append([i, j, k])
                points.append([i+0.5, j+0.5, k+0.5])
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _generate_icosahedral(n):
    """Generate icosahedral shell points."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    points = []
    # Center
    if n >= 1:
        points.append([0,0,0])
        n -= 1
    
    # First shell (12)
    if n >= 12:
        points.extend(ico * 1.0)
        n -= 12
    
    # Second shell (30) - edges of icosahedron
    if n >= 30:
        # Approximate by scaling ico and adding rotated versions
        # For simplicity, just add more ico shells with scaling
        points.extend(ico * 1.8)
        n -= 12
        if n >= 12:
             points.extend(ico * 2.5)
             n -= 12
             
    # Fill remaining with random points on sphere
    if n > 0:
        rng = np.random.default_rng(42)
        for _ in range(n):
            p = rng.normal(size=3)
            p = p / np.linalg.norm(p) * 2.5
            points.append(p)
            
    return np.array(points[:int(np.sum(np.array([len(p) for p in [points]])))])[:len(points)] if False else np.array(points[:n])

def _ensure_min_dist(pos, min_dist=0.5, max_iter=100):
    """Nudge atoms to ensure minimum separation."""
    n = len(pos)
    for _ in range(max_iter):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d >= min_dist:
            break
        # Find closest pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        if i > j: i, j = j, i
        d = pos[i] - pos[j]
        norm = np.linalg.norm(d)
        if norm < 1e-10:
            d = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        direction = d / norm
        push = (min_dist - norm + 0.01) / 2.0
        pos[i] += push * direction
        pos[j] -= push * direction
    return pos

def construct(instance):
    """Find global minimum for Morse cluster using multi-start local optimization."""
    start_time = time.time()
    deadline = 290.0  # Safety margin
    
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    best_energy = np.inf
    best_positions = None
    
    # List of initial configurations
    inits = []
    
    # 1. FCC
    try:
        fcc = _generate_fcc(size)
        inits.append(fcc)
    except Exception:
        pass
        
    # 2. HCP
    try:
        hcp = _generate_hcp(size)
        inits.append(hcp)
    except Exception:
        pass
        
    # 3. BCC
    try:
        bcc = _generate_bcc(size)
        inits.append(bcc)
    except Exception:
        pass
        
    # 4. Icosahedral
    try:
        ico = _generate_icosahedral(size)
        inits.append(ico)
    except Exception:
        pass
        
    # 5. Random compact
    rng = np.random.default_rng(size)
    radius = size**(1.0/3.0) * 0.8
    random_pts = rng.uniform(-radius, radius, (size, 3))
    inits.append(random_pts)
    
    # Optimization function
    def objective_and_grad(x):
        p = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(p, rho)
        return e, g.flatten()
        
    def try_optimize(pos, max_iter=2000):
        nonlocal best_energy, best_positions
        pos = _ensure_min_dist(pos.copy())
        try:
            result = minimize(
                objective_and_grad,
                pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success or result.fun < best_energy:
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_positions = result.x.reshape(-1, 3).copy()
        except Exception:
            pass

    # Try each initial configuration
    for i, init in enumerate(inits):
        if time.time() - start_time > deadline:
            break
        if len(init) != size:
            continue
        try_optimize(init)
        
    # Basin hopping / Perturbation of best solution
    if best_positions is not None:
        for restart in range(50):
            if time.time() - start_time > deadline - 10:
                break
            rng_local = np.random.default_rng(1000 + restart)
            noise_scale = 0.2
            perturbed = best_positions + rng_local.normal(scale=noise_scale, size=(size, 3))
            try_optimize(perturbed, max_iter=1000)
            
    # Final fallback
    if best_positions is None:
        best_positions = inits[0] if inits else np.zeros((size, 3))
        
    return best_positions
