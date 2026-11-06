"""Morse cluster global minima search - Multi-start local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy(pos, rho):
    """Compute Morse potential energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)

def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient."""
    try:
        return morse_energy_gradient(pos, rho)
    except Exception:
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

def _ensure_min_dist(pos, min_d=0.51, max_iter=200):
    """Separate atoms that are too close using a simple iterative push."""
    pos = pos.copy()
    n = len(pos)
    if n < 2: return pos
    for _ in range(max_iter):
        min_dist = np.inf
        min_pair = (0, 1)
        # Vectorized min distance check is faster
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_val = np.min(dists)
        if min_val >= min_d:
            break
        # Find the pair
        min_pair = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair
        rij = pos[i] - pos[j]
        r = np.linalg.norm(rij)
        if r < 1e-10:
            rij = np.array([1.0, 0, 0])
            r = 1.0
        push = (min_d - r + 0.01) / 2.0
        pos[i] += push * rij / r
        pos[j] -= push * rij / r
    return pos

def _optimize_local(pos, rho, time_limit=5.0):
    """Optimize a configuration using L-BFGS-B with a time limit."""
    def fun_and_grad(x):
        p = x.reshape(-1, 3)
        e, g = _morse_energy_grad(p, rho)
        return e, g.reshape(-1)
    
    x0 = pos.flatten()
    start_time = time.time()
    try:
        result = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        if result.fun < 1e10:
            return result.fun, result.x.reshape(-1, 3)
    except Exception:
        pass
    return _morse_energy(pos, rho), pos

def _generate_fcc_init(n, scale=1.0):
    """Generate an FCC-like initial structure."""
    pts = []
    a = scale
    max_r = int(np.ceil(n ** (1/3.0))) + 2
    lattice_pts = []
    for i in range(-max_r, max_r+1):
        for j in range(-max_r, max_r+1):
            for k in range(-max_r, max_r+1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i+dx, j+dy, k+dz]) * a
                    lattice_pts.append(p)
    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    idx = np.argsort(dists)[:n]
    return lattice_pts[idx]

def _generate_ico_init(n, scale=1.0):
    """Generate an icosahedral shell initial structure."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    scale_factor = scale / (2.0 * np.sqrt(1 + 1/phi**2))
    
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v * scale_factor)
        
    faces = [
        [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
        [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
        [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
        [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
        [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
    ]
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        r = np.linalg.norm(center)
        if r > 1e-10:
            center = center / r * r * 1.3
            pts.append(center * scale_factor)
            
    # If we still need more, add random points on the surface
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            v = rng.normal(0, 1, 3)
            v = v / np.linalg.norm(v)
            pts.append(v * scale_factor * 1.4)
            
    return np.array(pts[:n])

def _generate_random_init(n, seed=0):
    """Generate a random close-packed initial structure."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0/3.0) * 0.8
    pos = rng.uniform(-radius, radius, (n, 3))
    return pos

def construct(instance):
    """Find low-energy Morse cluster configurations."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if potential == "morse" else 1.0
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse initial guesses
    inits = []
    
    # 1. FCC-like
    try:
        inits.append(_generate_fcc_init(size, scale=0.95))
    except: pass
    
    # 2. Icosahedral
    try:
        inits.append(_generate_ico_init(size, scale=0.95))
    except: pass
    
    # 3. Random
    try:
        inits.append(_generate_random_init(size, seed=123))
    except: pass
    
    # 4. Another Random
    try:
        inits.append(_generate_random_init(size, seed=456))
    except: pass
    
    # Evaluate and optimize each initial guess
    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Ensure min distance
        init_pos = _ensure_min_dist(init_pos, min_d=0.51)
        
        # Optimize
        try:
            energy, pos_opt = _optimize_local(init_pos, rho, time_limit=60)
            pos_opt = _ensure_min_dist(pos_opt, min_d=0.51)
            energy = _morse_energy(pos_opt, rho)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = pos_opt
        except Exception as e:
            # Fallback to initial if optimization fails
            energy = _morse_energy(init_pos, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = init_pos

    if best_pos is None:
        # Emergency fallback
        best_pos = _generate_random_init(size, seed=999)
        best_pos = _ensure_min_dist(best_pos, min_d=0.51)

    return best_pos
# EVOLVE-BLOCK-END
