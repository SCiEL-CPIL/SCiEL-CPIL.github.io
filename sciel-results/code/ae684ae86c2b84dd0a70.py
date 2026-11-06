import numpy as np
import time
import scipy.optimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_GLOBAL_START_TIME = None

def _get_time_budget():
    """Calculate remaining time budget for the current call."""
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        return 300.0
    elapsed = time.time() - _GLOBAL_START_TIME
    # Safety margin of 5 seconds
    return max(0.0, 300.0 - elapsed - 5.0)

def _min_distance_check(pos):
    """Check if all pairwise distances are at least 0.5."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _generate_fcc_cluster(n, rng):
    """Generate an FCC cluster of approximately n atoms, centered."""
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for b in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i, j, k]) + np.array(b)
                    pts.append(p)
    
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def _generate_bcc_cluster(n, rng):
    """Generate a BCC cluster of approximately n atoms, centered."""
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k+0.5])
                
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def _generate_random_packed(n, rng):
    """Generate a random packed cluster using rejection sampling."""
    radius = n ** (1.0/3.0) * 1.2
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in pts):
                pts.append(p)
        attempts += 1
        
    if len(pts) < n:
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p)
            
    return np.array(pts)

def _relax_hybrid(pos, potential, rho, timeout=5.0):
    """
    Hybrid relaxation: 
    1. Use built-in relax.
    2. Polish with L-BFGS-B using the provided gradient.
    """
    start_time = time.time()
    
    # Step 1: Built-in relax
    try:
        energy, positions = relax(pos, potential, rho)
    except Exception:
        energy, positions = None, pos
        
    if energy is None:
        return np.inf, pos

    # Check time
    if time.time() - start_time > timeout * 0.5:
        return energy, positions

    # Step 2: L-BFGS-B polish
    def objective(x):
        pos_2d = x.reshape(-1, 3)
        if not _min_distance_check(pos_2d):
            return 1e10
        try:
            e, _ = morse_energy_gradient(pos_2d, rho)
            return e[0] if isinstance(e, tuple) else e
        except Exception:
            return 1e10

    def gradient(x):
        pos_2d = x.reshape(-1, 3)
        if not _min_distance_check(pos_2d):
            return np.zeros_like(x)
        try:
            e, g = morse_energy_gradient(pos_2d, rho)
            return g.flatten() if isinstance(g, np.ndarray) else np.zeros_like(x)
        except Exception:
            return np.zeros_like(x)

    try:
        result = scipy.optimize.minimize(
            objective, 
            positions.flatten(), 
            method='L-BFGS-B', 
            jac=gradient,
            options={'maxiter': 50, 'gtol': 1e-6, 'ftol': 1e-10}
        )
        
        if result.success and result.fun < energy:
            # Verify distance constraint
            final_pos = result.x.reshape(-1, 3)
            if _min_distance_check(final_pos):
                return result.fun, final_pos
    except Exception:
        pass
        
    return energy, positions

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Strategy: Basin Hopping with Hybrid Relaxation and Global Time Management.
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    # Get time budget for this specific call
    time_budget = _get_time_budget()
    if time_budget <= 0:
        # Fallback: return a valid random config
        rng = np.random.default_rng(42 + n)
        radius = n ** (1.0/3.0) * 1.2
        pts = []
        for _ in range(n):
            p = rng.uniform(-radius, radius, 3)
            pts.append(p)
        return np.array(pts)

    deadline = time.time() + time_budget
    
    rng = np.random.default_rng(42 + n)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. FCC
    try:
        cfg = _generate_fcc_cluster(n, rng)
        if _min_distance_check(cfg):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 2. BCC
    try:
        cfg = _generate_bcc_cluster(n, rng)
        if _min_distance_check(cfg):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 3. Random Packed
    try:
        cfg = _generate_random_packed(n, rng)
        if _min_distance_check(cfg):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 4. FCC with larger noise
    try:
        cfg = _generate_fcc_cluster(n, rng)
        cfg += rng.normal(0, 0.2, size=cfg.shape)
        if _min_distance_check(cfg):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 5. Random Packed with different seed
    try:
        cfg = _generate_random_packed(n, rng)
        if _min_distance_check(cfg):
            initial_configs.append(cfg)
    except Exception:
        pass

    if not initial_configs:
        cfg = rng.uniform(-n**0.33, n**0.33, size=(n, 3))
        initial_configs.append(cfg)

    # Relax all initial configurations to find the best starting point
    best_energy = np.inf
    best_positions = initial_configs[0]
    current_energy = np.inf
    current_positions = initial_configs[0]
    
    for cfg in initial_configs:
        if time.time() > deadline - 10.0:
            break
        try:
            energy, positions = _relax_hybrid(cfg, potential, rho, timeout=5.0)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                current_energy = energy
                current_positions = positions.copy()
        except Exception:
            continue

    # Basin Hopping
    # Adaptive kick size
    kick_sigma = 0.1
    T = 0.1 # Initial temperature
    min_T = 0.001
    cool_rate = 0.95
    no_improve_count = 0
    max_no_improve = 20

    while time.time() < deadline - 2.0:
        # Perturb
        new_positions = current_positions + rng.normal(0, kick_sigma, size=(n, 3))
        
        # Check min distance
        if not _min_distance_check(new_positions):
            # Try to repair by scaling? Or just reject.
            # Reject is safer.
            continue
            
        # Relax
        try:
            energy, positions = _relax_hybrid(new_positions, potential, rho, timeout=5.0)
        except Exception:
            continue
            
        # Acceptance: Metropolis
        delta_e = energy - current_energy
        if energy < current_energy:
            current_energy = energy
            current_positions = positions
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                no_improve_count = 0
                # Reduce kick size slightly on improvement
                kick_sigma *= 0.95
            else:
                no_improve_count += 1
        else:
            if np.random.rand() < np.exp(-delta_e / T):
                current_energy = energy
                current_positions = positions
                no_improve_count += 1
            else:
                no_improve_count += 1

        # Annealing
        T *= cool_rate
        if T < min_T:
            T = min_T
            
        # Adaptive kick size
        if no_improve_count > max_no_improve:
            # Increase kick size to escape
            kick_sigma *= 1.5
            if kick_sigma > 1.0:
                kick_sigma = 1.0
            no_improve_count = 0
            T = max(T, 0.05) # Reheat slightly

    return best_positions
