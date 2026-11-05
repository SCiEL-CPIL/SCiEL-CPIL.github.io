"""Search for global minima of Morse clusters using diverse initializations and BFGS relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_fast(pos, rho):
    """Compute Morse energy for a set of positions."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1e10)  # avoid self-interaction
    e = np.exp(rho * (1.0 - dist))
    e = e * (e - 2.0)
    return float(np.sum(e) / 2.0)


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like configuration."""
    points = []
    # FCC lattice points
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # Corner positions
                p = np.array([i, j, k]) * spacing
                if np.linalg.norm(p) <= 4.0 * spacing:
                    points.append(p)
                # Face center positions
                p2 = np.array([i + 0.5, j + 0.5, k]) * spacing
                if np.linalg.norm(p2) <= 4.0 * spacing:
                    points.append(p2)
                p3 = np.array([i + 0.5, j, k + 0.5]) * spacing
                if np.linalg.norm(p3) <= 4.0 * spacing:
                    points.append(p3)
                p4 = np.array([i, j + 0.5, k + 0.5]) * spacing
                if np.linalg.norm(p4) <= 4.0 * spacing:
                    points.append(p4)
    
    # Deduplicate
    unique_points = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 0.1 * spacing for q in unique_points):
            unique_points.append(p)
    
    # Select n points closest to center
    unique_points.sort(key=lambda p: np.linalg.norm(p))
    return np.array(unique_points[:n])


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like configuration."""
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                p = np.array([i, j, k]) * spacing
                if np.linalg.norm(p) <= 4.0 * spacing:
                    points.append(p)
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * spacing
                if np.linalg.norm(p2) <= 4.0 * spacing:
                    points.append(p2)
    
    unique_points = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 0.1 * spacing for q in unique_points):
            unique_points.append(p)
    
    unique_points.sort(key=lambda p: np.linalg.norm(p))
    return np.array(unique_points[:n])


def _generate_simple_cubic(n, spacing=1.0):
    """Generate simple cubic configuration."""
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                p = np.array([i, j, k]) * spacing
                if np.linalg.norm(p) <= 4.0 * spacing:
                    points.append(p)
    
    unique_points = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 0.1 * spacing for q in unique_points):
            unique_points.append(p)
    
    unique_points.sort(key=lambda p: np.linalg.norm(p))
    return np.array(unique_points[:n])


def _generate_random_compact(n, spacing=1.0, seed=42):
    """Generate random compact configuration with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = 10000
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min((np.linalg.norm(p - q) for q in points), default=np.inf)
            if min_dist >= 0.6 * spacing:
                points.append(p)
    
    # If we couldn't place all points, fill remaining with random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _generate_jittered_crystal(n, crystal_type, spacing=1.0, jitter=0.1, seed=42):
    """Generate crystal with random jitter."""
    rng = np.random.default_rng(seed)
    if crystal_type == "fcc":
        base = _generate_fcc(n, spacing)
    elif crystal_type == "bcc":
        base = _generate_bcc(n, spacing)
    else:
        base = _generate_simple_cubic(n, spacing)
    
    if len(base) < n:
        # Pad with random points
        radius = spacing * n ** (1.0 / 3.0) * 0.8
        while len(base) < n:
            p = rng.uniform(-radius, radius, 3)
            base = np.vstack([base, p])
    
    jittered = base + rng.normal(0, jitter, base.shape)
    return jittered


def _relax_with_bfgs(pos, potential, rho, maxiter=1000):
    """Relax using BFGS with gradient."""
    try:
        from scipy.optimize import minimize
        
        def objective(x):
            pos_2d = x.reshape(-1, 3)
            return _morse_energy_fast(pos_2d, rho)
        
        def gradient(x):
            pos_2d = x.reshape(-1, 3)
            grad = morse_energy_gradient(pos_2d, rho)
            return grad.flatten()
        
        result = minimize(
            objective,
            pos.flatten(),
            jac=gradient,
            method='BFGS',
            options={'maxiter': maxiter, 'gtol': 1e-6}
        )
        
        if result.success:
            return result.x.reshape(-1, 3), float(result.fun)
        else:
            return pos, _morse_energy_fast(pos, rho)
    except Exception:
        return pos, _morse_energy_fast(pos, rho)


def _check_min_distance(pos, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < min_dist:
                return False
    return True


def _enforce_min_distance(pos, min_dist=0.5, max_iter=100):
    """Nudge atoms to satisfy minimum distance constraint."""
    pos = pos.copy()
    for _ in range(max_iter):
        violated = False
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist and d > 1e-10:
                    direction = (pos[i] - pos[j]) / d
                    push = (min_dist - d) / 2.0 + 0.01
                    pos[i] += direction * push
                    pos[j] -= direction * push
                    violated = True
        if not violated:
            break
    return pos


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6.0))
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial configurations
    initializations = []
    
    # Crystalline starts
    for crystal_type in ["fcc", "bcc", "simple_cubic"]:
        try:
            if crystal_type == "fcc":
                init = _generate_fcc(size)
            elif crystal_type == "bcc":
                init = _generate_bcc(size)
            else:
                init = _generate_simple_cubic(size)
            if len(init) == size:
                initializations.append(init)
        except Exception:
            pass
    
    # Jittered crystals
    for seed in [42, 123, 456, 789]:
        try:
            init = _generate_jittered_crystal(size, "fcc", jitter=0.15, seed=seed)
            if len(init) == size:
                initializations.append(init)
        except Exception:
            pass
    
    # Random compact
    for seed in [42, 123, 456, 789, 1011, 1013]:
        try:
            init = _generate_random_compact(size, seed=seed)
            if len(init) == size:
                initializations.append(init)
        except Exception:
            pass
    
    # Relax each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 10:
            break
        
        try:
            # First try the built-in relax
            energy, positions = relax(init.copy(), potential, rho)
            
            if energy < best_energy:
                # Check minimum distance
                if _check_min_distance(positions, 0.5):
                    best_energy = energy
                    best_positions = positions.copy()
                else:
                    # Enforce minimum distance and re-relax
                    positions = _enforce_min_distance(positions, 0.5)
                    energy, positions = relax(positions, potential, rho)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                        
        except Exception:
            pass
    
    # Try BFGS on the best found so far and a few others
    if best_positions is not None and time.time() < deadline - 20:
        try:
            relaxed_pos, relaxed_energy = _relax_with_bfgs(best_positions, potential, rho, maxiter=500)
            if relaxed_energy < best_energy:
                if _check_min_distance(relaxed_pos, 0.5):
                    best_energy = relaxed_energy
                    best_positions = relaxed_pos.copy()
        except Exception:
            pass
    
    # If no valid configuration found, generate one
    if best_positions is None:
        init = _generate_random_compact(size, seed=42)
        init = _enforce_min_distance(init, 0.5)
        best_positions = init
        best_energy = _morse_energy_fast(init, rho)
    
    # Final check and enforcement
    best_positions = _enforce_min_distance(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
