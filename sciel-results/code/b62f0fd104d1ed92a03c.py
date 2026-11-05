# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline: 300s total for 6 calls. Keep a safety margin.
TOTAL_DEADLINE = 295.0
_START_TIME = time.time()

def _time_left():
    return max(0.0, TOTAL_DEADLINE - (time.time() - _START_TIME))

def _morse_energy_fast(positions, rho):
    """Vectorized Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    x = rho * (1.0 - dists)
    # Clip x to avoid overflow (exp(700) overflows double)
    x = np.clip(x, -50, 50)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    iu = np.triu_indices(n, k=1)
    return float(np.sum(e[iu]))

def _morse_gradient_fast(positions, rho):
    """Vectorized Morse gradient."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    x = rho * (1.0 - dists)
    x = np.clip(x, -50, 50)
    ex = np.exp(x)
    # dE/dr = rho * (2*exp(x) - 1) * exp(x) * (-1) ... wait
    # E(r) = exp(rho(1-r))*(exp(rho(1-r))-2)
    # Let u = rho(1-r), E = e^u(e^u-2) = e^{2u} - 2e^u
    # dE/du = 2e^{2u} - 2e^u = 2e^u(e^u-1)
    # du/dr = -rho
    # dE/dr = -rho * 2 * exp(x) * (exp(x) - 1)
    dE_dr = -rho * 2.0 * ex * (ex - 1.0)
    # dE/dx_i = sum_{j != i} dE/dr_ij * (x_i - x_j)/r_ij
    # Using vectorization:
    # grad[i] = sum_j dE_dr[i,j] * (pos[i] - pos[j]) / r_ij
    # Note: dE_dr is symmetric, diff is antisymmetric.
    # We can compute:
    # factor[i,j] = dE_dr[i,j] / r_ij
    # grad[i] = sum_j factor[i,j] * (pos[i] - pos[j])
    # grad[i] = pos[i] * sum_j factor[i,j] - sum_j factor[i,j] * pos[j]
    
    # Avoid division by zero on diagonal (but diagonal is inf, so dE_dr is 0? No, exp(inf) is inf. 
    # We set dists diagonal to inf, so x is -inf, exp is 0. dE_dr is 0. 
    # But 0/inf is 0. So factor diagonal is 0.
    # However, if dists is 0, x is rho, exp is finite. 
    # We assume atoms are distinct.
    
    factor = np.zeros_like(dists)
    mask = dists > 1e-10
    factor[mask] = dE_dr[mask] / dists[mask]
    
    # grad[i] = pos[i] * sum_j factor[i,j] - sum_j factor[i,j] * pos[j]
    row_sums = np.sum(factor, axis=1)
    grad = positions * row_sums[:, np.newaxis] - factor.dot(positions)
    return grad

def _enforce_min_distance(positions, min_dist=0.5):
    """Nudge atoms apart if they are too close."""
    n = len(positions)
    positions = positions.copy()
    for _ in range(10):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    if d < 1e-10:
                        diff = np.array([0.01, 0, 0])
                    else:
                        diff = (positions[i] - positions[j]) / d
                    shift = (min_dist - d + 0.01) / 2.0
                    positions[i] += diff * shift
                    positions[j] -= diff * shift
                    moved = True
        if not moved:
            break
    return positions

def _make_fcc_cluster(n, spacing=1.0):
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    a = spacing * np.sqrt(2)
    basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]) * a
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for b in basis:
                    pt = np.array([ix, iy, iz]) * a + b
                    points.append(pt)
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_bcc_cluster(n, spacing=1.0):
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    a = spacing * (2.0 / np.sqrt(3.0))
    basis = np.array([[0,0,0], [0.5,0.5,0.5]]) * a
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for b in basis:
                    pt = np.array([ix, iy, iz]) * a + b
                    points.append(pt)
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_hcp_cluster(n, spacing=1.0):
    a = spacing
    c = a * np.sqrt(8/3)
    points = []
    radius = int(np.ceil(np.sqrt(n / 3))) + 2
    for layer in range(2 * radius + 1):
        z = (layer - radius) * c
        for q in range(-radius, radius + 1):
            for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
                x = a * (q + r/2)
                y = a * (np.sqrt(3)/2) * r
                if layer % 2 == 0:
                    points.append([x, y, z])
                else:
                    points.append([x + a/2, y + a*np.sqrt(3)/6, z])
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_icosahedron(n, spacing=1.0):
    # Generate icosahedral shell points
    t = (1 + np.sqrt(5)) / 2
    pts = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, t, 1/t], [0, t, -1/t], [0, -t, 1/t], [0, -t, -1/t],
        [t, 1/t, 0], [t, -1/t, 0], [-t, 1/t, 0], [-t, -1/t, 0],
        [1/t, 0, t], [1/t, 0, -t], [-1/t, 0, t], [-1/t, 0, -t]
    ]
    pts = np.array(pts, dtype=float)
    pts /= np.linalg.norm(pts, axis=1)[:, np.newaxis]
    # Scale to have NN ~ 1.0
    # For icosahedron, NN distance is 2 * sin(pi/10) * R? No.
    # Distance between [1,1,1] and [1,1,-1] is 2. Norm is sqrt(3).
    # Angle is small.
    # Let's just scale so that min dist is ~1.0
    if len(pts) > 1:
        min_d = np.inf
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_d:
                    min_d = d
        if min_d > 0:
            pts *= (1.0 / min_d)
    
    # If n > 20, add inner layers or more random points
    if n <= 20:
        return pts[:n]
    
    # For larger n, use multiple shells or random compact
    return _make_random_compact(n, np.random.default_rng(42), spacing=1.0)

def _make_random_compact(n, rng, spacing=1.0):
    radius = spacing * n ** (1/3) * 1.2
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points], default=np.inf)
            if min_dist >= 0.8 * spacing:
                points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _relax_config(pos, rho, max_iter=2000):
    """Relax a single configuration using L-BFGS-B with custom gradient."""
    try:
        pos = pos.copy()
        # Ensure min distance before optimization to avoid singularities
        pos = _enforce_min_distance(pos, 0.5)
        
        res = minimize(
            _morse_energy_fast,
            pos,
            jac=_morse_gradient_fast,
            args=(rho,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < np.inf:
            final_pos = _enforce_min_distance(res.x, 0.5)
            final_energy = _morse_energy_fast(final_pos, rho)
            return final_energy, final_pos
        else:
            # Try the relax function from library as fallback
            energy, relaxed_pos = relax(pos, 'morse', rho)
            relaxed_pos = _enforce_min_distance(relaxed_pos, 0.5)
            final_energy = _morse_energy_fast(relaxed_pos, rho)
            return final_energy, relaxed_pos
    except Exception:
        return np.inf, pos

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    best_energy = np.inf
    best_positions = None
    
    rng = np.random.default_rng(42)
    
    initial_configs = []
    
    # 1. Crystalline templates
    initial_configs.append(_make_fcc_cluster(n))
    initial_configs.append(_make_bcc_cluster(n))
    initial_configs.append(_make_hcp_cluster(n))
    initial_configs.append(_make_icosahedron(n))
    
    # 2. Random compact
    initial_configs.append(_make_random_compact(n, rng))
    
    # 3. Jittered versions
    for i in range(3):
        base = initial_configs[i].copy()
        jitter = rng.normal(0, 0.2, size=base.shape)
        initial_configs.append(base + jitter)
    
    # 4. More random starts
    for i in range(3):
        initial_configs.append(_make_random_compact(n, rng))
    
    # Shuffle
    rng.shuffle(initial_configs)
    
    time_limit = _time_left()
    if time_limit < 10:
        for pos in initial_configs:
            e = _morse_energy_fast(pos, rho)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        return _enforce_min_distance(best_positions, 0.5)
    
    # Parallel relaxation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_pos = {executor.submit(_relax_config, pos, rho): pos for pos in initial_configs}
        
        for future in concurrent.futures.as_completed(future_to_pos):
            if _time_left() < 5.0:
                pass # Can't cancel, but break if we have a good result? 
            try:
                energy, pos = future.result(timeout=2.0)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                continue
    
    if best_positions is None:
        best_positions = initial_configs[0]
        best_energy = _morse_energy_fast(best_positions, rho)
        
    best_positions = _enforce_min_distance(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
