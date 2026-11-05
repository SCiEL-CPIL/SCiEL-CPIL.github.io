# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 320.0

def _dzugutov_energy_and_grad(positions):
    """Compute Dzugutov energy and gradient, fully vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n,n,3)
    dist_sq = np.sum(diff**2, axis=2)  # (n,n)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Masks
    mask_a = (dist < a) & (dist > 1e-10)
    mask_b_only = (dist < b) & (dist >= a) & (dist > 1e-10)
    
    energy = 0.0
    # Force matrix: F[i,j] = force on i due to j
    F = np.zeros_like(positions)  # (n,n,3)
    
    # Part 1: r < a
    if np.any(mask_a):
        r_a = dist[mask_a]
        exp_a = np.exp(c / (r_a - a))
        v_a = A * (r_a**(-m) - B) * exp_a
        energy += np.sum(v_a)
        
        # dV/dr
        dvdr_a = A * (-m * r_a**(-m-1) * exp_a + (r_a**(-m) - B) * exp_a * (-c / (r_a - a)**2))
        
        # Force on i from j: dvdr * (pos_i - pos_j) / r
        # diff[i,j] = pos_i - pos_j
        # F[i,j] = dvdr_ij * diff[i,j] / r_ij
        F_a = np.zeros_like(positions)
        F_a[mask_a] = (dvdr_a / r_a)[:, np.newaxis] * diff[mask_a]
        F += F_a
    
    # Part 2: a <= r < b
    if np.any(mask_b_only):
        r_b = dist[mask_b_only]
        exp_b = np.exp(d / (r_b - b))
        v_b = B * exp_b
        energy += np.sum(v_b)
        
        dvdr_b = B * exp_b * (-d / (r_b - b)**2)
        
        F_b = np.zeros_like(positions)
        F_b[mask_b_only] = (dvdr_b / r_b)[:, np.newaxis] * diff[mask_b_only]
        F += F_b
    
    # Total force on atom i = sum_j F[i,j]
    grad = -np.sum(F, axis=1)  # gradient = -force
    
    return energy, grad


def _make_icosahedron(n):
    """Generate icosahedral-like structure."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms
    
    positions = [np.zeros(3)]
    if n <= 1:
        return np.array(positions)
    
    scale1 = 1.1
    for v in ico_verts:
        if len(positions) >= n:
            break
        positions.append(v * scale1)
    
    if len(positions) < n:
        # Second shell
        scale2 = 1.8
        edge_mids = []
        for i in range(12):
            for j in range(i+1, 12):
                mid = ico_verts[i] + ico_verts[j]
                nm = np.linalg.norm(mid)
                if nm > 0.1:
                    edge_mids.append(mid / nm)
        edge_mids = np.array(edge_mids)
        unique = []
        for em in edge_mids:
            if all(np.linalg.norm(em - um) > 0.15 for um in unique):
                unique.append(em)
        for em in unique:
            if len(positions) >= n:
                break
            positions.append(em * scale2)
    
    if len(positions) < n:
        rng = np.random.default_rng(42)
        scale3 = 2.4
        while len(positions) < n:
            p = rng.normal(size=3)
            p = p / (np.linalg.norm(p) + 1e-10) * scale3
            if all(np.linalg.norm(p - q) > 0.5 for q in positions):
                positions.append(p)
    
    return np.array(positions[:n])


def _make_fcc_like(n):
    """FCC-like cluster."""
    rng = np.random.default_rng(n)
    positions = []
    a = 1.1
    for x in range(-4, 5):
        for y in range(-4, 5):
            for z in range(-4, 5):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([x + dx, y + dy, z + dz]) * a
                    if len(positions) < n and all(np.linalg.norm(p - q) > 0.5 for q in positions):
                        positions.append(p)
    while len(positions) < n:
        p = rng.uniform(-3, 3, 3)
        if all(np.linalg.norm(p - q) > 0.5 for q in positions):
            positions.append(p)
    return np.array(positions[:n])


def _make_random_compact(n, seed=0):
    """Random compact cluster."""
    rng = np.random.default_rng(seed)
    spacing = 1.1
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.normal(0, 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def _optimize(positions, max_iter=300):
    """L-BFGS-B optimization."""
    def fun_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _dzugutov_energy_and_grad(pos)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun_and_grad, positions.flatten(), method='L-BFGS-B',
                         jac=True, options={'maxiter': max_iter, 'maxfun': 3000,
                                           'ftol': 1e-12, 'gtol': 1e-8})
        return result.fun, result.x.reshape(-1, 3)
    except Exception:
        return np.inf, positions


def construct(instance):
    """Find low-energy Dzugutov clusters."""
    size = instance["n"]
    start_t = time.time()
    
    # Diverse starts
    starts = [
        _make_icosahedron(size),
        _make_fcc_like(size),
        _make_random_compact(size, seed=size),
        _make_random_compact(size, seed=size+100),
        _make_random_compact(size, seed=size+200),
    ]
    
    valid_starts = []
    for s in starts:
        if len(s) == size:
            # Check minimum distance
            d = np.sqrt(np.sum((s[:, np.newaxis, :] - s[np.newaxis, :, :])**2, axis=2))
            np.fill_diagonal(d, 1.0)
            if np.min(d) > 0.49:
                valid_starts.append(s)
    
    if not valid_starts:
        rng = np.random.default_rng(size)
        valid_starts = [rng.uniform(-2, 2, (size, 3))]
    
    best_energy = np.inf
    best_positions = valid_starts[0]
    
    n_starts = len(valid_starts)
    for i, start in enumerate(valid_starts):
        if time.time() - start_t > 310:
            break
        e, pos = _optimize(start, max_iter=200)
        if e < best_energy:
            best_energy = e
            best_positions = pos
    
    # Second pass: perturb best
    if time.time() - start_t < 290 and best_energy < np.inf:
        rng = np.random.default_rng(999)
        for _ in range(4):
            if time.time() - start_t > 310:
                break
            perturbed = best_positions + rng.normal(0, 0.08, size=(size, 3))
            e, pos = _optimize(perturbed, max_iter=150)
            if e < best_energy:
                best_energy = e
                best_positions = pos
    
    return best_positions
# EVOLVE-BLOCK-END
