"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(coords, radii, tol=1e-6):
    """Check if circles are inside unit square and non-overlapping."""
    if np.any(radii < 0):
        return False
    # Check bounds: each circle must fit in [0,1]^2
    if np.any(coords[:, 0] - radii < -tol):
        return False
    if np.any(coords[:, 0] + radii > 1 + tol):
        return False
    if np.any(coords[:, 1] - radii < -tol):
        return False
    if np.any(coords[:, 1] + radii > 1 + tol):
        return False
    # Check overlaps (vectorized)
    n = len(radii)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _vectorized_valid(coords, radii, tol=1e-6):
    """Faster validity check using vectorized pairwise distances."""
    if np.any(radii < 0):
        return False
    if np.any(coords[:, 0] - radii < -tol):
        return False
    if np.any(coords[:, 0] + radii > 1 + tol):
        return False
    if np.any(coords[:, 1] - radii < -tol):
        return False
    if np.any(coords[:, 1] + radii > 1 + tol):
        return False
    n = len(radii)
    # Build pairwise distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    # Check upper triangle
    iu = np.triu_indices(n, k=1)
    if np.any(dist[iu] < r_sum[iu] - tol):
        return False
    return True

def _relax(coords, radii, n_iters=200, dt=0.02):
    """Force-based relaxation: push overlapping circles apart and away from walls.
    Returns updated coords (radii unchanged)."""
    n = len(radii)
    coords = coords.copy()
    for it in range(n_iters):
        # Compute pairwise forces
        dx = coords[:, 0:1] - coords[0:1, :]
        dy = coords[:, 1:2] - coords[0:1, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, 1.0)  # avoid division by zero
        
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0, r_sum - dist)
        
        # Force magnitude along each pair
        # For overlapping pairs, push apart
        force_x = np.zeros((n, n))
        force_y = np.zeros((n, n))
        
        mask = overlap > 0
        # Avoid division by zero
        safe_dist = np.where(dist > 1e-12, dist, 1e-12)
        fx = dx / safe_dist * overlap
        fy = dy / safe_dist * overlap
        fx = np.where(mask, fx, 0.0)
        fy = np.where(mask, fy, 0.0)
        
        # Sum forces on each circle (row i gets force from all j)
        total_fx = np.sum(fx, axis=1)
        total_fy = np.sum(fy, axis=1)
        
        # Wall forces: push inward if circle violates boundary
        # Left wall: x - r < 0 -> push right
        left_viol = np.maximum(0, radii - coords[:, 0])
        # Right wall: x + r > 1 -> push left
        right_viol = np.maximum(0, coords[:, 0] + radii - 1)
        # Bottom wall: y - r < 0 -> push up
        bottom_viol = np.maximum(0, radii - coords[:, 1])
        # Top wall: y + r > 1 -> push down
        top_viol = np.maximum(0, coords[:, 1] + radii - 1)
        
        total_fx += left_viol - right_viol
        total_fy += bottom_viol - top_viol
        
        # Update positions
        coords[:, 0] += dt * total_fx
        coords[:, 1] += dt * total_fy
        
        # Clamp to valid region (hard constraint)
        coords[:, 0] = np.clip(coords[:, 0], radii, 1 - radii)
        coords[:, 1] = np.clip(coords[:, 1], radii, 1 - radii)
    
    return coords

def _grow_and_relax(coords, radii, n_cycles=50, growth_step=0.002, relax_iters=100, dt=0.02):
    """Iteratively grow all radii and relax positions.
    Returns updated coords and radii."""
    coords = coords.copy()
    radii = radii.copy()
    for cycle in range(n_cycles):
        # Try to grow all radii
        new_radii = radii + growth_step
        # Check if new radii fit in the square
        if np.any(new_radii > 0.5) or np.any(coords[:, 0] - new_radii < -0.5) or \
           np.any(coords[:, 0] + new_radii > 1.5) or np.any(coords[:, 1] - new_radii < -0.5) or \
           np.any(coords[:, 1] + new_radii > 1.5):
            break
        radii = new_radii
        # Relax positions to accommodate new radii
        coords = _relax(coords, radii, n_iters=relax_iters, dt=dt)
        # Clamp
        coords[:, 0] = np.clip(coords[:, 0], radii, 1 - radii)
        coords[:, 1] = np.clip(coords[:, 1], radii, 1 - radii)
    return coords, radii

def _lbfgs_polish(coords, radii, N, time_limit=10.0):
    """Use L-BFGS-B to polish the packing. Returns (coords, radii) or None if fails."""
    start = time.time()
    
    def _objective(params):
        if time.time() - start > time_limit:
            return np.inf
        c = params[:2 * N].reshape(-1, 2)
        r = np.exp(params[2 * N:])
        r = np.clip(r, 1e-6, 0.5)
        
        penalty = 0.0
        penalty += np.sum(np.maximum(0, r - c[:, 0]) ** 2)
        penalty += np.sum(np.maximum(0, c[:, 0] + r - 1) ** 2)
        penalty += np.sum(np.maximum(0, r - c[:, 1]) ** 2)
        penalty += np.sum(np.maximum(0, c[:, 1] + r - 1) ** 2)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist_sq = dx * dx + dy * dy
                rs = r[i] + r[j]
                if dist_sq < rs * rs:
                    overlap = rs - np.sqrt(dist_sq)
                    penalty += overlap * overlap
        
        return -np.sum(r) + 10.0 * penalty
    
    log_r = np.log(np.maximum(radii, 1e-10))
    x0 = np.concatenate([coords.flatten(), log_r])
    
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        c = result.x[:2 * N].reshape(-1, 2)
        r = np.exp(result.x[2 * N:])
        r = np.clip(r, 1e-6, 0.5)
        c[:, 0] = np.clip(c[:, 0], r, 1 - r)
        c[:, 1] = np.clip(c[:, 1], r, 1 - r)
        return c, r
    except Exception:
        return None

def _seed_grid(N):
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0.5 / side, 1 - 0.5 / side, side)
    ys = np.linspace(0.5 / side, 1 - 0.5 / side, side)
    coords = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(coords) >= N:
                break
            coords.append([xs[i], ys[j]])
            radii.append(0.5 / side * 0.7)
    return np.array(coords[:N]), np.array(radii[:N])

def _seed_hex(N):
    """Hexagonal initialization."""
    # Hexagonal packing: rows offset
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    coords = []
    radii = []
    r0 = 0.15
    for row in range(rows):
        n_cols = int(np.ceil(1 / (r0 * np.sqrt(3)))) + 1
        offset = (row % 2) * r0 * np.sqrt(3) / 2
        for col in range(n_cols):
            if len(coords) >= N:
                break
            x = r0 * np.sqrt(3) * (col + 0.5) + offset
            y = r0 * (row + 0.5) * 1.0
            if 0 < x < 1 and 0 < y < 1:
                coords.append([x, y])
                radii.append(r0 * 0.7)
    # If not enough, fill with random
    while len(coords) < N:
        coords.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
        radii.append(0.1)
    return np.array(coords[:N]), np.array(radii[:N])

def _seed_random(N, seed=None):
    """Random initialization."""
    if seed is not None:
        np.random.seed(seed)
    coords = np.random.uniform(0.15, 0.85, (N, 2))
    radii = np.full(N, 0.08)
    return coords, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    N = 26
    
    best_result = None
    best_score = -np.inf
    
    def _evaluate_and_update(coords, radii):
        nonlocal best_result, best_score
        # Clamp
        radii = np.clip(radii, 1e-6, 0.5)
        coords[:, 0] = np.clip(coords[:, 0], radii, 1 - radii)
        coords[:, 1] = np.clip(coords[:, 1], radii, 1 - radii)
        if _is_valid(coords, radii, tol=1e-5):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([coords, radii])
    
    # Try multiple diverse initializations
    initializations = []
    
    # 1. Grid
    c, r = _seed_grid(N)
    initializations.append(('grid', c, r))
    
    # 2. Hexagonal
    c, r = _seed_hex(N)
    initializations.append(('hex', c, r))
    
    # 3. Random (multiple)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        c, r = _seed_random(N, seed=seed)
        initializations.append((f'random_{seed}', c, r))
    
    # 4. Perturbed best (added dynamically)
    
    n_restarts = len(initializations)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        name, coords, radii = initializations[restart]
        
        # Relax initial positions
        coords = _relax(coords, radii, n_iters=100, dt=0.02)
        coords[:, 0] = np.clip(coords[:, 0], radii, 1 - radii)
        coords[:, 1] = np.clip(coords[:, 1], radii, 1 - radii)
        
        _evaluate_and_update(coords, radii)
        
        # Grow and relax
        coords_grown, radii_grown = _grow_and_relax(
            coords, radii, n_cycles=40, growth_step=0.0015, relax_iters=80, dt=0.02
        )
        coords_grown[:, 0] = np.clip(coords_grown[:, 0], radii_grown, 1 - radii_grown)
        coords_grown[:, 1] = np.clip(coords_grown[:, 1], radii_grown, 1 - radii_grown)
        _evaluate_and_update(coords_grown, radii_grown)
        
        # L-BFGS polish
        if time.time() < deadline - 15:
            polished = _lbfgs_polish(coords_grown, radii_grown, N, time_limit=8.0)
            if polished is not None:
                pc, pr = polished
                _evaluate_and_update(pc, pr)
        
        # If we found a good result, add a perturbed version for next restarts
        if best_result is not None and restart < n_restarts - 2:
            bc = best_result[:, :2].copy()
            br = best_result[:, 2].copy()
            perturbed_c = bc + np.random.randn(N, 2) * 0.05
            perturbed_r = np.maximum(br * (1 + np.random.randn(N) * 0.05), 0.01)
            initializations.append(('perturbed', perturbed_c, perturbed_r))
            n_restarts = len(initializations)
    
    # Final: try to grow the best result further
    if best_result is not None and time.time() < deadline - 10:
        bc = best_result[:, :2].copy()
        br = best_result[:, 2].copy()
        # Try more aggressive growth
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            coords_grown, radii_grown = _grow_and_relax(
                bc, br, n_cycles=30, growth_step=0.001, relax_iters=100, dt=0.015
            )
            coords_grown[:, 0] = np.clip(coords_grown[:, 0], radii_grown, 1 - radii_grown)
            coords_grown[:, 1] = np.clip(coords_grown[:, 1], radii_grown, 1 - radii_grown)
            _evaluate_and_update(coords_grown, radii_grown)
            bc, br = best_result[:, :2].copy(), best_result[:, 2].copy()
    
    # If no valid result found, fall back to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
