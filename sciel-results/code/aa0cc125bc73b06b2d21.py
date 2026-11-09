"""Physics-based soft-body simulation for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _lj_energy(pos, radii, W, H):
    """
    Lennard-Jones-like repulsive potential between circles,
    wall confinement, and objective to maximize sum of radii.
    """
    n = len(radii)
    energy = 0.0
    
    # Inter-circle: use a smooth repulsive potential
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-12
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap = min_dist - dist
                energy += 500.0 * overlap * overlap
    
    # Wall confinement
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        if x < r:
            energy += 500.0 * (r - x) ** 2
        if x > W - r:
            energy += 500.0 * (x - (W - r)) ** 2
        if y < r:
            energy += 500.0 * (r - y) ** 2
        if y > H - r:
            energy += 500.0 * (y - (H - r)) ** 2
    
    # Objective: maximize sum of radii
    energy -= 8.0 * np.sum(radii)
    
    # Soft bound on radii
    energy += 0.01 * np.sum(np.maximum(0, radii - 0.3) ** 2)
    
    return energy


def _lj_grad(pos, radii, W, H):
    """Gradient of the energy function."""
    n = len(radii)
    grad_pos = np.zeros_like(pos)
    grad_radii = np.zeros_like(radii)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-12
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap = min_dist - dist
                factor = 1000.0 * overlap / dist
                grad_pos[i, 0] += factor * dx
                grad_pos[i, 1] += factor * dy
                grad_pos[j, 0] -= factor * dx
                grad_pos[j, 1] -= factor * dy
                grad_radii[i] += 1000.0 * overlap
                grad_radii[j] += 1000.0 * overlap
    
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        if x < r:
            g = 1000.0 * (r - x)
            grad_pos[i, 0] += -g
            grad_radii[i] += g
        if x > W - r:
            g = 1000.0 * (x - (W - r))
            grad_pos[i, 0] += g
            grad_radii[i] += -g
        if y < r:
            g = 1000.0 * (r - y)
            grad_pos[i, 1] += -g
            grad_radii[i] += g
        if y > H - r:
            g = 1000.0 * (y - (H - r))
            grad_pos[i, 1] += g
            grad_radii[i] += -g
    
    grad_radii -= 8.0
    for i in range(n):
        if radii[i] > 0.3:
            grad_radii[i] += 0.02 * (radii[i] - 0.3)
    
    return grad_pos, grad_radii


def _is_valid(pos, radii, W, H, tol=1e-8):
    """Check if packing is valid."""
    for i in range(N):
        r = radii[i]
        x, y = pos[i]
        if x < r - tol or x > W - r + tol:
            return False
        if y < r - tol or y > H - r + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _make_initial_pos(W, H, seed):
    """Create diverse initial configurations."""
    rng = np.random.RandomState(seed)
    
    if seed % 3 == 0:
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) * W / side + rng.uniform(-0.02, 0.02)
            pos[i, 1] = (row + 0.5) * H / side + rng.uniform(-0.02, 0.02)
        r_max = min(W, H) / (2 * side)
        radii = np.full(N, r_max * 0.4)
    elif seed % 3 == 1:
        pos = np.column_stack([
            rng.uniform(0.05, W - 0.05, N),
            rng.uniform(0.05, H - 0.05, N)
        ])
        radii = np.full(N, 0.03)
    else:
        pos = np.zeros((N, 2))
        row_spacing = 0.3
        col_spacing = 0.32
        for i in range(N):
            row = i // 5
            col = i % 5
            offset = 0.16 if row % 2 == 1 else 0.0
            pos[i, 0] = (col + 0.5) * col_spacing + offset
            pos[i, 1] = (row + 0.5) * row_spacing
        pos[:, 0] = pos[:, 0] / pos[:, 0].max() * W * 0.9
        pos[:, 1] = pos[:, 1] / pos[:, 1].max() * H * 0.9
        radii = np.full(N, 0.04)
    
    return pos, radii


def _optimize_single(W, H, pos0, radii0, maxiter=2000):
    """Run L-BFGS-B optimization for a single configuration."""
    x0 = np.concatenate([pos0.flatten(), radii0])
    
    def objective(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        return _lj_energy(p, r, W, H)
    
    def grad(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        gp, gr = _lj_grad(p, r, W, H)
        return np.concatenate([gp.flatten(), gr])
    
    try:
        result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        radii_opt = np.maximum(radii_opt, 1e-6)
        return pos_opt, radii_opt, result.success
    except:
        return pos0, radii0, False


def _fix_overlaps(pos, radii, W, H, iterations=200):
    """Iteratively shrink radii to resolve any remaining overlaps."""
    for _ in range(iterations):
        resolved = True
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                min_dist = radii[i] + radii[j]
                if dist < min_dist - 1e-10:
                    resolved = False
                    shrink = (min_dist - dist) / 2.0 + 1e-4
                    radii[i] = max(radii[i] - shrink, 1e-6)
                    radii[j] = max(radii[j] - shrink, 1e-6)
        if resolved:
            break
    
    for i in range(N):
        radii[i] = min(radii[i],
                       pos[i, 0],
                       W - pos[i, 0],
                       pos[i, 1],
                       H - pos[i, 1])
        radii[i] = max(radii[i], 1e-6)
    
    return radii


def construct():
    """
    Pack 21 circles in a perimeter-4 rectangle to maximize sum of radii.
    Uses multi-start L-BFGS-B optimization with diverse initializations.
    """
    best_result = None
    best_sum = 0.0
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.1, 0.9, 1.3, 0.75]
    n_restarts = 5
    
    for aspect in aspect_ratios:
        W = 2.0 * aspect / (1.0 + aspect)
        H = 2.0 / (1.0 + aspect)
        
        for restart in range(n_restarts):
            seed = int(aspect * 100) + restart * 7
            pos0, radii0 = _make_initial_pos(W, H, seed)
            
            pos_opt, radii_opt, success = _optimize_single(W, H, pos0, radii0, maxiter=1500)
            
            radii_opt = _fix_overlaps(pos_opt, radii_opt, W, H)
            
            if _is_valid(pos_opt, radii_opt, W, H):
                total = np.sum(radii_opt)
                if total > best_sum:
                    best_sum = total
                    best_result = (pos_opt.copy(), radii_opt.copy(), W, H)
    
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        W, H = 1.0, 1.0
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) * W / side
            y = (row + 0.5) / side * H
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
        return np.array(cells)
    
    pos_opt, radii_opt, W, H = best_result
    result = np.column_stack([pos_opt, radii_opt])
    return result
# EVOLVE-BLOCK-END
