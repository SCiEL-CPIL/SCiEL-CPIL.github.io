"""Circle packing in unit square - force-based repulsion + L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Grid with jitter
    rng = np.random.default_rng(42)
    side = int(np.ceil(np.sqrt(N)))
    grid = []
    for i in range(side):
        for j in range(side):
            if len(grid) >= N:
                break
            grid.append([(i + 0.5) / side, (j + 0.5) / side])
    grid = np.array(grid[:N]) + rng.uniform(-0.02, 0.02, size=(N, 2))
    inits.append(grid)
    
    # 2. Hexagonal-ish pattern
    hex_pos = []
    for row in range(8):
        for col in range(8):
            x = col * 0.14 + (0.07 if row % 2 else 0)
            y = row * 0.12
            if 0 <= x <= 1 and 0 <= y <= 1 and len(hex_pos) < N:
                hex_pos.append([x, y])
    if len(hex_pos) < N:
        while len(hex_pos) < N:
            hex_pos.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)])
    inits.append(np.array(hex_pos[:N]))
    
    # 3. Random
    inits.append(rng.uniform(0.05, 0.95, size=(N, 2)))
    
    # 4. Corner-biased random
    corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
                        [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]])
    corner_init = np.array([corners[i % 8] + rng.uniform(-0.08, 0.08, 2) for i in range(N)])
    inits.append(np.clip(corner_init, 0.02, 0.98))
    
    return inits


def _energy(params, n):
    """Compute energy: negative sum of radii + penalty for overlaps and boundary violations."""
    pos = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    # Sum of radii (we want to maximize, so minimize negative)
    base = -np.sum(radii)
    
    # Overlap penalty: for each pair, if dist < r_i + r_j, penalize
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((pos[i] - pos[j])**2) + 1e-12)
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                penalty += overlap**2
    
    # Boundary penalty: circle must be inside [0,1]^2
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        # Distance to each wall
        d_left = x - r
        d_right = 1 - x - r
        d_bottom = y - r
        d_top = 1 - y - r
        for d in [d_left, d_right, d_bottom, d_top]:
            if d < 0:
                penalty += d**2
    
    return base + 10.0 * penalty


def _energy_grad(params, n):
    """Gradient of energy function."""
    pos = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(radii)
    grad[2 * n:] = -1.0
    
    # Overlap gradient
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[i] - pos[j]
            d = np.sqrt(np.sum(diff**2) + 1e-12)
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                # d(penalty)/d(pos_i) = 2*overlap * (pos_i - pos_j)/d
                factor = 20.0 * overlap / d
                grad[2*i] += factor * diff[0]
                grad[2*i+1] += factor * diff[1]
                grad[2*j] -= factor * diff[0]
                grad[2*j+1] -= factor * diff[1]
                # d(penalty)/d(r_i) = 2*overlap
                grad[2*n+i] += 20.0 * overlap
                grad[2*n+j] += 20.0 * overlap
    
    # Boundary gradient
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        walls = [(x - r, 1, 0), (1 - x - r, -1, 0), (y - r, 0, 1), (1 - y - r, 0, -1)]
        for d, sx, sy in walls:
            if d < 0:
                grad[2*i] += 20.0 * d * sx
                grad[2*i+1] += 20.0 * d * sy
                grad[2*n+i] += 20.0 * d * (-1)  # dr/d(r) for wall distance
    
    return grad


def _is_valid(params, n, tol=1e-6):
    """Check if configuration is valid (no overlaps, all inside square)."""
    pos = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if r <= 0 or x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((pos[i] - pos[j])**2))
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_params = None
    best_score = -np.inf
    
    # Start with a reasonable grid
    inits = _initializations()
    
    # Initial radii estimate
    initial_r = 0.5 / np.ceil(np.sqrt(N)) * 0.9
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Build initial parameter vector
        params0 = np.concatenate([init_pos.flatten(), np.full(N, initial_r)])
        
        # Try L-BFGS-B
        try:
            result = minimize(
                _energy,
                params0,
                jac=_energy_grad,
                method='L-BFGS-B',
                args=(N,),
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                if _is_valid(result.x, N):
                    score = np.sum(result.x[2*N:])
                    if score > best_score:
                        best_score = score
                        best_params = result.x.copy()
        except Exception:
            pass
        
        # Also try a quick force-based relaxation if time permits
        if time.time() > deadline - 5:
            break
        
        # Force-based relaxation as secondary optimizer
        pos = init_pos.copy()
        radii = np.full(N, initial_r)
        
        for step in range(500):
            if time.time() > deadline - 5:
                break
            forces = np.zeros((N, 2))
            
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pos[i] - pos[j]
                    d = np.sqrt(np.sum(diff**2)) + 1e-10
                    min_d = radii[i] + radii[j]
                    if d < min_d:
                        f = (min_d - d) / d
                        forces[i] += f * diff
                        forces[j] -= f * diff
                # Wall repulsion
                for axis in range(2):
                    if pos[i, axis] < radii[i]:
                        forces[i, axis] += (radii[i] - pos[i, axis])
                    if pos[i, axis] > 1 - radii[i]:
                        forces[i, axis] -= (pos[i, axis] - (1 - radii[i]))
            
            pos += 0.1 * forces
            pos = np.clip(pos, 0.01, 0.99)
        
        # Check if force-based result is better
        fb_params = np.concatenate([pos.flatten(), radii])
        if _is_valid(fb_params, N):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_params = fb_params.copy()
    
    # Final polish: if we have a valid solution, try to improve it
    if best_params is not None and time.time() < deadline - 3:
        try:
            result = minimize(
                _energy,
                best_params,
                jac=_energy_grad,
                method='L-BFGS-B',
                args=(N,),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if _is_valid(result.x, N):
                score = np.sum(result.x[2*N:])
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Extract final result
    pos = best_params[:2*N].reshape(N, 2)
    radii = best_params[2*N:]
    result = np.column_stack([pos, radii])
    
    # Final validity check
    if not _is_valid(result, N):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return result
# EVOLVE-BLOCK-END
