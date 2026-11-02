import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

def construct():
    N = 26
    deadline = time.time() + 135  # Safety margin for 150s limit
    
    def _elapsed():
        return time.time()
    
    def _remaining():
        return max(0, deadline - _elapsed())
    
    def _softplus(x, k=100):
        """Smooth approximation of max(0, x) with curvature controlled by k."""
        return np.where(x > 0, x + np.log1p(np.exp(-k * x)) / k, 
                        np.log1p(np.exp(k * x)) / k)
    
    def _validate(pos, radii):
        """Check if configuration is valid (all circles inside unit square, no overlaps)."""
        # Boundary check
        if np.any(pos[:, 0] - radii < -1e-7) or np.any(pos[:, 0] + radii > 1 + 1e-7):
            return False
        if np.any(pos[:, 1] - radii < -1e-7) or np.any(pos[:, 1] + radii > 1 + 1e-7):
            return False
        # Overlap check using Delaunay for efficiency
        if N < 3:
            return True
        try:
            tri = Delaunay(pos)
            edges = set()
            for simplex in tri.simplices:
                edges.add((min(simplex[0], simplex[1]), max(simplex[0], simplex[1])))
                edges.add((min(simplex[0], simplex[2]), max(simplex[0], simplex[2])))
                edges.add((min(simplex[1], simplex[2]), max(simplex[1], simplex[2])))
            for i, j in edges:
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radii[i] + radii[j] - 1e-7:
                    return False
        except Exception:
            # Fallback: check all pairs
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < radii[i] + radii[j] - 1e-7:
                        return False
        return True
    
    def _fix_boundaries(pos, radii):
        """Iteratively fix boundary violations and overlaps by shrinking radii."""
        pos = pos.copy()
        radii = radii.copy()
        for _ in range(100):
            # Fix boundary violations
            for i in range(N):
                max_r = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
                if radii[i] > max_r:
                    radii[i] = max_r * 0.999
            # Fix overlaps by shrinking larger radius
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < radii[i] + radii[j]:
                        # Shrink the larger circle
                        if radii[i] >= radii[j]:
                            radii[i] = max(0, (d - radii[j]) * 0.999)
                        else:
                            radii[j] = max(0, (d - radii[i]) * 0.999)
                        changed = True
            if not changed:
                break
        return pos, radii
    
    def _get_delaunay_edges(pos):
        """Get contact edges from Delaunay triangulation."""
        if N < 3:
            return [(0, 1)] if N == 2 else []
        try:
            tri = Delaunay(pos)
            edges = set()
            for simplex in tri.simplices:
                edges.add((min(simplex[0], simplex[1]), max(simplex[0], simplex[1])))
                edges.add((min(simplex[0], simplex[2]), max(simplex[0], simplex[2])))
                edges.add((min(simplex[1], simplex[2]), max(simplex[1], simplex[2])))
            return list(edges)
        except Exception:
            return [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def _objective_penalty(x, radii):
        """Penalty function: sum of overlap violations (smooth)."""
        pos = x.reshape(N, 2)
        penalty = 0.0
        # Boundary penalties
        for i in range(N):
            penalty += _softplus(radii[i] - pos[i, 0], 50)
            penalty += _softplus(pos[i, 0] + radii[i] - 1, 50)
            penalty += _softplus(radii[i] - pos[i, 1], 50)
            penalty += _softplus(pos[i, 1] + radii[i] - 1, 50)
        # Overlap penalties using Delaunay edges
        edges = _get_delaunay_edges(pos)
        for i, j in edges:
            d = np.linalg.norm(pos[i] - pos[j])
            overlap = radii[i] + radii[j] - d
            penalty += _softplus(overlap, 50)
        return penalty
    
    def _objective_grad(x, radii):
        """Gradient of penalty function."""
        pos = x.reshape(N, 2)
        grad = np.zeros_like(x)
        # Boundary gradients
        for i in range(N):
            # Left boundary
            pen = radii[i] - pos[i, 0]
            if pen > 0:
                grad[2*i] += 1.0
            # Right boundary
            pen = pos[i, 0] + radii[i] - 1
            if pen > 0:
                grad[2*i] -= 1.0
            # Bottom boundary
            pen = radii[i] - pos[i, 1]
            if pen > 0:
                grad[2*i+1] += 1.0
            # Top boundary
            pen = pos[i, 1] + radii[i] - 1
            if pen > 0:
                grad[2*i+1] -= 1.0
        # Overlap gradients using Delaunay edges
        edges = _get_delaunay_edges(pos)
        for i, j in edges:
            d_vec = pos[i] - pos[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-10:
                continue
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                force = d_vec / d
                grad[2*i:2*i+2] += force
                grad[2*j:2*j+2] -= force
        return grad
    
    def _hex_initial():
        """Create hexagonal-like initial configuration."""
        # Use a staggered grid pattern
        pos = np.zeros((N, 2))
        # 5 columns, 5-6 rows
        cols = 5
        margin = 0.1
        x_spacing = (1 - 2*margin) / (cols - 1) if cols > 1 else 0.5
        y_spacing = x_spacing * 0.866  # hexagonal spacing
        
        idx = 0
        for row in range(6):
            for col in range(cols):
                if idx >= N:
                    break
                x = margin + col * x_spacing
                if row % 2 == 1:
                    x += x_spacing / 2
                y = margin + row * y_spacing
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pos[idx] = [x, y]
                    idx += 1
        # If we didn't fill all, add remaining at random valid positions
        while idx < N:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            pos[idx] = [x, y]
            idx += 1
        
        # Compute initial radii based on minimum distance
        min_d = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                min_d = min(min_d, d)
        for i in range(N):
            min_d = min(min_d, pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
        
        radii = np.full(N, min_d * 0.9)
        pos, radii = _fix_boundaries(pos, radii)
        return pos, radii
    
    def _grid_initial():
        """Create grid initial configuration."""
        pos = np.zeros((N, 2))
        # 5x5 grid = 25, add 1 more
        xs = np.linspace(0.1, 0.9, 5)
        ys = np.linspace(0.1, 0.9, 5)
        idx = 0
        for y in ys:
            for x in xs:
                if idx < N:
                    pos[idx] = [x, y]
                    idx += 1
        if idx < N:
            pos[idx] = [0.5, 0.5]
        
        min_d = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                min_d = min(min_d, d)
        for i in range(N):
            min_d = min(min_d, pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
        
        radii = np.full(N, min_d * 0.9)
        pos, radii = _fix_boundaries(pos, radii)
        return pos, radii
    
    def _optimize_single(pos, radii, time_budget):
        """Optimize a single configuration using L-BFGS-B."""
        x0 = pos.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        def obj(x):
            return _objective_penalty(x, radii)
        
        def grad(x):
            return _objective_grad(x, radii)
        
        best_x = x0.copy()
        best_val = obj(x0)
        
        start_time = time.time()
        max_iter = 5000
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B', jac=grad, bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.x is not None:
                if obj(result.x) < best_val:
                    best_x = result.x
                    best_val = obj(result.x)
        except Exception:
            pass
        
        # If still time, try to grow radii iteratively
        pos_opt = best_x.reshape(N, 2)
        radii_opt = radii.copy()
        
        while time.time() - start_time < time_budget * 0.5:
            # Try to grow all radii slightly
            new_radii = radii_opt * 1.005
            # Check if valid
            if _validate(pos_opt, new_radii):
                radii_opt = new_radii
            else:
                # Try to re-optimize positions with new radii
                x_new = pos_opt.flatten()
                try:
                    result = minimize(
                        lambda x: _objective_penalty(x, new_radii),
                        x_new, method='L-BFGS-B', 
                        jac=lambda x: _objective_grad(x, new_radii),
                        bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if result.x is not None:
                        pos_new = result.x.reshape(N, 2)
                        if _validate(pos_new, new_radii):
                            pos_opt = pos_new
                            radii_opt = new_radii
                        else:
                            # Partial growth
                            for frac in [0.8, 0.5, 0.2]:
                                trial_radii = radii_opt + (new_radii - radii_opt) * frac
                                if _validate(pos_opt, trial_radii):
                                    radii_opt = trial_radii
                                    break
                except Exception:
                    break
                break
        
        return pos_opt, radii_opt
    
    def _shake_and_refine(pos, radii, time_budget):
        """Perturb and re-optimize to escape local minima."""
        best_pos = pos.copy()
        best_radii = radii.copy()
        best_score = np.sum(radii)
        
        start_time = time.time()
        num_shakes = 0
        
        while time.time() - start_time < time_budget:
            num_shakes += 1
            # Create perturbed version
            shake = 0.05
            pos_shake = pos + np.random.normal(0, shake, (N, 2))
            pos_shake = np.clip(pos_shake, 0.05, 0.95)
            
            # Keep radii same initially
            pos_opt, radii_opt = _optimize_single(pos_shake, radii, time_budget * 0.3)
            
            score = np.sum(radii_opt)
            if score > best_score:
                best_score = score
                best_pos = pos_opt
                best_radii = radii_opt
            
            if _remaining() < 10:
                break
        
        return best_pos, best_radii
    
    # Generate multiple initial configurations
    inits = []
    
    # Hexagonal
    pos_hex, radii_hex = _hex_initial()
    inits.append((pos_hex, radii_hex))
    
    # Grid
    pos_grid, radii_grid = _grid_initial()
    inits.append((pos_grid, radii_grid))
    
    # Random with structured bias
    for _ in range(3):
        pos_rand = np.random.uniform(0.1, 0.9, (N, 2))
        # Sort by x for better structure
        order = np.argsort(pos_rand[:, 0])
        pos_rand = pos_rand[order]
        min_d = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos_rand[i] - pos_rand[j])
                min_d = min(min_d, d)
        for i in range(N):
            min_d = min(min_d, pos_rand[i, 0], 1 - pos_rand[i, 0], pos_rand[i, 1], 1 - pos_rand[i, 1])
        radii_rand = np.full(N, min_d * 0.9)
        pos_rand, radii_rand = _fix_boundaries(pos_rand, radii_rand)
        inits.append((pos_rand, radii_rand))
    
    # Optimize each configuration
    best_pos = inits[0][0]
    best_radii = inits[0][1]
    best_score = np.sum(best_radii)
    
    time_per_init = _remaining() / (len(inits) + 2)  # +2 for shake phases
    
    for pos_init, radii_init in inits:
        if _remaining() < 15:
            break
        pos_opt, radii_opt = _optimize_single(pos_init, radii_init, time_per_init)
        score = np.sum(radii_opt)
        if score > best_score:
            best_score = score
            best_pos = pos_opt
            best_radii = radii_opt
    
    # Shake and refine phase
    if _remaining() > 30:
        shake_time = min(_remaining() - 20, 60)
        best_pos, best_radii = _shake_and_refine(best_pos, best_radii, shake_time)
        best_score = np.sum(best_radii)
    
    # Final validity enforcement
    if not _validate(best_pos, best_radii):
        best_pos, best_radii = _fix_boundaries(best_pos, best_radii)
    
    # Final safety check
    if not _validate(best_pos, best_radii):
        # Absolute fallback
        best_pos = np.random.uniform(0.1, 0.9, (N, 2))
        best_radii = np.full(N, 0.02)
        best_pos, best_radii = _fix_boundaries(best_pos, best_radii)
    
    result = np.column_stack((best_pos, best_radii))
    return result
