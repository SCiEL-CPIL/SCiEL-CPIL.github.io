import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def _compute_distances(pos):
        """Compute pairwise distances between all circles."""
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, np.inf)
        return dist
    
    def _objective(pos, W):
        """Compute the objective: sum of radii, or -inf if invalid."""
        H = 2.0 - W
        if W <= 0.1 or H <= 0.1:
            return -np.inf
        if np.any(pos[:, 0] < 0) or np.any(pos[:, 0] > W) or np.any(pos[:, 1] < 0) or np.any(pos[:, 1] > H):
            return -np.inf
        
        dist = _compute_distances(pos)
        min_dist_to_other = np.min(dist, axis=1) / 2.0
        
        dx = np.minimum(pos[:, 0], W - pos[:, 0])
        dy = np.minimum(pos[:, 1], H - pos[:, 1])
        dist_to_boundary = np.minimum(dx, dy)
        
        radii = np.minimum(dist_to_boundary, min_dist_to_other)
        return np.sum(radii)
    
    def _force_relax(pos, W, max_iter=500, tol=1e-10):
        """Force-directed relaxation to find local optimum."""
        H = 2.0 - W
        pos = pos.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute forces
            forces = np.zeros_like(pos)
            dist = _compute_distances(pos)
            
            # Repulsive forces between circles
            for i in range(N):
                for j in range(i + 1, N):
                    d = dist[i, j]
                    if d < 1e-10:
                        # Random push if overlapping
                        forces[i] += np.random.normal(0, 0.01, 2)
                        forces[j] -= np.random.normal(0, 0.01, 2)
                        continue
                    # Repulsion strength increases as circles get closer
                    repulsion = 1.0 / (d * d)
                    direction = (pos[i] - pos[j]) / d
                    forces[i] += repulsion * direction
                    forces[j] -= repulsion * direction
            
            # Attractive forces toward boundaries (to maximize radii)
            # Actually, we want to push circles toward corners to maximize distance from boundaries
            # But for now, let's just use repulsion and boundary constraints
            
            # Apply forces with damping
            damping = 0.1 / (1 + it * 0.01)
            pos += damping * forces
            
            # Constrain to bounds
            pos[:, 0] = np.clip(pos[:, 0], 0, W)
            pos[:, 1] = np.clip(pos[:, 1], 0, H)
            
            # Check convergence
            if np.max(np.abs(damping * forces)) < tol:
                break
        
        return pos
    
    def _polish_result(pos, W):
        """Final polish using local optimization."""
        H = 2.0 - W
        current_score = _objective(pos, W)
        
        # Try small perturbations
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            
            new_pos = pos.copy()
            idx = np.random.randint(N)
            new_pos[idx] += np.random.normal(0, 0.005, 2)
            new_pos[idx, 0] = np.clip(new_pos[idx, 0], 0, W)
            new_pos[idx, 1] = np.clip(new_pos[idx, 1], 0, H)
            
            new_score = _objective(new_pos, W)
            if new_score > current_score:
                pos = new_pos
                current_score = new_score
        
        return pos, current_score
    
    # Multiple restarts with different initial configurations and aspect ratios
    best_pos = None
    best_W = 1.0
    best_score = -np.inf
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.9, 1.1]
    
    for W in aspect_ratios:
        if time.time() > deadline - 10:
            break
        
        H = 2.0 - W
        
        # Initialize with grid
        side = int(np.ceil(np.sqrt(N)))
        xs = (np.arange(side) + 0.5) / side * W
        ys = (np.arange(side) + 0.5) / side * H
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
        init_pos = np.column_stack((grid_x.ravel()[:N], grid_y.ravel()[:N]))
        
        # Add noise
        init_pos += np.random.normal(0, 0.02, init_pos.shape)
        init_pos[:, 0] = np.clip(init_pos[:, 0], 0.01, W - 0.01)
        init_pos[:, 1] = np.clip(init_pos[:, 1], 0.01, H - 0.01)
        
        # Force relaxation
        relaxed_pos = _force_relax(init_pos, W)
        
        # Polish
        polished_pos, score = _polish_result(relaxed_pos, W)
        
        if score > best_score:
            best_score = score
            best_pos = polished_pos.copy()
            best_W = W
    
    # If we have time, try more random restarts
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        
        W = np.random.uniform(0.7, 1.3)
        H = 2.0 - W
        
        # Random initialization
        init_pos = np.random.uniform(0.1, 0.9, (N, 2))
        init_pos[:, 0] *= W
        init_pos[:, 1] *= H
        init_pos[:, 0] = np.clip(init_pos[:, 0], 0.01, W - 0.01)
        init_pos[:, 1] = np.clip(init_pos[:, 1], 0.01, H - 0.01)
        
        relaxed_pos = _force_relax(init_pos, W)
        polished_pos, score = _polish_result(relaxed_pos, W)
        
        if score > best_score:
            best_score = score
            best_pos = polished_pos.copy()
            best_W = W
    
    # Final refinement with SLSQP on the best solution
    if time.time() < deadline - 10:
        H = 2.0 - best_W
        
        def _slsqp_objective(flat_pos):
            pos = flat_pos.reshape(N, 2)
            score = _objective(pos, best_W)
            if score == -np.inf:
                return 1e6
            return -score  # Minimize negative
        
        constraints = []
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*2] - 0.0})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: best_W - x[i*2] - 0.0})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*2+1] - 0.0})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: H - x[i*2+1] - 0.0})
        
        # Add non-overlap constraints (approximate)
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: 
                        (x[i*2] - x[j*2])**2 + (x[i*2+1] - x[j*2+1])**2 - 1e-6
                })
        
        initial_flat = best_pos.ravel()
        
        try:
            result = minimize(
                _slsqp_objective,
                initial_flat,
                method='SLSQP',
                bounds=[(0, best_W)] * N + [(0, H)] * N,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < -best_score + 1e-6:
                new_pos = result.x.reshape(N, 2)
                new_score = _objective(new_pos, best_W)
                if new_score > best_score:
                    best_pos = new_pos
                    best_score = new_score
        except Exception:
            pass
    
    # Construct final array
    H = 2.0 - best_W
    dist = _compute_distances(best_pos)
    min_dist_to_other = np.min(dist, axis=1) / 2.0
    
    dx = np.minimum(best_pos[:, 0], best_W - best_pos[:, 0])
    dy = np.minimum(best_pos[:, 1], H - best_pos[:, 1])
    dist_to_boundary = np.minimum(dx, dy)
    
    radii = np.minimum(dist_to_boundary, min_dist_to_other)
    radii = np.maximum(radii, 0.0)
    
    result = np.column_stack((best_pos, radii))
    
    return result
