"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True
    
    def compute_radii_from_centers(centers):
        """
        Given circle centers, compute maximum non-overlapping radii using
        all pairwise distances and boundary distances.
        Returns array of radii.
        """
        centers = np.asarray(centers)
        n = len(centers)
        radii = np.zeros(n)
        
        # Compute boundary distances
        for i in range(n):
            d_left = centers[i, 0]
            d_right = 1.0 - centers[i, 0]
            d_bottom = centers[i, 1]
            d_top = 1.0 - centers[i, 1]
            radii[i] = min(d_left, d_right, d_bottom, d_top)
        
        # For each pair, enforce non-overlap constraint
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Each circle can have at most half the distance to neighbor
                max_r = dist / 2.0
                radii[i] = min(radii[i], max_r)
                radii[j] = min(radii[j], max_r)
        
        return radii
    
    def objective_value(centers):
        """Return negative sum of radii (to maximize)."""
        radii = compute_radii_from_centers(centers)
        return -np.sum(radii)
    
    def local_search(centers, max_iter=500, step_scale=0.05):
        """
        Discrete best-fit local search: try moving each circle in 8 directions,
        accept if it improves the objective.
        """
        current = centers.copy()
        current_obj = objective_value(current)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try each circle
            for i in range(N):
                # Try 8 directions + stay
                for dx in [-step_scale, 0, step_scale]:
                    for dy in [-step_scale, 0, step_scale]:
                        if dx == 0 and dy == 0:
                            continue
                        new_centers = current.copy()
                        new_x = current[i, 0] + dx
                        new_y = current[i, 1] + dy
                        # Keep within bounds
                        new_x = np.clip(new_x, 0.02, 0.98)
                        new_y = np.clip(new_y, 0.02, 0.98)
                        new_centers[i, 0] = new_x
                        new_centers[i, 1] = new_y
                        
                        new_obj = objective_value(new_centers)
                        if new_obj < current_obj - 1e-10:
                            current = new_centers
                            current_obj = new_obj
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # No improvement; try smaller steps
                step_scale *= 0.7
                if step_scale < 0.001:
                    break
        
        return current, current_obj
    
    def gradient_polish(centers, max_iter=200):
        """
        Gradient-based polish using finite differences.
        """
        current = centers.copy()
        current_obj = objective_value(current)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient via central differences
            grad = np.zeros_like(current.flatten())
            eps = 1e-5
            for k in range(len(current.flatten())):
                center_plus = current.copy().flatten()
                center_minus = current.copy().flatten()
                center_plus[k] += eps
                center_minus[k] -= eps
                
                c_plus = center_plus.reshape(-1, 2)
                c_minus = center_minus.reshape(-1, 2)
                
                # Keep within bounds
                c_plus = np.clip(c_plus, 0.01, 0.99)
                c_minus = np.clip(c_minus, 0.01, 0.99)
                
                obj_plus = objective_value(c_plus)
                obj_minus = objective_value(c_minus)
                grad[k] = (obj_plus - obj_minus) / (2 * eps)
            
            # Gradient descent step
            lr = 0.01
            new_flat = current.flatten() - lr * grad
            new_centers = new_flat.reshape(-1, 2)
            new_centers = np.clip(new_centers, 0.01, 0.99)
            
            new_obj = objective_value(new_centers)
            if new_obj < current_obj - 1e-12:
                current = new_centers
                current_obj = new_obj
            else:
                # Reduce learning rate
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        return current, current_obj
    
    def initial_grid():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])
    
    def random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))
    
    def simulated_annealing(init_centers, max_iter=2000, t_start=0.1, t_end=0.001):
        """
        Simulated annealing to escape local optima.
        """
        current = init_centers.copy()
        current_obj = objective_value(current)
        best = current.copy()
        best_obj = current_obj
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max_iter)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Propose a move: pick a random circle and perturb it
            i = np.random.randint(N)
            dx = np.random.normal(0, 0.03)
            dy = np.random.normal(0, 0.03)
            
            new_centers = current.copy()
            new_x = current[i, 0] + dx
            new_y = current[i, 1] + dy
            new_x = np.clip(new_x, 0.02, 0.98)
            new_y = np.clip(new_y, 0.02, 0.98)
            new_centers[i, 0] = new_x
            new_centers[i, 1] = new_y
            
            new_obj = objective_value(new_centers)
            delta = new_obj - current_obj
            
            # Accept if better, or with probability exp(-delta/t)
            if delta < 0 or np.random.random() < np.exp(-delta / t):
                current = new_centers
                current_obj = new_obj
                
                if current_obj < best_obj:
                    best = current.copy()
                    best_obj = current_obj
            
            t *= cooling
        
        return best, best_obj
    
    # Strategy: Simulated annealing with local search polish
    best_result = None
    best_score = np.inf
    
    # Multiple initializations
    inits = [initial_grid()]
    for seed in [1, 7, 13, 23, 37, 53, 71, 89]:
        inits.append(random_init(seed))
    
    # Run simulated annealing from each initialization
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        try:
            # Phase 1: Simulated annealing to find good region
            sa_centers, sa_obj = simulated_annealing(init_centers, max_iter=1000)
            
            # Phase 2: Local search to polish
            ls_centers, ls_obj = local_search(sa_centers, max_iter=300)
            
            # Phase 3: Gradient polish for final refinement
            gp_centers, gp_obj = gradient_polish(ls_centers, max_iter=100)
            
            # Compute final radii
            radii = compute_radii_from_centers(gp_centers)
            x_full = np.column_stack([gp_centers, radii])
            
            if is_valid(x_full, tol=1e-6):
                score = -np.sum(radii)
                if score < best_score:
                    best_score = score
                    best_result = x_full.copy()
        except Exception:
            continue
    
    # If no valid result found, fall back to grid
    if best_result is None:
        centers = initial_grid()
        radii = compute_radii_from_centers(centers)
        best_result = np.column_stack([centers, radii])
    
    # Additional refinement: try perturbations around best
    if time.time() < deadline - 10:
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(100 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.02, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.02, 0.98)
            
            try:
                sa_centers, sa_obj = simulated_annealing(perturbed, max_iter=500)
                ls_centers, ls_obj = local_search(sa_centers, max_iter=200)
                gp_centers, gp_obj = gradient_polish(ls_centers, max_iter=50)
                
                radii = compute_radii_from_centers(gp_centers)
                x_try = np.column_stack([gp_centers, radii])
                
                if is_valid(x_try, tol=1e-6):
                    score = -np.sum(radii)
                    if score < best_score:
                        best_score = score
                        best_result = x_try.copy()
            except Exception:
                continue
    
    # Ensure final validity
    if best_result is None or not is_valid(best_result, tol=1e-6):
        centers = initial_grid()
        radii = compute_radii_from_centers(centers)
        best_result = np.column_stack([centers, radii])
    
    return best_result

# EVOLVE-BLOCK-END
