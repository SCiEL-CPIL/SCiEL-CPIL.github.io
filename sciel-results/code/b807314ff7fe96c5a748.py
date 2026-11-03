"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    
    def compute_radii_from_centers(centers, margin=0.0):
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
        
        # Enforce non-overlap for all pairs
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = dist / 2.0
                radii[i] = min(radii[i], max_r)
                radii[j] = min(radii[j], max_r)
        
        # Apply margin if needed
        if margin > 0:
            radii = np.maximum(0.0, radii - margin)
        
        return radii
    
    def objective_only(centers_flat):
        """Just the objective for L-BFGS-B."""
        centers = centers_flat.reshape(-1, 2)
        radii = compute_radii_from_centers(centers)
        return -np.sum(radii)
    
    def initial_grid_centers():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])
    
    def random_centers(seed=0):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        return centers
    
    def local_search(centers, max_iters=5000, step_size=0.02, seed=0):
        """
        Discrete best-fit local search: try small perturbations to centers,
        accept if it improves the sum of radii.
        """
        rng = np.random.RandomState(seed)
        current_centers = centers.copy()
        current_radii = compute_radii_from_centers(current_centers)
        current_score = np.sum(current_radii)
        
        best_centers = current_centers.copy()
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random circle and try to move it
            i = rng.randint(N)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            
            new_centers = current_centers.copy()
            new_centers[i, 0] = np.clip(new_centers[i, 0] + dx, 0.02, 0.98)
            new_centers[i, 1] = np.clip(new_centers[i, 1] + dy, 0.02, 0.98)
            
            new_radii = compute_radii_from_centers(new_centers)
            new_score = np.sum(new_radii)
            
            if new_score > current_score + 1e-10:
                current_centers = new_centers
                current_radii = new_radii
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
        
        return best_centers
    
    def gradient_polish(centers, max_iters=100):
        """
        Final high-precision polish using L-BFGS-B with tighter tolerances.
        """
        try:
            result = minimize(
                objective_only,
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * N),
                options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            opt_centers = result.x.reshape(-1, 2)
            radii = compute_radii_from_centers(opt_centers)
            x_full = np.column_stack([opt_centers, radii])
            if is_valid(x_full, tol=1e-6):
                return x_full
        except Exception:
            pass
        return None
    
    # Strategy: Multiple initializations with local search + gradient polish
    best_result = None
    best_score = -np.inf
    
    # Multiple initializations
    inits = []
    
    # Grid initialization
    inits.append(initial_grid_centers())
    
    # Random initializations with different seeds
    for seed in [1, 7, 13, 23, 37, 53, 71, 89, 101, 131]:
        inits.append(random_centers(seed))
    
    # Also try some structured patterns
    # Hexagonal-like pattern
    hex_centers = []
    for i in range(6):
        for j in range(5):
            if len(hex_centers) < N:
                x = 0.1 + i * 0.16 + (0.08 if j % 2 else 0)
                y = 0.1 + j * 0.17
                if x < 0.9 and y < 0.9:
                    hex_centers.append([x, y])
    if len(hex_centers) >= N:
        inits.append(np.array(hex_centers[:N]))
    
    # Optimize each initialization
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Step 1: Local search
        ls_centers = local_search(init_centers, max_iters=3000, step_size=0.03, seed=init_idx)
        ls_radii = compute_radii_from_centers(ls_centers)
        ls_score = np.sum(ls_radii)
        
        # Step 2: Gradient polish
        polished = gradient_polish(ls_centers, max_iters=150)
        
        if polished is not None:
            p_score = np.sum(polished[:, 2])
            if p_score > best_score:
                best_score = p_score
                best_result = polished.copy()
        else:
            # Use local search result if polish failed
            x_ls = np.column_stack([ls_centers, ls_radii])
            if is_valid(x_ls, tol=1e-6) and ls_score > best_score:
                best_score = ls_score
                best_result = x_ls.copy()
    
    # If no valid result found, fall back to grid
    if best_result is None:
        centers = initial_grid_centers()
        radii = compute_radii_from_centers(centers)
        best_result = np.column_stack([centers, radii])
    
    # Final refinement: more local search from best result
    if best_result is not None and time.time() < deadline - 15:
        x_ref = best_result.copy()
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            # Perturb centers slightly
            rng = np.random.RandomState(100 + trial)
            perturbed = x_ref[:, :2].copy()
            perturbed += rng.normal(0, 0.015, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.02, 0.98)
            
            # Local search from perturbed
            ls_centers = local_search(perturbed, max_iters=2000, step_size=0.02, seed=200 + trial)
            ls_radii = compute_radii_from_centers(ls_centers)
            ls_score = np.sum(ls_radii)
            
            # Gradient polish
            polished = gradient_polish(ls_centers, max_iters=100)
            
            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            elif is_valid(np.column_stack([ls_centers, ls_radii]), tol=1e-6) and ls_score > best_score:
                best_score = ls_score
                best_result = np.column_stack([ls_centers, ls_radii]).copy()
    
    # Ensure final validity
    if best_result is None or not is_valid(best_result, tol=1e-6):
        # Last resort: very conservative grid
        centers = initial_grid_centers()
        radii = compute_radii_from_centers(centers, margin=0.001)
        best_result = np.column_stack([centers, radii])
    
    return best_result

# EVOLVE-BLOCK-END
