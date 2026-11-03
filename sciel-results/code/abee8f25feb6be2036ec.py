import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s kill
    
    def get_score(centers):
        """
        Compute the score: min(min_pair_dist/2, min_margin_to_boundary)
        Returns 0 if invalid (any center outside or NaN).
        """
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check boundary
        dists = np.linalg.norm(centers, axis=1)
        min_margin = 1.0 - np.max(dists)
        if min_margin <= 0:
            return 0.0
        
        # Check pair distances
        # Vectorized pair distance calculation
        # D[i, j] = ||c_i - c_j||
        # To save memory and time, we can do it in chunks or just full if N=50 is small enough
        # N=50 -> 50x50 matrix is 2500 floats, very small.
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            diff = centers - centers[i]
            dist_matrix[i, i+1:] = np.linalg.norm(diff[i+1:], axis=1)
            dist_matrix[i+1:, i] = dist_matrix[i, i+1:]
        
        min_dist = np.min(dist_matrix[~np.eye(N, dtype=bool)])
        
        return min(min_dist / 2.0, min_margin)

    def hexagonal_initial():
        """Generate a hexagonal lattice initial guess."""
        # Approximate density of hex packing is pi/(2*sqrt(3)) ~ 0.9069
        # Area of unit circle is pi. Area of 50 circles is 50*pi*r^2.
        # 50*r^2 ~ 0.9069 => r ~ sqrt(0.9069/50) ~ 0.134
        # Let's try a grid that fits.
        # Simple fallback: random points
        return np.random.uniform(-0.8, 0.8, (N, 2))

    def simulated_annealing(initial_centers, time_budget):
        """
        Run Simulated Annealing.
        """
        current_centers = initial_centers.copy()
        current_score = get_score(current_centers)
        
        # If initial is invalid, push it in
        norms = np.linalg.norm(current_centers, axis=1)
        if np.any(norms >= 1.0):
            mask = norms >= 1.0
            current_centers[mask] *= 0.99 / norms[mask]
            current_score = get_score(current_centers)
        
        best_centers = current_centers.copy()
        best_score = current_score
        
        # SA Parameters
        T_start = 0.05
        T_end = 1e-6
        alpha = 0.9995  # Cooling factor
        # We will run until time budget is exhausted or T is very low
        
        # Estimate steps per second to determine max steps?
        # Or just loop with time check.
        
        while time.time() - start_time < time_budget:
            T = max(T_end, T_start * (alpha ** (time.time() - start_time) * 100)) # Rough mapping
            # Actually, better to use a step counter for consistent cooling
            # Let's use a step-based approach but check time.
            
            # Generate perturbation
            # Scale step size with T
            step_size = 0.01 * T / T_start + 0.001
            new_centers = current_centers + np.random.normal(0, step_size, (N, 2))
            
            # Keep inside circle? 
            # If a point goes outside, the score drops to 0.
            # It's better to project back or just let the score penalize it.
            # Let's just let the score handle it, but if it's 0, we might get stuck.
            # A common trick: if new_score is 0, try a smaller move or project.
            
            new_score = get_score(new_centers)
            
            # Metropolis criterion
            if new_score > current_score:
                current_centers = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_centers = current_centers.copy()
                    best_score = current_score
            else:
                # Accept with probability exp((new - current) / T)
                # Note: We are maximizing score. So delta = new - current (negative).
                # prob = exp(delta / T)
                delta = new_score - current_score
                if delta < 0:
                    # If T is very small, exp is 0
                    if T > 1e-10:
                        prob = np.exp(delta / T)
                        if np.random.random() < prob:
                            current_centers = new_centers
                            current_score = new_score
                # If delta >= 0, it would have been accepted above
                
            # Small chance to "kick" the system if stuck?
            # Maybe not necessary with good cooling.
            
        return best_centers, best_score

    # 1. Generate diverse initial guesses
    inits = []
    
    # Guess 1: Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.8, 0.8, (N, 2)))
    
    # Guess 2: Ring-based (similar to current but free angles)
    # 1 center, 6 ring, 12 ring, 12 ring, 19 ring
    centers = [[0.0, 0.0]]
    radii = [0.25, 0.5, 0.75, 0.95]
    counts = [6, 12, 12, 19]
    for r, n in zip(radii, counts):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        centers.extend(zip(xs, ys))
    inits.append(np.array(centers))
    
    # Guess 3: Another ring variation
    centers = []
    # 2 centers? No, keep 1 center for simplicity in init
    centers.append([0.0, 0.0])
    radii = [0.2, 0.4, 0.6, 0.8, 0.98]
    counts = [6, 12, 12, 10, 10] # Total 50? 1+6+12+12+10+10 = 51. Adjust.
    # Let's just use random for others to save code complexity
    np.random.seed(123)
    inits.append(np.random.uniform(-0.8, 0.8, (N, 2)))
    
    np.random.seed(999)
    inits.append(np.random.uniform(-0.8, 0.8, (N, 2)))

    best_overall_centers = None
    best_overall_score = 0.0
    
    # Run SA for each init with a time slice
    # Total time ~ 120s. 4 inits -> 30s each? 
    # Or run one for longer?
    # Let's run 3 inits for 25s each, leaving 15s for polishing.
    
    time_per_init = 25.0
    
    for i, init in enumerate(inits):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Ensure init is valid (inside circle)
        norms = np.linalg.norm(init, axis=1)
        mask = norms >= 1.0
        if np.any(mask):
            init[mask] *= 0.99 / norms[mask]
            
        # Run SA
        # We pass the current global time start to SA? 
        # SA needs to know when to stop. 
        # Let's modify SA to take a deadline.
        
        # Refactor SA to accept deadline
        def sa_with_deadline(initial_centers, deadline):
            current_centers = initial_centers.copy()
            current_score = get_score(current_centers)
            if current_score == 0:
                # Force inside
                norms = np.linalg.norm(current_centers, axis=1)
                mask = norms >= 1.0
                if np.any(mask):
                    current_centers[mask] *= 0.99 / norms[mask]
                current_score = get_score(current_centers)
                
            best_centers = current_centers.copy()
            best_score = current_score
            
            T = 0.05
            alpha = 0.999
            min_T = 1e-7
            
            # Pre-calculate step size scaling
            # We want step size to go from ~0.05 down to ~0.001
            # step = base * (T / T_start) + min_step
            
            while time.time() < deadline:
                T *= alpha
                if T < min_T:
                    T = min_T
                    
                step_size = 0.02 * (T / 0.05) + 0.0005
                new_centers = current_centers + np.random.normal(0, step_size, (N, 2))
                
                # Optional: Project out-of-bound points back to boundary to maintain validity?
                # No, let score handle it. But if score is 0, we might drift.
                # If new_score is 0, we can try to repair it.
                
                new_score = get_score(new_centers)
                
                if new_score > current_score:
                    current_centers = new_centers
                    current_score = new_score
                    if current_score > best_score:
                        best_centers = current_centers.copy()
                        best_score = current_score
                else:
                    delta = new_score - current_score
                    if delta < 0 and T > 1e-10:
                        # Avoid overflow in exp
                        # delta is negative.
                        # If delta/T is very negative, exp is 0.
                        # Clip delta/T to avoid underflow issues if necessary, but exp handles it.
                        prob = np.exp(delta / T)
                        if np.random.random() < prob:
                            current_centers = new_centers
                            current_score = new_score
                            
            return best_centers, best_score

        deadline = time.time() + time_per_init
        c, s = sa_with_deadline(init, deadline)
        
        if s > best_overall_score:
            best_overall_score = s
            best_overall_centers = c

    # 2. Polishing with Local Optimization
    # Use the best result from SA as start point for Nelder-Mead
    if best_overall_centers is not None and time.time() < time_limit - 5:
        
        def objective(centers_flat):
            centers = centers_flat.reshape((N, 2))
            score = get_score(centers)
            # We want to maximize score. Minimize -score.
            # If score is 0, add penalty for being outside.
            dists = np.linalg.norm(centers, axis=1)
            violation = np.sum(np.maximum(0, dists - 1.0))
            return -score + 10.0 * violation

        initial_flat = best_overall_centers.flatten()
        
        try:
            res = minimize(
                objective, 
                initial_flat, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            polished_centers = res.x.reshape((N, 2))
            polished_score = get_score(polished_centers)
            
            if polished_score > best_overall_score:
                best_overall_score = polished_score
                best_overall_centers = polished_centers
        except Exception:
            pass

    # 3. Final Validation and Return
    if best_overall_centers is None or len(best_overall_centers) != N or not np.all(np.isfinite(best_overall_centers)):
        # Fallback
        # Simple random valid config
        centers = np.random.uniform(-0.5, 0.5, (N, 2))
        return centers

    # Ensure strictly inside
    dists = np.linalg.norm(best_overall_centers, axis=1)
    max_d = np.max(dists)
    if max_d >= 1.0:
        # Scale down slightly
        scale = 0.999 / max_d
        best_overall_centers *= scale

    return best_overall_centers

# EVOLVE-BLOCK-END
