"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using weighted-sum decomposition + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Track all evaluated points and their objective values
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # --- Phase 1: Sobol-like quasi-random initialization ---
        # Use deterministic quasi-random points for better coverage
        n_init = min(40, budget // 5)
        # Simple quasi-random: use golden ratio sequence for 4D
        for i in range(n_init):
            if time.time() > deadline:
                break
            # Use a simple LHS-like approach with fixed seeds
            x = np.array([(i + 0.5) / n_init * (lo[d] + hi[d]) + (hi[d] - lo[d]) * 
                         ((i * 7919 + d * 104729) % 1000) / 1000 for d in range(dim)])
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            if f is None:
                break

        # Add some pure random points
        rng = np.random.default_rng(42)
        n_rand = min(20, budget - used)
        for _ in range(n_rand):
            if time.time() > deadline or used >= budget:
                break
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        if used >= budget or time.time() > deadline:
            return all_X

        # --- Phase 2: Weighted-sum decomposition with local refinement ---
        # Generate a set of weight vectors covering the objective space
        # For 3 objectives, use a set of weight vectors on the simplex
        weight_vectors = []
        # Uniform distribution on the 2-simplex for 3 objectives
        n_weights = 30
        # Use a grid-like approach
        for i in range(n_weights + 1):
            for j in range(n_weights - i + 1):
                w1 = i / n_weights
                w2 = j / (n_weights - i) if n_weights - i > 0 else 0
                w3 = 1 - w1 - w2
                if w3 >= -1e-9:
                    w3 = max(0, w3)
                    # Normalize
                    s = w1 + w2 + w3
                    if s > 1e-9:
                        weight_vectors.append(np.array([w1/s, w2/s, w3/s]))
        
        # Deduplicate and limit
        weight_vectors = np.array(weight_vectors)
        # Keep only unique-ish vectors
        unique_w = []
        for w in weight_vectors:
            if not any(np.allclose(w, uw, atol=0.05) for uw in unique_w):
                unique_w.append(w)
        weight_vectors = np.array(unique_w[:50])  # Limit to 50 weight vectors

        # For each weight vector, do a local search
        for w in weight_vectors:
            if time.time() > deadline or used >= budget:
                break
            
            # Find the best starting point for this weight
            # Use the current non-dominated set
            if len(all_F) == 0:
                continue
            
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            # Compute weighted sum for each point
            ws = F_arr @ w
            best_idx = np.argmin(ws)
            x0 = X_arr[best_idx].copy()
            
            # Local refinement using a simple Nelder-Mead-like approach
            # Or use gradient-free local search
            best_x = x0.copy()
            best_f = F_arr[best_idx].copy()
            best_ws = ws[best_idx]
            
            # Local search with adaptive step size
            step_size = 0.1 * (hi - lo)
            n_local = 0
            max_local = min(15, budget - used)
            
            for _ in range(max_local):
                if time.time() > deadline or used >= budget:
                    break
                
                # Try several directions
                improved = False
                for attempt in range(4):
                    if time.time() > deadline or used >= budget:
                        break
                    
                    # Random direction
                    direction = rng.standard_normal(dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    
                    # Try different step sizes
                    for scale in [step_size, step_size * 0.5, step_size * 2]:
                        if time.time() > deadline or used >= budget:
                            break
                        
                        x_new = best_x + scale * direction
                        x_new = np.clip(x_new, lo, hi)
                        
                        # Check if we've already evaluated this point (approx)
                        if any(np.allclose(x_new, px, atol=1e-4) for px in all_X[-10:]):
                            continue
                        
                        f_new = evaluate(x_new)
                        if f_new is None:
                            break
                        
                        ws_new = f_new @ w
                        if ws_new < best_ws - 1e-8:
                            best_x = x_new.copy()
                            best_f = f_new.copy()
                            best_ws = ws_new
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if not improved:
                    # Reduce step size
                    step_size *= 0.5
                    if step_size < 1e-5 * (hi - lo).max():
                        break
                else:
                    # Keep or slightly increase step size
                    step_size = min(step_size * 1.2, 0.2 * (hi - lo).max())
                    n_local += 1

        # --- Phase 3: Final random exploration with remaining budget ---
        while used < budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        return all_X
# EVOLVE-BLOCK-END
