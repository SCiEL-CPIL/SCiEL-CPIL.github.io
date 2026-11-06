    def _simulated_annealing(centers, steps=2000, T_init=0.01, T_final=0.0001):
        """
        Simulated Annealing to escape local minima.
        """
        current_centers = centers.copy()
        current_score = _raw_score(current_centers)
        best_centers = current_centers.copy()
        best_score = current_score
        
        # Pre-compute indices for efficiency? 
        # For N=50, recalculating score is O(N^2) ~ 2500 ops. 
        # 2000 steps * 2500 ops = 5M ops. This is very fast in NumPy.
        
        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
                
            # Cooling schedule
            T = T_init * (T_final / T_init) ** (step / max(1, steps - 1))
            
            # Perturb one random circle
            i = np.random.randint(0, N)
            original_pos = current_centers[i].copy()
            
            # Random step size, decreasing with T? Or fixed small?
            # Let's use a step size proportional to T to allow larger moves early
            step_size = T * 5.0 
            if step_size < 0.001:
                step_size = 0.001
            if step_size > 0.05:
                step_size = 0.05
                
            dx = np.random.uniform(-step_size, step_size, size=2)
            new_pos = original_pos + dx
            
            # Keep inside unit circle (hard constraint for validity, but score handles soft)
            # If it goes outside, score drops significantly. 
            # We can project it back to the boundary to keep it "valid" in terms of not being invalid, 
            # but the score function `_raw_score` returns 0 if outside. 
            # To make SA effective, we should probably project to the circle if it goes outside,
            # or just let the score drop. Let's project to maintain a valid candidate structure.
            norm_new = np.linalg.norm(new_pos)
            if norm_new > 1.0:
                new_pos = new_pos / norm_new * 0.999
                
            current_centers[i] = new_pos
            new_score = _raw_score(current_centers)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_centers = current_centers.copy()
            else:
                # Reject move
                current_centers[i] = original_pos
                
        return best_centers, best_score
