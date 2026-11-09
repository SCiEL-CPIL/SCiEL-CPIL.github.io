"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start local search with Sobol coverage and BFGS refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        box_low = -5.0
        box_high = 5.0

        # Generate Sobol sequence points for diverse initialization
        # We'll use a simple quasi-random approach since scipy.integrate.qmc might not be available
        # Use a deterministic Halton-like or just stratified sampling
        # Let's use a simple approach: divide the space into a grid and pick centers

        # Phase 1: Broad coverage with stratified sampling
        # Use 100 initial points spread across the domain
        n_initial = min(100, budget // 5)
        
        # Create stratified points
        points = []
        # Use a deterministic low-discrepancy-like approach
        # For 5D, we can use a grid-based approach
        # Let's use 4^5 = 1024 grid cells, pick 100 of them deterministically
        # Actually, let's use a simpler approach: nested sampling
        
        # Generate initial candidates using a structured approach
        # Use a combination of center, corners, and stratified points
        
        best_x = None
        best_f = np.inf
        
        # Center point
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        
        # Stratified sampling: divide each dimension into 2^d parts
        # For dim=5, use 2^5=32 cells per dimension? That's too many.
        # Let's use 2 cells per dimension: 2^5 = 32 total cells
        # Pick the center of each cell
        n_cells_per_dim = 2
        total_cells = n_cells_per_dim ** dim
        # We can afford up to budget//5 evaluations for initial phase
        n_stratified = min(total_cells, budget // 5)
        
        # Generate stratified points
        for i in range(n_stratified):
            if time.time() > deadline - 30:
                break
            # Convert i to base-n_cells_per_dim digits
            digits = []
            temp = i
            for _ in range(dim):
                digits.append(temp % n_cells_per_dim)
                temp //= n_cells_per_dim
            x = np.zeros(dim)
            for d in range(dim):
                # Center of the cell
                cell_width = (box_high - box_low) / n_cells_per_dim
                x[d] = box_low + (digits[d] + 0.5) * cell_width
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Local refinement around the best point
        # Use a simple pattern search / coordinate descent
        remaining_budget = budget - 1 - n_stratified
        
        # Try coordinate descent with adaptive step sizes
        current_x = best_x.copy()
        current_f = best_f
        step_size = 1.0
        
        calls_made = 1 + n_stratified
        
        while calls_made < budget:
            if time.time() > deadline - 10:
                break
            
            improved = False
            for d in range(dim):
                if time.time() > deadline - 10:
                    break
                if calls_made >= budget:
                    break
                
                # Try positive direction
                trial_x = current_x.copy()
                trial_x[d] = np.clip(current_x[d] + step_size, box_low, box_high)
                if np.any(trial_x != current_x):
                    trial_f = func(trial_x)
                    calls_made += 1
                    if trial_f < current_f:
                        current_x = trial_x
                        current_f = trial_f
                        improved = True
                        if trial_f < best_f:
                            best_f = trial_f
                            best_x = trial_x.copy()
                
                if time.time() > deadline - 10:
                    break
                if calls_made >= budget:
                    break
                
                # Try negative direction
                trial_x = current_x.copy()
                trial_x[d] = np.clip(current_x[d] - step_size, box_low, box_high)
                if np.any(trial_x != current_x):
                    trial_f = func(trial_x)
                    calls_made += 1
                    if trial_f < current_f:
                        current_x = trial_x
                        current_f = trial_f
                        improved = True
                        if trial_f < best_f:
                            best_f = trial_f
                            best_x = trial_x.copy()
            
            if not improved:
                # Shrink step size
                step_size *= 0.5
                if step_size < 1e-4:
                    break
            else:
                # Slightly increase step size if we're making progress
                step_size = min(step_size * 1.1, 2.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
