# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2, maximizing
    the sum of their side lengths, with pairwise disjoint interiors.

    Approach:
    1. Start from a good initial packing (e.g., a grid-like arrangement).
    2. Use nonlinear optimization to maximize sum(sides) subject to:
       - Each square inside [0,1]^2
       - Non-overlap constraints (linearized via big-M or pairwise separation)
    3. Return centers and sides.

    We use a sequential pairwise separation approach: for each pair (i,j),
    enforce that they are separated in at least one axis. This is a disjunctive
    constraint. We handle it by trying different separation patterns and
    optimizing for each, then keeping the best.

    For 10 squares, we can use a simpler approach: start with a known good
    packing and refine with local optimization.

    Let's use a different strategy: we'll use a simulated annealing / local
    search approach with random restarts, which is more robust for this kind
    of combinatorial + continuous problem.
    """
    n = 10
    
    def compute_score(centers, sides):
        """Compute the sum of sides, or -inf if infeasible."""
        if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(sides)):
            return -np.inf
        if np.any(sides <= 0):
            return -np.inf
        
        # Check bounds
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -1e-7 or cx + s/2 > 1 + 1e-7:
                return -np.inf
            if cy - s/2 < -1e-7 or cy + s/2 > 1 + 1e-7:
                return -np.inf
        
        # Check overlaps
        tol = 1e-7
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return -np.inf
        
        return np.sum(sides)
    
    def random_init():
        """Generate a random feasible-ish initial solution."""
        # Start with a grid and add small random perturbations
        k = 4  # 4x4 grid, use 10 cells
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side + np.random.uniform(-0.02, 0.02, n)
        cy = (rows + 0.5) * side + np.random.uniform(-0.02, 0.02, n)
        sides = np.full(n, side, dtype=np.float64)
        
        # Clamp to ensure feasibility
        for i in range(n):
            cx[i] = np.clip(cx[i], sides[i]/2, 1 - sides[i]/2)
            cy[i] = np.clip(cy[i], sides[i]/2, 1 - sides[i]/2)
        
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        return centers, sides
    
    def local_search(centers, sides, max_iters=5000, lr=0.01):
        """Simple local search with random perturbations."""
        best_score = compute_score(centers, sides)
        best_centers = centers.copy()
        best_sides = sides.copy()
        
        current_score = best_score
        current_centers = centers.copy()
        current_sides = sides.copy()
        
        for it in range(max_iters):
            # Perturb one square
            i = np.random.randint(n)
            
            # Try changing position
            if np.random.random() < 0.5:
                delta_x = np.random.normal(0, lr)
                delta_y = np.random.normal(0, lr)
                new_cx = current_centers[i, 0] + delta_x
                new_cy = current_centers[i, 1] + delta_y
                
                # Clamp
                s = current_sides[i]
                new_cx = np.clip(new_cx, s/2, 1 - s/2)
                new_cy = np.clip(new_cy, s/2, 1 - s/2)
                
                trial_centers = current_centers.copy()
                trial_centers[i] = [new_cx, new_cy]
                trial_sides = current_sides.copy()
            else:
                # Try changing size
                delta_s = np.random.normal(0, lr)
                new_s = current_sides[i] + delta_s
                new_s = np.clip(new_s, 0.01, 1.0)
                
                # Adjust center to keep square inside
                cx = current_centers[i, 0]
                cy = current_centers[i, 1]
                cx = np.clip(cx, new_s/2, 1 - new_s/2)
                cy = np.clip(cy, new_s/2, 1 - new_s/2)
                
                trial_centers = current_centers.copy()
                trial_centers[i] = [cx, cy]
                trial_sides = current_sides.copy()
                trial_sides[i] = new_s
            
            trial_score = compute_score(trial_centers, trial_sides)
            
            if trial_score > current_score:
                current_score = trial_score
                current_centers = trial_centers
                current_sides = trial_sides
                
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
                    best_sides = current_sides.copy()
            else:
                # Accept with some probability (simulated annealing-like)
                if np.random.random() < 0.1:
                    current_centers = trial_centers
                    current_sides = trial_sides
                    current_score = trial_score
            
            # Decrease learning rate
            lr = max(0.001, lr * 0.999)
        
        return best_centers, best_sides, best_score
    
    # Run multiple restarts
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    np.random.seed(42)
    
    for restart in range(20):
        centers, sides = random_init()
        centers, sides, score = local_search(centers, sides, max_iters=2000)
        
        if score > best_score:
            best_score = score
            best_centers = centers
            best_sides = sides
    
    # Final verification
    final_score = compute_score(best_centers, best_sides)
    if final_score < 0:
        # Fallback to grid
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
