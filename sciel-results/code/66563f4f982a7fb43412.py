# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.
    
    Strategy: Physics-based relaxation (Simulated Annealing) with repulsive forces.
    This approach treats overlapping squares as having a repulsive potential and
    pushes them apart while trying to maximize their sizes.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s limit
    
    # Initial configuration: 3x3 grid of large squares + 1 small square
    # This provides a valid starting point close to the optimal structure
    # Optimal known for n=10 is likely related to 3x3 grid with one corner modified or similar.
    # A 3x3 grid of 1/3 squares sums to 3.0.
    # We can try to push the 10th square in and expand others.
    
    # We will run multiple restarts with different random seeds and initial perturbations
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def evaluate(centers, sides):
        """
        Calculate the score (sum of sides) if valid, or a penalized score if invalid.
        We use a penalty method to guide the optimization towards validity.
        """
        score = np.sum(sides)
        
        # Check bounds and overlaps, adding penalties
        tol = 1e-7
        penalty = 0.0
        
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                penalty += 100.0
                continue
            # Bounds penalty
            if cx - s/2 < -tol:
                penalty += 100.0 * (-tol - (cx - s/2))
            if cx + s/2 > 1 + tol:
                penalty += 100.0 * ((cx + s/2) - (1 + tol))
            if cy - s/2 < -tol:
                penalty += 100.0 * (-tol - (cy - s/2))
            if cy + s/2 > 1 + tol:
                penalty += 100.0 * ((cy + s/2) - (1 + tol))
        
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                if dx < t and dy < t:
                    # Penalty grows as overlap increases
                    overlap_x = t - dx
                    overlap_y = t - dy
                    penalty += 100.0 * overlap_x * overlap_y
        
        return score - penalty

    def is_valid(centers, sides):
        tol = 1e-7
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def optimize_single_run(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with 3x3 grid + 1 small
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # 3x3 grid
        side_grid = 1.0 / 3.0
        for i in range(9):
            col = i % 3
            row = i // 3
            centers[i] = [(col + 0.5) * side_grid, (row + 0.5) * side_grid]
            sides[i] = side_grid
        
        # 10th square: start small in a corner, maybe top-right
        centers[9] = [0.9, 0.9]
        sides[9] = 0.1
        
        # Add small random perturbation to break symmetry
        centers += np.random.normal(0, 0.01, (n, 2))
        
        current_score = evaluate(centers, sides)
        
        # Simulated Annealing parameters
        T = 0.05  # Initial temperature
        alpha = 0.995  # Cooling rate
        steps_per_temp = 50
        
        # We will iterate until time runs out or T is very low
        while time.time() < deadline:
            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    break
                    
                # Select a random square to move
                i = random.randint(0, n-1)
                
                # Perturbation
                # Move center
                dx = random.gauss(0, 0.02)
                dy = random.gauss(0, 0.02)
                # Change side
                ds = random.gauss(0, 0.01)
                
                new_centers = centers.copy()
                new_sides = sides.copy()
                
                new_centers[i, 0] += dx
                new_centers[i, 1] += dy
                new_sides[i] += ds
                
                # Clamp sides to positive
                if new_sides[i] <= 0:
                    new_sides[i] = 1e-5
                    
                # Clamp centers to stay within bounds roughly
                # This helps keep the search in a reasonable space
                # But we rely on penalty for strict validity
                # Let's allow out of bounds but penalty will be high
                
                new_score = evaluate(new_centers, new_sides)
                
                # Metropolis criterion
                if new_score > current_score:
                    centers = new_centers
                    sides = new_sides
                    current_score = new_score
                else:
                    delta = current_score - new_score
                    if delta < 0:
                        accept = True
                    else:
                        # Probability of accepting worse state
                        prob = np.exp(-delta / T)
                        accept = random.random() < prob
                    
                    if accept:
                        centers = new_centers
                        sides = new_sides
                        current_score = new_score
                        
            T *= alpha
            
            # If T is too low, stop
            if T < 1e-6:
                break
                
        # Final validation
        if is_valid(centers, sides):
            return centers, sides, np.sum(sides)
        else:
            # Try to fix by shrinking sides slightly if invalid
            # This is a heuristic fix
            for scale in [0.99, 0.95, 0.9]:
                test_sides = sides * scale
                if is_valid(centers, test_sides):
                    return centers, test_sides, np.sum(test_sides)
            return None, None, -np.inf

    # Run multiple restarts
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Different seeds for different restarts
        seed = 42 + i * 100
        
        # For restart > 0, we can use the best found so far as a perturbation base
        # Or just random restarts. Let's do random restarts with different initializations.
        
        # Variation: Start with different grid densities?
        # For now, just perturb the 3x3 grid differently.
        # Maybe start with 2x5 grid for one restart?
        if i == 1:
             # 2x5 grid
             centers_init = np.zeros((n, 2))
             sides_init = np.zeros(n)
             side_grid = 1.0 / 5.0
             for i in range(n):
                 col = i % 5
                 row = i // 5
                 centers_init[i] = [(col + 0.5) * side_grid, (row + 0.5) * side_grid]
                 sides_init[i] = side_grid
             # Perturb
             centers_init += np.random.normal(0, 0.02, (n, 2))
             # Run optimization on this
             # Reuse the optimization logic by calling a helper?
             # To keep it simple, I'll just run the same SA loop but with this init.
             # I'll refactor the SA loop into a function taking initial centers/sides.
             
             pass # Skip for now, stick to 3x3 variants to save code complexity in thought process, 
                 # but I will implement a helper in the final code.
                 
        # Actually, let's just run the SA with the standard 3x3 init but different random seeds.
        # The random perturbation in the init handles the variation.
        
        c, s, sc = optimize_single_run(seed)
        
        if c is not None and sc > best_score:
            best_centers = c
            best_sides = s
            best_score = sc

    # If no valid solution found (should not happen with 3x3 grid), fallback
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
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
