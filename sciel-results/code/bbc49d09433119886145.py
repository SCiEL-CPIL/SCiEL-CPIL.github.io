import numpy as np
import time
import random

def squares_in_square10() -> tuple:
    """
    Maximizes the total side length of 10 non-overlapping axis-aligned squares
    inside the unit square [0, 1]^2 using Simulated Annealing with hard constraints.
    
    Strategy:
    1. Represent state as (centers, sides).
    2. Use hard constraint checking: any move that violates non-overlap or boundaries is rejected immediately (except in early cooling if we allow slight tolerance? No, hard is better for validity).
    3. Actually, strict hard constraints can get stuck. 
       Better approach: 
       - Start with a valid configuration (e.g., small squares in a grid).
       - Moves: 
         a) Perturb position of one square.
         b) Perturb size of one square.
         c) Perturb both.
       - Feasibility: 
         Check if all squares are within [0,1] and non-overlapping.
         If invalid, reject.
       - To escape local minima, we might need to allow "temporary" invalid states? 
         No, the task requires valid output. 
         However, pure hill climbing on valid states gets stuck.
         Simulated Annealing typically accepts worse valid states to explore.
         But it cannot cross invalid regions.
         
    Alternative Structural Change: 
    Use a **Hybrid**: 
    1. Generate many diverse valid initial configurations (Grids, Corner packs).
    2. For each, run a fast **Local Search** (Hill Climbing) that only accepts moves that improve the objective (sum of sides) while maintaining validity.
    3. If stuck, apply a **Random Restart** or **Perturbation** (e.g., shrink all squares slightly to create space, then re-expand).
    
    Actually, the most robust "different" method for packing is **Simulated Annealing with a penalty-based objective** but using **Metropolis-Hastings** logic where we *always* check validity for the final return, but during search we might allow soft violations to escape traps? 
    No, the prompt says "Invalid result ... scores 0". 
    So we must return a valid result.
    
    Let's stick to **Strict Validity SA**:
    - State must always be valid.
    - Objective: Sum of sides.
    - Move: Randomly select a square, perturb its center and/or side.
    - Acceptance: 
      - If new state is valid and better (higher sum sides): Accept.
      - If new state is valid and worse: Accept with probability exp(-delta/T).
      - If new state is invalid: Reject.
    
    This is a random walk on the valid manifold. It can be slow to find the global max because the valid manifold might be disconnected or have narrow corridors.
    
    To improve this, we can use a **"Grow and Pack"** heuristic:
    1. Start with very small squares.
    2. Iteratively increase sizes while adjusting positions to maintain non-overlap.
    
    Let's implement a **Multi-start Local Search with Random Restarts and "Shake"**.
    1. Initialize with a valid configuration (e.g., 10 squares of size 0.1 in a grid).
    2. Local Search:
       - Try to increase the size of the smallest square.
       - If it overlaps, try to move it or move neighbors.
       - This is complex.
    
    Let's go back to **Simulated Annealing on Valid States**.
    Key insight: The space of valid configurations is large.
    We will use a **cooled** SA.
    
    To ensure we find a good solution:
    1. Create multiple diverse initial valid states.
       - State A: 10 squares of size 0.1, arranged in a 5x2 grid.
       - State B: 10 squares of size 0.1, arranged in a 4x3 grid (with 2 extra spots).
       - State C: Random valid placement of small squares.
    2. For each initial state, run SA for a fixed time budget.
    3. Return the best valid state found.
    
    To make it efficient:
    - Vectorized validity check.
    - Adaptive temperature.
    """
    n = 10
    start_time = time.time()
    deadline = 600.0
    hard_deadline = start_time + deadline - 10.0

    def is_valid(centers, sides, tol=1e-7):
        """
        Check if configuration is valid.
        Returns True if all squares are inside [0,1] and non-overlapping.
        """
        # Boundary check
        if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
            return False
        
        # Overlap check
        # |cx_i - cx_j| >= (s_i + s_j)/2 - tol OR |cy_i - cy_j| >= (s_i + s_j)/2 - tol
        # Violation if: |dx| < (s_i+s_j)/2 - tol AND |dy| < (s_i+s_j)/2 - tol
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = np.abs(centers[i_idx, 0] - centers[j_idx, 0])
        dy = np.abs(centers[i_idx, 1] - centers[j_idx, 1])
        t = (sides[i_idx] + sides[j_idx]) / 2.0 - tol
        
        # If t <= 0, they don't overlap even if centers coincide (impossible for positive sides, but safe)
        # Violation if dx < t AND dy < t
        violations = (dx < t) & (dy < t)
        
        return not np.any(violations)

    def get_score(centers, sides):
        return np.sum(sides)

    def random_valid_init(rng):
        """
        Generate a random valid initial configuration.
        Strategy: Place squares one by one in random positions, reducing size if collision.
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # Try to place 10 squares
        for i in range(n):
            placed = False
            # Start with a desired size, e.g., 0.3, shrink if needed
            max_attempts = 100
            for _ in range(max_attempts):
                # Try a random size between 0.1 and 0.4
                s = rng.uniform(0.1, 0.4)
                # Random position
                cx = rng.uniform(s/2, 1 - s/2)
                cy = rng.uniform(s/2, 1 - s/2)
                
                # Check against existing squares
                valid = True
                for j in range(i):
                    dx = abs(centers[j, 0] - cx)
                    dy = abs(centers[j, 1] - cy)
                    t = (sides[j] + s) / 2.0
                    if dx < t and dy < t:
                        valid = False
                        break
                
                if valid:
                    centers[i] = [cx, cy]
                    sides[i] = s
                    placed = True
                    break
            
            if not placed:
                # Fallback: tiny square
                s = 0.01
                # Find a spot (random)
                for _ in range(1000):
                    cx = rng.uniform(s/2, 1 - s/2)
                    cy = rng.uniform(s/2, 1 - s/2)
                    valid = True
                    for j in range(i):
                        dx = abs(centers[j, 0] - cx)
                        dy = abs(centers[j, 1] - cy)
                        t = (sides[j] + s) / 2.0
                        if dx < t and dy < t:
                            valid = False
                            break
                    if valid:
                        centers[i] = [cx, cy]
                        sides[i] = s
                        break
                else:
                    # If still can't place, put it at a corner with tiny size
                    centers[i] = [0.005, 0.005]
                    sides[i] = 0.01
                    
        return centers, sides

    def grid_init(rng, rows, cols):
        """
        Initialize with a grid pattern.
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # Calculate max side for grid
        # If we have R rows and C cols, and we need to fit N squares, 
        # some cells might be empty or we split cells.
        # Simpler: Just place N squares in a grid of max(rows, cols) x max(rows, cols)
        # Let's use a 4x4 grid concept, placing 10 squares.
        
        # For simplicity, let's just do a 5x2 or 4x3 layout.
        if rows == 5 and cols == 2:
            side = 1.0 / 5.0 * 0.95 # 5% margin
            idx = 0
            for r in range(5):
                for c in range(2):
                    if idx >= n: break
                    cx = (c + 0.5) * (1.0/cols)
                    cy = (r + 0.5) * (1.0/rows)
                    centers[idx] = [cx, cy]
                    sides[idx] = side
                    idx += 1
        elif rows == 4 and cols == 3:
            side = min(1.0/4.0, 1.0/3.0) * 0.95
            idx = 0
            for r in range(4):
                for c in range(3):
                    if idx >= n: break
                    cx = (c + 0.5) * (1.0/cols)
                    cy = (r + 0.5) * (1.0/rows)
                    centers[idx] = [cx, cy]
                    sides[idx] = side
                    idx += 1
        else:
            # Fallback random
            return random_valid_init(rng)
            
        # Add small random jitter to break symmetry
        centers += rng.uniform(-0.01, 0.01, (n, 2))
        
        # Ensure validity
        if not is_valid(centers, sides):
            # Shrink sides slightly
            sides *= 0.9
            if not is_valid(centers, sides):
                return random_valid_init(rng)
                
        return centers, sides

    def simulate_annealing(centers, sides, rng, time_limit=5.0):
        """
        Run SA on a valid initial state.
        Returns best centers, sides found.
        """
        current_centers = centers.copy()
        current_sides = sides.copy()
        current_score = get_score(current_centers, current_sides)
        
        best_centers = current_centers.copy()
        best_sides = current_sides.copy()
        best_score = current_score
        
        T = 0.05 # Initial temperature
        T_min = 0.001
        cooling_rate = 0.9995
        
        start = time.time()
        
        while T > T_min and time.time() - start < time_limit:
            # Select a square to move
            i = rng.integers(0, n)
            
            # Perturbation
            # Move center
            dx = rng.normal(0, 0.05)
            dy = rng.normal(0, 0.05)
            # Perturb size
            ds = rng.normal(0, 0.05)
            
            new_centers = current_centers.copy()
            new_sides = current_sides.copy()
            
            new_centers[i, 0] += dx
            new_centers[i, 1] += dy
            new_sides[i] += ds
            
            # Ensure side is positive
            if new_sides[i] <= 0.01:
                new_sides[i] = 0.01
            
            # Check validity
            if is_valid(new_centers, new_sides):
                new_score = get_score(new_centers, new_sides)
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    # Accept
                    current_centers = new_centers
                    current_sides = new_sides
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_centers = current_centers.copy()
                        best_sides = current_sides.copy()
            else:
                # Reject invalid move
                pass
            
            T *= cooling_rate
            
        return best_centers, best_sides, best_score

    # Main Loop
    rng = np.random.default_rng(42)
    
    best_global_centers = None
    best_global_sides = None
    best_global_score = -np.inf
    
    # List of initialization strategies
    inits = []
    
    # 1. Grid 5x2
    c, s = grid_init(rng, 5, 2)
    inits.append((c, s))
    
    # 2. Grid 4x3
    c, s = grid_init(rng, 4, 3)
    inits.append((c, s))
    
    # 3. Random valid
    c, s = random_valid_init(rng)
    inits.append((c, s))
    
    # 4. Another random valid
    c, s = random_valid_init(rng)
    inits.append((c, s))
    
    # 5. Corner pack: 4 corners with larger squares, 6 smaller in middle?
    # Let's just use random for now to keep it simple and robust.
    
    for init_c, init_s in inits:
        if time.time() > hard_deadline:
            break
            
        # Run SA for a portion of the remaining time
        # Divide time among initializations
        time_per_init = 10.0 # 10 seconds per init, 4 inits = 40s total. Plenty of time.
        
        b_c, b_s, b_score = simulate_annealing(init_c, init_s, rng, time_limit=time_per_init)
        
        if b_score > best_global_score:
            best_global_score = b_score
            best_global_centers = b_c
            best_global_sides = b_s

    # If no solution found (should not happen), fallback
    if best_global_centers is None:
        centers = np.zeros((10, 2))
        sides = np.full(10, 0.1)
        # Simple grid
        for i in range(10):
            row = i // 5
            col = i % 5
            centers[i, 0] = (col + 0.5) * 0.2
            centers[i, 1] = (row + 0.5) * 0.2
        return centers, sides

    return best_global_centers, best_global_sides

# EVOLVE-BLOCK-END
