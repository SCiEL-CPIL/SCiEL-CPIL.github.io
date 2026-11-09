import numpy as np
import time

def squares_in_square10() -> tuple:
    """
    Maximizes the total side length of 10 non-overlapping axis-aligned squares
    inside the unit square [0, 1]^2 using Deterministic Hierarchical Grid Refinement.
    
    Strategy:
    1. Define a set of diverse "topological" layouts (e.g., 3x3 grid with one split, 4x3, etc.).
    2. For each layout, construct an initial valid configuration where squares are
       aligned in rows/columns with small gaps.
    3. Refine each configuration using a fast, vectorized Coordinate Descent local search.
       - Move: Perturb one coordinate (cx, cy, or side) of one square.
       - Accept: Only if the new state is valid (hard constraints) and improves the objective.
       - This is a strict hill-climb on the valid manifold, but starting from good
         structural seeds avoids the trapping issues of pure random-walk SA.
    4. Return the best valid configuration found across all seeds.
    """
    n = 10
    start_time = time.time()
    deadline = 600.0
    hard_deadline = start_time + deadline - 10.0  # Safety margin

    def is_valid(centers, sides, tol=1e-7):
        """
        Vectorized validity check.
        """
        # Boundary check
        if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
            return False
        
        # Overlap check
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = np.abs(centers[i_idx, 0] - centers[j_idx, 0])
        dy = np.abs(centers[i_idx, 1] - centers[j_idx, 1])
        t = (sides[i_idx] + sides[j_idx]) / 2.0 - tol
        
        # Violation if dx < t AND dy < t
        violations = (dx < t) & (dy < t)
        
        return not np.any(violations)

    def get_score(centers, sides):
        return np.sum(sides)

    def generate_grid_layout(rows, cols, rng):
        """
        Generate a valid configuration based on a grid topology.
        Distributes 10 squares into a rows x cols grid.
        Some cells may contain multiple squares if the grid is larger than 10, 
        or we just fill the first 10 cells.
        For simplicity, we define specific layouts.
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # We will define specific "blueprints" for 10 squares.
        # Blueprint 1: 3 rows, 4 cols (12 cells, use 10)
        # Blueprint 2: 4 rows, 3 cols (12 cells, use 10)
        # Blueprint 3: 2 rows, 5 cols (10 cells)
        # Blueprint 4: 5 rows, 2 cols (10 cells)
        
        if rows == 3 and cols == 4:
            # 3x4 grid. Each cell is 1/3 x 1/4.
            # We place squares in the first 10 cells.
            # To make it valid, we start with small squares and let refinement grow them.
            # Initial size: min(cell_w, cell_h) * 0.8
            cell_w = 1.0 / cols
            cell_h = 1.0 / rows
            init_side = min(cell_w, cell_h) * 0.8
            
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= n:
                        break
                    cx = (c + 0.5) * cell_w
                    cy = (r + 0.5) * cell_h
                    centers[idx, 0] = cx
                    centers[idx, 1] = cy
                    sides[idx] = init_side
                    idx += 1
                    
            # If we didn't fill 10, the rest are zero? No, we need 10.
            # 3*4 = 12 cells. So we fill 10.
            # The last 2 cells are empty.
            # We should probably place the remaining squares in the empty cells or merge?
            # Actually, for a valid packing, we just need 10 distinct squares.
            # If we leave cells empty, we have wasted space.
            # Better: Use a layout that exactly fits or allows splitting.
            
        elif rows == 4 and cols == 3:
            cell_w = 1.0 / cols
            cell_h = 1.0 / rows
            init_side = min(cell_w, cell_h) * 0.8
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= n:
                        break
                    cx = (c + 0.5) * cell_w
                    cy = (r + 0.5) * cell_h
                    centers[idx, 0] = cx
                    centers[idx, 1] = cy
                    sides[idx] = init_side
                    idx += 1
        elif rows == 2 and cols == 5:
            cell_w = 1.0 / cols
            cell_h = 1.0 / rows
            init_side = min(cell_w, cell_h) * 0.8
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= n:
                        break
                    cx = (c + 0.5) * cell_w
                    cy = (r + 0.5) * cell_h
                    centers[idx, 0] = cx
                    centers[idx, 1] = cy
                    sides[idx] = init_side
                    idx += 1
        elif rows == 5 and cols == 2:
            cell_w = 1.0 / cols
            cell_h = 1.0 / rows
            init_side = min(cell_w, cell_h) * 0.8
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= n:
                        break
                    cx = (c + 0.5) * cell_w
                    cy = (r + 0.5) * cell_h
                    centers[idx, 0] = cx
                    centers[idx, 1] = cy
                    sides[idx] = init_side
                    idx += 1
        else:
            # Fallback: Random valid
            return random_valid_init(rng)
            
        # Add small jitter to break symmetry
        centers += rng.uniform(-0.001, 0.001, (n, 2))
        
        # Ensure validity
        if not is_valid(centers, sides):
            sides *= 0.9
            if not is_valid(centers, sides):
                return random_valid_init(rng)
                
        return centers, sides

    def random_valid_init(rng):
        """
        Generate a random valid initial configuration.
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        for i in range(n):
            placed = False
            for _ in range(100):
                s = rng.uniform(0.1, 0.3)
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
                    placed = True
                    break
            
            if not placed:
                s = 0.01
                cx = rng.uniform(s/2, 1 - s/2)
                cy = rng.uniform(s/2, 1 - s/2)
                centers[i] = [cx, cy]
                sides[i] = s
                
        return centers, sides

    def coordinate_descent(centers, sides, rng, time_limit=10.0):
        """
        Strict Coordinate Descent Local Search.
        Only accepts moves that improve the score and maintain validity.
        Uses adaptive step sizes.
        """
        current_centers = centers.copy()
        current_sides = sides.copy()
        current_score = get_score(current_centers, current_sides)
        
        best_centers = current_centers.copy()
        best_sides = current_sides.copy()
        best_score = current_score
        
        start = time.time()
        step_size = 0.05
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - start < time_limit:
            # Select a square and a dimension to perturb
            i = rng.integers(0, n)
            dim = rng.integers(0, 3) # 0: cx, 1: cy, 2: side
            
            # Adaptive step: shrink if stuck, grow if improving
            if no_improve_count > max_no_improve:
                step_size *= 0.5
                no_improve_count = 0
                if step_size < 1e-5:
                    break
            
            # Generate a move
            # We try a few random directions for this dimension to find an improvement
            improved = False
            for _ in range(10):
                delta = rng.uniform(-step_size, step_size)
                
                new_centers = current_centers.copy()
                new_sides = current_sides.copy()
                
                if dim == 0:
                    new_centers[i, 0] += delta
                elif dim == 1:
                    new_centers[i, 1] += delta
                else:
                    new_sides[i] += delta
                    if new_sides[i] <= 0.01:
                        new_sides[i] = 0.01
                    elif new_sides[i] > 1.0:
                        new_sides[i] = 1.0

                if is_valid(new_centers, new_sides):
                    new_score = get_score(new_centers, new_sides)
                    if new_score > current_score:
                        # Accept
                        current_centers = new_centers
                        current_sides = new_sides
                        current_score = new_score
                        improved = True
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_centers = current_centers.copy()
                            best_sides = current_sides.copy()
                        
                        # Grow step size slightly if we improved
                        step_size = min(step_size * 1.1, 0.2)
                        break
            
            if not improved:
                no_improve_count += 1
                
        return best_centers, best_sides, best_score

    # Main Loop
    rng = np.random.default_rng(42)
    
    best_global_centers = None
    best_global_sides = None
    best_global_score = -np.inf
    
    # List of initialization strategies
    inits = []
    
    # Grid layouts
    inits.append(generate_grid_layout(3, 4, rng))
    inits.append(generate_grid_layout(4, 3, rng))
    inits.append(generate_grid_layout(2, 5, rng))
    inits.append(generate_grid_layout(5, 2, rng))
    
    # Random inits
    inits.append(random_valid_init(rng))
    inits.append(random_valid_init(rng))
    
    # Run refinement for each init
    time_per_init = 15.0 # 6 inits * 15s = 90s. Well within budget.
    
    for init_c, init_s in inits:
        if time.time() > hard_deadline:
            break
            
        b_c, b_s, b_score = coordinate_descent(init_c, init_s, rng, time_limit=time_per_init)
        
        if b_score > best_global_score:
            best_global_score = b_score
            best_global_centers = b_c
            best_global_sides = b_s

    # Fallback
    if best_global_centers is None:
        centers = np.zeros((10, 2))
        sides = np.full(10, 0.1)
        for i in range(10):
            row = i // 5
            col = i % 5
            centers[i, 0] = (col + 0.5) * 0.2
            centers[i, 1] = (row + 0.5) * 0.2
        return centers, sides

    return best_global_centers, best_global_sides

# EVOLVE-BLOCK-END
