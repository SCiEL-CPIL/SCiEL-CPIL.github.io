import numpy as np
import time

def squares_in_square10() -> tuple:
    """
    Maximizes the total side length of 10 non-overlapping axis-aligned squares
    inside the unit square [0, 1]^2 using Hierarchical Block Packing and 
    Coordinate Descent Refinement.
    
    Strategy:
    1. Define a hierarchical layout (tree of rectangular blocks) that partitions 
       the unit square into 10 disjoint rectangular regions (bins).
    2. Each bin is assigned to hold exactly one square.
    3. The square in a bin is constrained to be inside that bin.
    4. We optimize the dimensions of the bins (splits in the tree) and the 
       size/position of the squares within their bins.
    5. Since bins are disjoint, squares in different bins never overlap.
    6. Squares in the same bin? No, each bin holds exactly 1 square.
    7. This reduces the problem to finding the optimal partition of the unit square 
       into 10 rectangles and fitting the largest possible square in each.
    8. We use coordinate descent to refine the split parameters of the tree and 
       the square sizes.
    """
    n = 10
    start_time = time.time()
    deadline = 600.0
    hard_deadline = start_time + deadline - 5.0  # 5s safety margin

    def is_valid(centers, sides, tol=1e-7):
        """
        Check if configuration is valid.
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
        violations = (dx < t) & (dy < t)
        
        return not np.any(violations)

    def get_score(centers, sides):
        return np.sum(sides)

    def refine_configuration(centers, sides, time_limit):
        """
        Coordinate descent refinement.
        We try to increase the size of each square by moving it or resizing it,
        while maintaining validity. Since we start from a valid block-based config,
        we can try to "grow" squares into the empty space of their bin or neighboring bins
        if we relax the bin constraints? 
        Actually, to keep it simple and robust: 
        We treat the initial block packing as a starting point. 
        Then we perform local search on the (centers, sides) directly, 
        but using a smarter move generation: 
        1. Try to increase side of square i by delta.
        2. If invalid, try to move square i to resolve conflict.
        3. If still invalid, try to move neighbors.
        
        However, the "Block Packing" insight is that the *topology* of the packing 
        (which square is to the left of which) is fixed by the tree. 
        If we stick to this topology, we can solve for the optimal sizes analytically 
        or with fast 1D optimization for each split.
        
        Let's implement a simpler, robust version:
        1. Generate a few different tree topologies (guillotine cuts).
        2. For each topology, optimize the split parameters to maximize the sum of 
           inscribed square sizes.
        3. Then, take the best result and do a final "free" local search (SA-like but 
           very targeted) to break out of the topological constraints if beneficial.
        """
        best_c = centers.copy()
        best_s = sides.copy()
        best_score = get_score(best_c, best_s)
        
        # Local search: Try to improve by small perturbations
        # We will use a simple hill climbing with random restarts for small perturbations
        # but focused on "growing" squares.
        
        T = 0.01
        cooling = 0.99
        rng = np.random.default_rng(123)
        
        current_c = centers.copy()
        current_s = sides.copy()
        current_score = get_score(current_c, current_s)
        
        start = time.time()
        
        while time.time() - start < time_limit and T > 0.001:
            i = rng.integers(0, n)
            
            # Move 1: Try to increase size
            if rng.random() < 0.7:
                ds = rng.uniform(0, 0.05)
                new_s = current_s.copy()
                new_s[i] += ds
                new_c = current_c.copy()
                
                # Adjust center to stay in bounds if possible, or just check validity
                # If size increases, it might go out of bounds.
                # Clamp center to keep square in [0,1]
                new_c[i, 0] = np.clip(new_c[i, 0], new_s[i]/2, 1 - new_s[i]/2)
                new_c[i, 1] = np.clip(new_c[i, 1], new_s[i]/2, 1 - new_s[i]/2)
                
                if is_valid(new_c, new_s):
                    new_score = get_score(new_c, new_s)
                    if new_score > current_score:
                        current_c, current_s, current_score = new_c, new_s, new_score
                        if current_score > best_score:
                            best_c, best_s, best_score = current_c.copy(), current_s.copy(), current_score
                    else:
                        # Accept worse with prob exp(-delta/T) to escape
                        delta = new_score - current_score
                        if rng.random() < np.exp(delta / T):
                            current_c, current_s, current_score = new_c, new_s, new_score
            else:
                # Move 2: Perturb position
                dx = rng.normal(0, 0.02)
                dy = rng.normal(0, 0.02)
                new_c = current_c.copy()
                new_c[i, 0] += dx
                new_c[i, 1] += dy
                new_s = current_s.copy()
                
                # Clamp to bounds
                new_c[i, 0] = np.clip(new_c[i, 0], new_s[i]/2, 1 - new_s[i]/2)
                new_c[i, 1] = np.clip(new_c[i, 1], new_s[i]/2, 1 - new_s[i]/2)
                
                if is_valid(new_c, new_s):
                    new_score = get_score(new_c, new_s)
                    delta = new_score - current_score
                    if delta > 0 or rng.random() < np.exp(delta / T):
                        current_c, current_s, current_score = new_c, new_s, new_score
                        if current_score > best_score:
                            best_c, best_s, best_score = current_c.copy(), current_s.copy(), current_score
            
            T *= cooling
            
        return best_c, best_s, best_score

    def generate_block_packing(rng, n_squares):
        """
        Generate a valid configuration using a guillotine cut tree.
        We recursively split the unit square into rectangles until we have n_squares rectangles.
        Each rectangle gets a square inscribed in it.
        """
        # State: list of (x, y, w, h) rectangles
        rects = [(0.0, 0.0, 1.0, 1.0)]
        
        while len(rects) < n_squares:
            # Pick a rectangle to split
            # Weighted by area to ensure we don't create too many tiny slivers
            areas = [r[2] * r[3] for r in rects]
            total_area = sum(areas)
            weights = [a / total_area for a in areas]
            
            idx = rng.choice(len(rects), p=weights)
            x, y, w, h = rects[idx]
            
            # Decide split direction: horizontal or vertical
            # Prefer splitting the longer dimension
            if w > h:
                # Vertical split
                split_pos = rng.uniform(0.3, 0.7) * w
                r1 = (x, y, split_pos, h)
                r2 = (x + split_pos, y, w - split_pos, h)
            else:
                # Horizontal split
                split_pos = rng.uniform(0.3, 0.7) * h
                r1 = (x, y, w, split_pos)
                r2 = (x, y + split_pos, w, h - split_pos)
            
            # Remove old, add new
            rects.pop(idx)
            rects.append(r1)
            rects.append(r2)
            
            # If we have more than n_squares, we stop. 
            # But we need exactly n_squares. 
            # The loop condition handles this: we stop when len(rects) == n_squares.
            # Wait, if we start with 1 and split, we get 2. 
            # We need 10. So we split 9 times.
            # The loop runs while len < 10.
            # When len is 9, we split one to get 10.
            # So the logic is correct.

        centers = np.zeros((n_squares, 2))
        sides = np.zeros(n_squares)
        
        for i, (x, y, w, h) in enumerate(rects):
            # Inscribed square in rectangle (x,y,w,h)
            s = min(w, h)
            # Center of the square
            # If w > h, square is h x h, centered vertically, but horizontally?
            # To maximize future flexibility, center it.
            cx = x + w / 2
            cy = y + h / 2
            centers[i] = [cx, cy]
            sides[i] = s
            
        return centers, sides

    # Main Loop
    rng = np.random.default_rng(42)
    
    best_global_centers = None
    best_global_sides = None
    best_global_score = -np.inf
    
    # Try multiple different block packings
    # The topology of the guillotine cuts matters.
    # We will try 50 different random guillotine packings.
    
    num_starts = 50
    time_per_start = (hard_deadline - time.time()) / num_starts
    
    for i in range(num_starts):
        if time.time() > hard_deadline:
            break
            
        # Generate a new block packing
        c, s = generate_block_packing(rng, n)
        
        # Check validity (should be valid by construction, but floating point might bite)
        if not is_valid(c, s):
            # Shrink slightly
            s *= 0.99
            if not is_valid(c, s):
                continue
        
        # Refine
        b_c, b_s, b_score = refine_configuration(c, s, time_limit=time_per_start)
        
        if b_score > best_global_score:
            best_global_score = b_score
            best_global_centers = b_c
            best_global_sides = b_s

    # If no solution found, fallback
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
