# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths, subject to non-overlapping interiors.
    
    Strategy:
    1. Constructive Heuristic (Serpentine Packing):
       - Sort squares by size (descending).
       - Place each square in the first available "shelf" in a serpentine pattern.
       - This creates a feasible starting configuration.
    
    2. Local Search (1D Coordinate Ascent):
       - Iteratively move each square in x and y directions to maximize its size
         while maintaining feasibility.
       - This effectively "tightens" the packing.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 600  # 600s limit, keep 10s margin

    def is_feasible(centers, sides):
        """
        Check if the configuration is feasible.
        centers: (n, 2) array
        sides: (n,) array
        """
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
                return False
            if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                    return False
        return True

    def serpentine_pack(sizes):
        """
        Pack squares of given sizes into a 1x1 square using serpentine shelves.
        Returns centers and sides if feasible, else None.
        """
        # Sort indices by size descending
        order = np.argsort(-sizes)
        sorted_sizes = sizes[order]
        
        # We will store positions in a list of (x, y, size)
        placed = []
        shelves = [] # List of [y_bottom, y_top, x_current]
        
        for idx in order:
            s = sizes[idx]
            if s > 1.0:
                return None
            
            placed_in_shelf = False
            # Try existing shelves
            for shelf in shelves:
                y_bottom, y_top, x_current = shelf
                if y_top - y_bottom >= s - tol and 1.0 - x_current >= s - tol:
                    # Place in this shelf
                    cx = x_current + s / 2
                    cy = y_bottom + s / 2
                    placed.append((cx, cy, s, idx))
                    shelf[2] = x_current + s # Update x_current
                    placed_in_shelf = True
                    break
            
            if not placed_in_shelf:
                # Start a new shelf
                # To maximize height usage, we might want to pack shelves tightly.
                # But simple serpentine just stacks them.
                # Let's try to place it in the lowest available y.
                # For simplicity in this heuristic, we just stack them from bottom up?
                # No, serpentine usually fills width, then goes down (or up).
                # Let's assume we fill from bottom (y=0) upwards.
                
                # Find the highest shelf
                if shelves:
                    # Check if we can fit below the lowest shelf? 
                    # Actually, standard serpentine fills row by row.
                    # Let's stick to filling row by row from bottom to top.
                    # If a new row is needed, its y_bottom is the max(y_top) of existing rows?
                    # No, if we fill bottom-up, the first row is at y=0.
                    # The second row is at y = max height of row 1?
                    # This is "Shelf Packing".
                    
                    # To keep it simple and effective for 10 squares:
                    # Just try to fit in any existing shelf.
                    # If not, create a new shelf at the bottom? No, that might overlap.
                    # Let's create a new shelf at the current "stack height".
                    
                    # Actually, a better simple heuristic:
                    # Try to place at (x, y) where y is the lowest possible y such that it fits.
                    # This is more like "Bottom-Left" heuristic.
                    
                    # Let's restart this logic with Bottom-Left.
                    pass

        # The simple shelf packing above is a bit tricky to implement correctly for "best" packing.
        # Let's use a simpler, more robust initialization:
        # 1. Sort by size desc.
        # 2. Try to place each square at the "bottom-left" most position.
        
        placed = [] # (cx, cy, s, idx)
        for idx in order:
            s = sizes[idx]
            if s > 1.0:
                return None
            
            # Search for position
            # We can discretize the search space or use a simple grid.
            # Since n=10, we can afford a bit of search.
            # Try y from 0 to 1, x from 0 to 1.
            # To speed up, we can check candidate positions:
            # 1. x=0, y=0
            # 2. x=1-s, y=0
            # 3. x=0, y=1-s
            # 4. x=1-s, y=1-s
            # 5. Adjacent to existing squares.
            
            found = False
            # Candidate x, y coordinates for the bottom-left corner
            candidates = [(0, 0), (1-s, 0), (0, 1-s), (1-s, 1-s)]
            
            # Add candidates adjacent to existing squares
            for (cx, cy, ss, ii) in placed:
                # Left
                candidates.append((cx - ss/2 - s/2, cy))
                # Right
                candidates.append((cx + ss/2 + s/2, cy))
                # Bottom
                candidates.append((cx, cy - ss/2 - s/2))
                # Top
                candidates.append((cx, cy + ss/2 + s/2))
                # Corners (diagonal)
                candidates.append((cx - ss/2 - s/2, cy - ss/2 - s/2))
                candidates.append((cx + ss/2 + s/2, cy - ss/2 - s/2))
                candidates.append((cx - ss/2 - s/2, cy + ss/2 + s/2))
                candidates.append((cx + ss/2 + s/2, cy + ss/2 + s/2))
            
            # Sort candidates by x+y (bottom-left preference)
            candidates.sort(key=lambda p: (p[0] + p[1], p[0]))
            
            for (bx, by) in candidates:
                # Check bounds
                if bx < -tol or bx + s > 1 + tol:
                    continue
                if by < -tol or by + s > 1 + tol:
                    continue
                
                # Check overlap with placed
                cx_new = bx + s/2
                cy_new = by + s/2
                overlap = False
                for (cx, cy, ss, ii) in placed:
                    t = (s + ss) / 2 - tol
                    if abs(cx_new - cx) < t and abs(cy_new - cy) < t:
                        overlap = True
                        break
                
                if not overlap:
                    placed.append((cx_new, cy_new, s, idx))
                    found = True
                    break
            
            if not found:
                # If we couldn't place it, try a random position?
                # Or just fail this configuration.
                return None

        # Construct arrays
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for (cx, cy, s, idx) in placed:
            centers[idx] = [cx, cy]
            sides[idx] = s
        
        return centers, sides

    def coordinate_ascent(centers, sides, max_iter=5000):
        """
        Perform 1D coordinate ascent to maximize sum of sides.
        """
        current_centers = centers.copy()
        current_sides = sides.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                # Try to increase side i
                # The maximum possible side is limited by:
                # 1. Boundaries: 2 * min(cx, 1-cx, cy, 1-cy)
                # 2. Neighbors: For each j, if they are "close", the max side is limited by the gap.
                
                max_s = min(2 * current_centers[i, 0], 2 * (1 - current_centers[i, 0]),
                            2 * current_centers[i, 1], 2 * (1 - current_centers[i, 1]))
                
                for j in range(n):
                    if i == j:
                        continue
                    # If i and j are aligned in y (overlap in y-range if we expand),
                    # then x-separation limits expansion.
                    # Actually, the constraint is:
                    # |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2
                    
                    # Let's calculate the max s_i such that no overlap with j occurs,
                    # assuming s_j is fixed and centers are fixed.
                    
                    dx = abs(current_centers[i, 0] - current_centers[j, 0])
                    dy = abs(current_centers[i, 1] - current_centers[j, 1])
                    
                    # Case 1: They are separated in X.
                    # Then we can expand i in X until we touch j, OR until boundary.
                    # If we expand i, s_i increases.
                    # Constraint: dx >= (s_i + s_j)/2  => s_i <= 2*dx - s_j
                    # Also boundary: s_i <= 2*min(cx_i, 1-cx_i)
                    # And we must ensure that expanding i doesn't cause overlap in Y if X-separation is lost?
                    # No, if we ensure X-separation, we are safe regardless of Y (as long as Y overlap doesn't force a different constraint? No, OR condition).
                    # If we maintain X-separation, the Y-separation doesn't matter.
                    # So if we choose to separate in X, max s_i is min(2*min(cx_i, 1-cx_i), 2*dx - s_j).
                    
                    # Case 2: They are separated in Y.
                    # Max s_i is min(2*min(cy_i, 1-cy_i), 2*dy - s_j).
                    
                    # We want the MAX s_i such that EITHER X-sep OR Y-sep holds.
                    # So max_s_i = max( min(bound_x, 2*dx - s_j), min(bound_y, 2*dy - s_j) )
                    # BUT, we also have to consider that expanding i might reduce the "separation" in the other dimension?
                    # No, centers are fixed in this step. We are only changing s_i.
                    # So dx and dy are fixed.
                    
                    bound_x = min(2 * current_centers[i, 0], 2 * (1 - current_centers[i, 0]))
                    bound_y = min(2 * current_centers[i, 1], 2 * (1 - current_centers[i, 1]))
                    
                    sep_x_limit = 2 * dx - current_sides[j]
                    sep_y_limit = 2 * dy - current_sides[j]
                    
                    # If sep_x_limit < 0, it means they are already overlapping in X even with s_i=0?
                    # No, if dx < s_j/2, then 2*dx - s_j < 0.
                    # This implies we CANNOT separate in X, even if s_i=0.
                    # So we MUST separate in Y.
                    
                    # So the valid s_i is the max of the two options, but only if the option is "valid" (i.e., >= 0).
                    # Actually, if sep_x_limit < 0, then any s_i >= 0 will overlap in X.
                    # So we are forced to rely on Y-separation.
                    # If sep_y_limit < 0, we are forced to rely on X-separation.
                    # If both < 0, we are stuck (overlap).
                    
                    # Let's compute the max s_i allowed by X-separation:
                    s_x = min(bound_x, sep_x_limit)
                    # Let's compute the max s_i allowed by Y-separation:
                    s_y = min(bound_y, sep_y_limit)
                    
                    # The square i is feasible with neighbor j if s_i <= s_x OR s_i <= s_y.
                    # So the max s_i is max(s_x, s_y).
                    # Wait, if s_x is negative, it means X-separation is impossible.
                    # If s_y is negative, Y-separation is impossible.
                    # If max(s_x, s_y) < 0, then we have a conflict.
                    
                    candidate_max = max(s_x, s_y)
                    if candidate_max < max_s:
                        max_s = candidate_max
                    
                # Ensure max_s is positive
                if max_s <= tol:
                    continue
                    
                # Try to set s_i to max_s
                new_s = max_s
                if new_s > current_sides[i] + tol:
                    # Check if this new configuration is globally feasible
                    # (We checked against each j individually, but is that sufficient?
                    # Yes, because the constraint is pairwise. If it's feasible with each j, it's feasible with all.)
                    # However, changing s_i might affect the "max_s" calculation for other k?
                    # No, we are just checking feasibility of the new state.
                    
                    # Update
                    current_sides[i] = new_s
                    improved = True
                    # Note: We don't move the center in this simple version.
                    # Moving the center could allow larger squares.
                    # Let's add center movement in the next step.
            
            if not improved:
                # Try to move centers to allow larger squares
                # For each i, try to move it to a "better" position
                # A better position is one that allows a larger s_i.
                # We can try moving i to touch a neighbor or boundary in X or Y.
                
                for i in range(n):
                    # Current max s_i
                    max_s = min(2 * current_centers[i, 0], 2 * (1 - current_centers[i, 0]),
                                2 * current_centers[i, 1], 2 * (1 - current_centers[i, 1]))
                    
                    for j in range(n):
                        if i == j: continue
                        dx = abs(current_centers[i, 0] - current_centers[j, 0])
                        dy = abs(current_centers[i, 1] - current_centers[j, 1])
                        bound_x = min(2 * current_centers[i, 0], 2 * (1 - current_centers[i, 0]))
                        bound_y = min(2 * current_centers[i, 1], 2 * (1 - current_centers[i, 1]))
                        sep_x_limit = 2 * dx - current_sides[j]
                        sep_y_limit = 2 * dy - current_sides[j]
                        candidate_max = max(min(bound_x, sep_x_limit), min(bound_y, sep_y_limit))
                        if candidate_max < max_s:
                            max_s = candidate_max
                    
                    # Generate candidate positions
                    # 1. Push to left boundary
                    # 2. Push to right boundary
                    # 3. Push to bottom boundary
                    # 4. Push to top boundary
                    # 5. Push to touch neighbor j in X (left/right)
                    # 6. Push to touch neighbor j in Y (up/down)
                    
                    candidates = []
                    s_i = current_sides[i]
                    
                    # Boundaries
                    candidates.append((s_i/2, current_centers[i, 1])) # Left
                    candidates.append((1 - s_i/2, current_centers[i, 1])) # Right
                    candidates.append((current_centers[i, 0], s_i/2)) # Bottom
                    candidates.append((current_centers[i, 0], 1 - s_i/2)) # Top
                    
                    for j in range(n):
                        if i == j: continue
                        # Touch in X
                        # Left of j
                        candidates.append((current_centers[j, 0] - (current_sides[j] + s_i)/2, current_centers[i, 1]))
                        # Right of j
                        candidates.append((current_centers[j, 0] + (current_sides[j] + s_i)/2, current_centers[i, 1]))
                        # Touch in Y
                        candidates.append((current_centers[i, 0], current_centers[j, 1] - (current_sides[j] + s_i)/2))
                        candidates.append((current_centers[i, 0], current_centers[j, 1] + (current_sides[j] + s_i)/2))
                    
                    best_new_pos = None
                    best_new_s = current_sides[i]
                    
                    for (nx, ny) in candidates:
                        if nx < -tol or nx + s_i > 1 + tol: continue
                        if ny < -tol or ny + s_i > 1 + tol: continue
                        
                        # Check feasibility with this new position
                        feasible = True
                        for j in range(n):
                            if i == j: continue
                            dx = abs(nx - current_centers[j, 0])
                            dy = abs(ny - current_centers[j, 1])
                            t = (s_i + current_sides[j]) / 2 - tol
                            if dx < t and dy < t:
                                feasible = False
                                break
                        
                        if feasible:
                            # Calculate max s_i at this position
                            new_max_s = min(2 * nx, 2 * (1 - nx), 2 * ny, 2 * (1 - ny))
                            for j in range(n):
                                if i == j: continue
                                dx = abs(nx - current_centers[j, 0])
                                dy = abs(ny - current_centers[j, 1])
                                bound_x = min(2 * nx, 2 * (1 - nx))
                                bound_y = min(2 * ny, 2 * (1 - ny))
                                sep_x_limit = 2 * dx - current_sides[j]
                                sep_y_limit = 2 * dy - current_sides[j]
                                candidate_max = max(min(bound_x, sep_x_limit), min(bound_y, sep_y_limit))
                                if candidate_max < new_max_s:
                                    new_max_s = candidate_max
                            
                            if new_max_s > best_new_s + tol:
                                best_new_s = new_max_s
                                best_new_pos = (nx, ny)
                    
                    if best_new_pos is not None:
                        current_centers[i] = best_new_pos
                        current_sides[i] = best_new_s
                        improved = True
            
            if not improved:
                break
                
        return current_centers, current_sides

    # Initial sizes: Start with a reasonable guess
    # We want to maximize sum. 
    # A good start is to have all squares roughly equal size, or a mix.
    # Let's try a few different initial size distributions.
    
    best_centers = None
    best_sides = None
    best_score = -np.inf

    # Strategy 1: Equal sizes
    sizes1 = np.full(n, 0.2)
    
    # Strategy 2: One big, rest small
    sizes2 = np.full(n, 0.15)
    sizes2[0] = 0.4
    
    # Strategy 3: Random
    rng = np.random.default_rng(42)
    sizes3 = rng.uniform(0.1, 0.4, n)
    
    for sizes in [sizes1, sizes2, sizes3]:
        if time.time() > deadline:
            break
        
        res = serpentine_pack(sizes)
        if res is None:
            continue
        
        centers, sides = res
        if not is_feasible(centers, sides):
            continue
        
        # Coordinate Ascent
        centers, sides = coordinate_ascent(centers, sides)
        
        if is_feasible(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers
                best_sides = sides

    # Fallback
    if best_centers is None:
        # Grid fallback
        centers = np.column_stack((np.linspace(0.1, 0.9, 5).repeat(2), 
                                   np.tile(np.linspace(0.1, 0.9, 2), 5)))
        sides = np.full(n, 0.2)
        return centers, sides

    return best_centers, best_sides

# EVOLVE-BLOCK-END
