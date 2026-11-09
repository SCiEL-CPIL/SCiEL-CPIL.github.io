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
       - This creates a valid, dense packing without needing gradient-based constraint satisfaction.
    2. Local Search (1D Coordinate Ascent):
       - Iterate through each square and each dimension (x, y).
       - For a fixed square and dimension, determine the valid interval for its center
         given the positions of all other squares.
       - Move the center to the boundary of this valid interval that maximizes the potential
         side length (i.e., pushes it towards the center of the container or away from neighbors).
       - Expand the side length as much as possible within the new valid interval.
    3. Multiple restarts with different initial orderings and random perturbations.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 610  # Leave 20s margin for safety

    def is_feasible(centers, sides):
        """Check if the configuration is valid."""
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(s):
                return False
            if s <= 0:
                return False
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

    def serpentine_pack(order, rng):
        """
        Attempt to pack squares in the given order using a serpentine shelf algorithm.
        Returns (centers, sides) or None if it fails (though with n=10 it usually succeeds).
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # We will determine sizes on the fly or use a heuristic.
        # A common heuristic for "maximize sum of sides" is to make them as large as possible.
        # However, equal sizes are often optimal for packing squares in a square.
        # Let's start with equal sizes and see if they fit. If not, we might need to adjust.
        # Actually, the objective is sum(sides). 
        # If we fix the topology (which square is where), the optimal size is often constrained by the tightest fit.
        
        # Let's try a simpler approach: 
        # Assume we want to maximize the minimum side length first, then expand.
        # Or, just use a greedy placement where we place square i, and its size is limited by space.
        
        # Better: Start with a base size, place them, then expand in local search.
        # Base size: 1/sqrt(n) is a good start, but for 10, 0.316 is too big for 2x5 grid (0.2).
        # Let's use 0.2 as a safe base for 2x5, or 0.3 for a more complex layout.
        
        # Let's try to fit them in a 2x5 grid initially, but allow the local search to break the grid.
        # Initial size: 0.2
        base_size = 0.2
        
        # Serpentine placement in a 2-row, 5-col grid
        # Row 0: Left to Right
        # Row 1: Right to Left
        
        # We will place them in the order specified by 'order'.
        # 'order' is a permutation of [0..9].
        
        # Grid positions
        # x positions: 0.1, 0.3, 0.5, 0.7, 0.9
        # y positions: 0.1, 0.9
        
        x_slots = [0.1, 0.3, 0.5, 0.7, 0.9]
        y_slots = [0.1, 0.9]
        
        # Create a list of slots (cx, cy)
        slots = []
        for r in range(2):
            for c in range(5):
                cx = x_slots[c]
                cy = y_slots[r]
                # Serpentine: if r is odd, reverse x order? 
                # Actually, let's just fill row 0 L->R, row 1 R->L to reduce "crossing" if we were drawing lines.
                # For placement, it doesn't matter much, but let's vary it.
                if r == 0:
                    slots.append((cx, cy))
                else:
                    slots.append((1.0 - cx, cy)) # Mirror x for serpentine feel
        
        # Assign squares to slots
        for idx, sq_id in enumerate(order):
            if idx < len(slots):
                centers[sq_id] = slots[idx]
                sides[sq_id] = base_size
            else:
                # Should not happen for n=10
                centers[sq_id] = [0.5, 0.5]
                sides[sq_id] = 0.1

        return centers, sides

    def local_search(centers, sides, max_iters=100):
        """
        1D Coordinate Ascent.
        For each square i and dimension d:
        1. Fix other squares.
        2. Determine the range of valid positions for center[i][d] such that square i does not overlap any other square.
           Note: Overlap is avoided if |dx| >= t OR |dy| >= t.
           This is complex to solve exactly for the intersection of all constraints.
           
           Simpler approximation:
           For a fixed i and d, we want to move center[i][d] to a position that maximizes the minimum "slack" 
           in the other dimension or expands the side.
           
           Let's use a simpler heuristic:
           For each pair (i, j), we know the current separation.
           If they are "touching" or close, we can push them apart.
           
           Actually, let's use a standard "push" method:
           For each i:
               For each d in [0, 1]:
                   # Determine the maximum extent we can grow side[i]
                   # and the valid range for center[i][d]
                   
                   # Let's try to maximize side[i] first, keeping center fixed.
                   # Then adjust center to allow further growth.
        """
        
        # We will iterate until no improvement or max_iters
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
                
            for i in range(n):
                for d in range(2): # 0 for x, 1 for y
                    other_d = 1 - d
                    
                    # Current state
                    cx = centers[i, d]
                    cy = centers[i, other_d]
                    s = sides[i]
                    
                    # We want to find a new position for center[i, d] and a new side s_new
                    # such that s_new > s (or position is better for future growth)
                    
                    # Strategy: 
                    # 1. Try to expand side[i] by a small epsilon, check if feasible.
                    # 2. If not, try to move center[i, d] to make room.
                    
                    # Let's try to maximize the "slack" in dimension d.
                    # The constraint for square i and j is:
                    # |cx_i - cx_j| >= (s_i + s_j)/2 - tol OR |cy_i - cy_j| >= (s_i + s_j)/2 - tol
                    
                    # If we fix all other squares, the valid region for (cx_i, cy_i) is the complement of the union of "forbidden" rectangles around each j.
                    # This is hard.
                    
                    # Heuristic: 
                    # Look at the neighbors that are currently "blocking" expansion in dimension d.
                    # A neighbor j blocks expansion in d if |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol.
                    # Actually, if they overlap in the other dimension, they MUST be separated in this dimension.
                    
                    # Let's identify "active" constraints for square i in dimension d.
                    # An active constraint exists if the separation in the OTHER dimension is less than the required separation for the current sides.
                    # i.e., |cy_i - cy_j| < (s_i + s_j)/2 - tol.
                    # In this case, we MUST have |cx_i - cx_j| >= (s_i + s_j)/2 - tol.
                    
                    min_left = 0.0
                    max_right = 1.0
                    
                    for j in range(n):
                        if i == j: continue
                        s_j = sides[j]
                        cx_j = centers[j, d]
                        cy_j = centers[j, other_d]
                        
                        # Check if they overlap in the other dimension
                        sep_other = abs(cy - cy_j)
                        req_sep = (s + s_j) / 2 - tol
                        
                        if sep_other < req_sep:
                            # They must be separated in dimension d
                            # |cx - cx_j| >= req_sep
                            # cx >= cx_j + req_sep  OR  cx <= cx_j - req_sep
                            
                            # This creates a union of allowed intervals.
                            # We are currently at cx. We want to stay in the current component of the allowed set.
                            # If cx > cx_j, we must have cx >= cx_j + req_sep.
                            # If cx < cx_j, we must have cx <= cx_j - req_sep.
                            
                            if cx >= cx_j:
                                min_left = max(min_left, cx_j + req_sep)
                            else:
                                max_right = min(max_right, cx_j - req_sep)
                    
                    # Also boundary constraints
                    min_left = max(min_left, s / 2 - tol)
                    max_right = min(max_right, 1.0 - s / 2 + tol)
                    
                    # If the interval is empty, we are stuck or infeasible.
                    if min_left > max_right:
                        # Try to shrink side to make it feasible? 
                        # Or just skip.
                        continue
                    
                    # Now, within [min_left, max_right], where should we put cx?
                    # We want to maximize the potential for side growth.
                    # The side s is limited by the boundaries and the neighbors in the OTHER dimension?
                    # No, s is limited by the "tightest" constraint in BOTH dimensions.
                    
                    # Let's try to move cx to the center of the available interval [min_left, max_right].
                    # This maximizes the "room" on both sides.
                    new_cx = (min_left + max_right) / 2
                    
                    # Can we grow the side?
                    # The new side s_new is limited by:
                    # 1. Boundaries: s_new <= 2 * min(new_cx, 1 - new_cx) + 2*tol
                    # 2. Neighbors: For each j, if they are "close" in the other dimension, 
                    #    the separation in this dimension must be >= (s_new + s_j)/2.
                    #    So s_new <= 2 * |new_cx - cx_j| - s_j.
                    
                    max_s = 2 * min(new_cx, 1 - new_cx) + 2*tol
                    
                    for j in range(n):
                        if i == j: continue
                        s_j = sides[j]
                        cx_j = centers[j, d]
                        cy_j = centers[j, other_d]
                        
                        # If they overlap in other dimension, they constrain s in this dimension
                        # sep_other = |cy - cy_j| is fixed (for this step)
                        # If sep_other < (s_new + s_j)/2, then we need |new_cx - cx_j| >= (s_new + s_j)/2
                        # This is an inequality involving s_new.
                        # It's circular.
                        
                        # Approximation: Assume the current separation in other dimension determines the constraint.
                        # If they are "touching" or overlapping in other dimension, they constrain us.
                        # Let's just check if moving to new_cx allows a larger side.
                        
                        # Constraint: |new_cx - cx_j| >= (s_new + s_j)/2 - tol
                        # => s_new <= 2 * |new_cx - cx_j| - s_j + 2*tol
                        
                        # But this only applies if they are "active".
                        # If they are far apart in other dimension, they don't constrain s_new in this dimension directly, 
                        # but they might if s_new grows large enough to make them overlap in other dimension.
                        
                        # For simplicity, let's assume all neighbors constrain us if they are in the same "shelf".
                        # Or just use the boundary constraint and a simple check.
                        
                        # Let's just try to set s_new to the boundary limit and see if it's feasible.
                        pass
                    
                    # Let's try a simpler local move:
                    # Move to the center of the available interval.
                    if abs(new_cx - cx) > 1e-8:
                        centers[i, d] = new_cx
                        improved = True
                        
                        # Try to expand side
                        # Recalculate max side with new center
                        # s_new <= 2 * min(new_cx, 1-new_cx) + 2*tol
                        # And for each j, if |cy_i - cy_j| < (s_new+s_j)/2, then |new_cx - cx_j| >= (s_new+s_j)/2
                        # Let's just try to expand s by 10% and check feasibility.
                        s_try = s * 1.1
                        if s_try < 0.1: s_try = 0.1
                        if s_try > 1.0: s_try = 1.0
                        
                        sides[i] = s_try
                        if is_feasible(centers, sides):
                            improved = True
                        else:
                            sides[i] = s # revert
                            # Try smaller expansion
                            s_try = s * 1.01
                            sides[i] = s_try
                            if is_feasible(centers, sides):
                                improved = True
                            else:
                                sides[i] = s # revert again

        return centers, sides

    best_centers = None
    best_sides = None
    best_score = -np.inf

    rng = np.random.default_rng(42)

    # Generate several initial configurations
    configs = []
    
    # 1. Standard Serpentine
    order1 = list(range(n))
    c1, s1 = serpentine_pack(order1, rng)
    configs.append((c1, s1))
    
    # 2. Reversed Serpentine
    order2 = list(range(n-1, -1, -1))
    c2, s2 = serpentine_pack(order2, rng)
    configs.append((c2, s2))
    
    # 3. Random Permutation
    for _ in range(5):
        order_r = rng.permutation(n)
        cr, sr = serpentine_pack(order_r, rng)
        configs.append((cr, sr))
        
    # 4. "Corner" start: One big square in corner
    c4 = np.zeros((n, 2))
    s4 = np.zeros(n)
    s4[0] = 0.5
    c4[0] = [0.25, 0.25]
    for i in range(1, n):
        s4[i] = 0.15
        c4[i] = rng.uniform(0.15, 0.85, 2)
    configs.append((c4, s4))
    
    for c0, s0 in configs:
        if time.time() > deadline:
            break
            
        # Ensure initial feasibility (might need repair)
        if not is_feasible(c0, s0):
            # Simple repair: shrink sides
            s0 = s0 * 0.5
            if not is_feasible(c0, s0):
                s0 = s0 * 0.5
            if not is_feasible(c0, s0):
                continue # Skip this config if it's too broken
        
        # Run local search
        c_loc, s_loc = local_search(c0.copy(), s0.copy(), max_iters=50)
        
        if is_feasible(c_loc, s_loc):
            score = np.sum(s_loc)
            if score > best_score:
                best_score = score
                best_centers = c_loc
                best_sides = s_loc

    # If we found a solution, do a final polish with more iterations on the best one
    if best_centers is not None:
        if time.time() < deadline - 10:
            c_pol, s_pol = local_search(best_centers.copy(), best_sides.copy(), max_iters=100)
            if is_feasible(c_pol, s_pol):
                score_pol = np.sum(s_pol)
                if score_pol > best_score:
                    best_centers = c_pol
                    best_sides = s_pol
                    best_score = score_pol

    # Fallback
    if best_centers is None:
        # Return a valid grid
        centers = np.column_stack((np.linspace(0.1, 0.9, 5).repeat(2), 
                                   np.tile(np.linspace(0.1, 0.9, 2), 5)))
        sides = np.full(n, 0.2)
        return centers, sides

    return best_centers, best_sides

# EVOLVE-BLOCK-END
