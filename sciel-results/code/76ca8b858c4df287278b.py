import time
import random
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # Generate all points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    
    def greedy_select(points_order):
        """
        Greedy selection with incremental validity checking.
        Returns the selected list and the dist_from structure.
        """
        selected = []
        # dist_from[i] is a set of squared distances from selected[i] to all other selected points
        dist_from = []
        
        for p in points_order:
            px, py = p
            valid = True
            d_p = []
            
            # Calculate distances from P to all selected points
            for i in range(len(selected)):
                ax, ay = selected[i]
                d_sq = (px - ax)**2 + (py - ay)**2
                d_p.append(d_sq)
            
            # Check 1: P is apex. All d_p must be unique.
            if len(set(d_p)) != len(d_p):
                valid = False
            else:
                # Check 2: For each A, d(P,A) must not be equal to d(A,B) for any B
                # If d(P,A) == d(A,B), then triangle PAB is isosceles with apex A.
                for i in range(len(selected)):
                    d_pa = d_p[i]
                    if d_pa in dist_from[i]:
                        valid = False
                        break
                        
            if valid:
                selected.append(p)
                # Update dist_from
                new_dist_set = set(d_p)
                dist_from.append(new_dist_set)
                
                # Update existing dist_from sets with new distance to P
                for i in range(len(selected) - 1):
                    dist_from[i].add(d_p[i])
                    
        return selected, dist_from

    def is_valid_set(points):
        """Check if a set of points has no three forming an isosceles triangle."""
        n = len(points)
        if n < 3:
            return True
        # Precompute all pairwise squared distances
        dist_sq = {}
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist_sq[(i, j)] = dx*dx + dy*dy
        
        # Check each triplet
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    d_ij = dist_sq[(i, j)]
                    d_ik = dist_sq[(i, k)]
                    d_jk = dist_sq[(j, k)]
                    # Isosceles if any two distances are equal
                    if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                        return False
        return True

    def local_improve(points_set, max_iters=20):
        """
        Try to swap points to improve the set size.
        Uses a randomized local search approach:
        1. Pick a point to remove from the set.
        2. Try to add a random point not in the set.
        3. If valid, keep the swap.
        4. Repeat until no improvement or time limit.
        """
        if not points_set:
            return points_set
        
        selected = list(points_set)
        selected_set = set(selected)
        non_selected = [p for p in all_points if p not in selected_set]
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try swapping each selected point with a random non-selected point
            # We shuffle non_selected to get variety
            random.shuffle(non_selected)
            
            # Try to find a beneficial swap
            # We iterate through selected points and try to swap with candidates
            # To be efficient, we don't check all O(K * M) swaps, but rather random ones
            # Or we can do a full pass if K is small.
            
            # Let's try a few random swaps per iteration
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                if not non_selected:
                    break
                
                # Pick a random index in selected to remove
                idx_remove = random.randint(0, len(selected) - 1)
                p_remove = selected[idx_remove]
                
                # Pick a random index in non_selected to add
                idx_add = random.randint(0, len(non_selected) - 1)
                p_add = non_selected[idx_add]
                
                # Construct new set
                # Remove p_remove
                new_selected = selected[:idx_remove] + selected[idx_remove+1:]
                # Add p_add
                new_selected.append(p_add)
                
                # Check validity
                if is_valid_set(new_selected):
                    # Swap succeeded
                    selected = new_selected
                    selected_set = set(selected)
                    non_selected = [p for p in all_points if p not in selected_set]
                    improved = True
                    break
        
        return selected

    # Strategy: Random restarts with greedy + local improvement
    while time.time() < deadline - 10:
        random.shuffle(all_points)
        
        # Use the fast incremental greedy
        current_set, _ = greedy_select(all_points)
        
        # Apply local improvement if we have time
        if time.time() < deadline - 15:
            current_set = local_improve(current_set, max_iters=10)
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
