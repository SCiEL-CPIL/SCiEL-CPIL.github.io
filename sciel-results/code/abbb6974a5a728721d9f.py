"""
Strategy: Construct 20 points by selecting 10 disjoint 3-point lines from a small affine grid.
This maximizes the number of collinear triples (forcing any large subset to likely contain a triple)
while ensuring no 4 points are collinear by construction (since lines are disjoint and grid is small).
We use a backtracking search to find a valid set of 10 disjoint lines that minimizes the max independent set.
"""
import time
import random
import sys
from itertools import combinations

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    # Use tuples for hashing to avoid TypeError with lists
    pts_tuple = [tuple(p) for p in points]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = pts_tuple[i]
            x2, y2 = pts_tuple[j]
            dx = x2 - x1
            dy = y2 - y1
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = pts_tuple[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _max_independent_set(points):
    """
    Exact branch and bound for the largest subset with no three collinear.
    """
    n = len(points)
    
    # Precompute collinear triples using bitmasks for speed
    # conflict_mask[i] is a bitmask of points that form a collinear triple with i and some other point?
    # No, better: Precompute all collinear triples (i, j, k) with i < j < k.
    # Or, for backtracking:
    # State: current_set (list of indices), candidates (list of indices)
    # If we add point p, we must remove from candidates any point q such that there exists r in current_set 
    # such that p, q, r are collinear? 
    # Actually, if p, q, r are collinear, we cannot have all three.
    # If we have p and r in current_set, we cannot add q.
    # So, when adding p, we remove from candidates any q that is collinear with p and ANY point in current_set.
    
    # Precompute collinear_with[i][j] = bitmask of k such that i, j, k are collinear.
    # Since n=20, we can use integers as bitmasks.
    
    collinear_with = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    mask |= (1 << k)
            collinear_with[i][j] = mask
            collinear_with[j][i] = mask

    best = 0
    
    def backtrack(current_set, candidates_mask):
        nonlocal best
        
        # Pruning
        # Upper bound: len(current_set) + popcount(candidates_mask)
        if len(current_set) + bin(candidates_mask).count('1') <= best:
            return
            
        if candidates_mask == 0:
            if len(current_set) > best:
                best = len(current_set)
            return

        # Pick a candidate
        # Heuristic: Pick the one with the most conflicts?
        # Let's just pick the lowest set bit for simplicity, or one with high degree.
        # For n=20, simple is fine if pruning is good.
        
        # Find an index to branch on
        # Let's pick the first available candidate
        p_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        rest_mask = candidates_mask & ~(1 << p_idx)
        
        # Branch 1: Include p_idx
        # New candidates: remove p_idx and any point that forms a collinear triple with p_idx and any point in current_set
        forbidden = 0
        for c in current_set:
            forbidden |= collinear_with[p_idx][c]
        
        new_candidates_mask = rest_mask & ~forbidden & ~(1 << p_idx)
        
        backtrack(current_set + [p_idx], new_candidates_mask)
        
        # Branch 2: Exclude p_idx
        backtrack(current_set, rest_mask)

    backtrack([], (1 << n) - 1)
    return best

def _generate_candidate_lines():
    """
    Generate all possible 3-point lines in a small grid (e.g., 10x10) that fit in [-100, 100].
    We restrict to a small grid to keep the search space manageable and ensure points are "close" 
    which tends to create more interactions.
    """
    # Use a 10x10 grid centered at 0
    # Points: (x, y) where x, y in [-5, 5]
    # Scale by 10 to use more of the range? No, small integers are fine for collinearity.
    # Actually, scaling doesn't change collinearity.
    # Let's use points in [-10, 10] to have 21x21 = 441 points.
    # But we only need 20 points.
    # Generating all 3-point lines in a 21x21 grid is expensive.
    # Let's restrict to a 7x7 grid [-3, 3]. 49 points.
    # Number of lines is manageable.
    
    points_in_grid = []
    for x in range(-3, 4):
        for y in range(-3, 4):
            points_in_grid.append((x, y))
            
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points_in_grid)}
    n_grid = len(points_in_grid)
    
    lines = []
    # Iterate over all pairs to define a line, then find the third point
    # To avoid duplicates, enforce i < j < k
    for i in range(n_grid):
        for j in range(i+1, n_grid):
            x1, y1 = points_in_grid[i]
            x2, y2 = points_in_grid[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Find k > j such that points[i], points[j], points[k] are collinear
            # And k is the "next" point on the line? 
            # Simpler: Just check all k > j.
            for k in range(j+1, n_grid):
                x3, y3 = points_in_grid[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    # Check if this is a "primitive" line segment? 
                    # We just need sets of 3 collinear points.
                    # To ensure we don't pick {1,2,3} and {1,2,4} as separate lines that overlap,
                    # we will handle disjointness in the selection phase.
                    lines.append((i, j, k))
                    
    return points_in_grid, lines

def _find_best_configuration():
    """
    Backtracking search to select 10 disjoint 3-point lines from the grid.
    """
    points_in_grid, lines = _generate_candidate_lines()
    n_lines = len(lines)
    
    # Precompute bitmask for each line
    line_masks = []
    for l in lines:
        mask = 0
        for idx in l:
            mask |= (1 << idx)
        line_masks.append(mask)
        
    # Sort lines by some heuristic? Maybe length or position.
    # Random shuffle to break symmetry
    random.shuffle(line_masks)
    
    best_score = float('inf')
    best_points = None
    deadline = time.time() + 130.0
    
    # Backtracking
    selected_lines = []
    used_points_mask = 0
    
    def backtrack():
        nonlocal best_score, best_points
        
        if time.time() > deadline:
            return
            
        if len(selected_lines) == 10:
            # We have 10 disjoint lines, so 30 points? 
            # Wait, the task requires exactly 20 points.
            # 10 lines * 3 points = 30 points. This is too many.
            # We need 20 points.
            # So we can select at most 6 disjoint 3-point lines (18 points) + 2 single points?
            # Or 5 lines (15 points) + 5 single points?
            # The "no 4 collinear" constraint is satisfied if lines are disjoint.
            # The "minimize max independent set" is helped by having many triples.
            # Let's target: 6 disjoint 3-point lines (18 points) and 2 extra points.
            # Total 20 points.
            
            # Extract points
            pts = []
            for mask in selected_lines:
                # Get indices from mask
                m = mask
                while m:
                    idx = (m & -m).bit_length() - 1
                    pts.append(points_in_grid[idx])
                    m &= m - 1
                    
            # Add 2 random extra points from the grid not used
            available = [i for i in range(len(points_in_grid)) if not (used_points_mask & (1 << i))]
            if len(available) < 2:
                return
            random.shuffle(available)
            pts.append(points_in_grid[available[0]])
            pts.append(points_in_grid[available[1]])
            
            # Check validity (should be valid by construction, but double check)
            if not _check_no_four_collinear(pts):
                return
                
            score = _max_independent_set(pts)
            if score < best_score:
                best_score = score
                best_points = pts[:]
            return

        # Pruning: if not enough lines left to reach 10, return
        # (We need 6 lines for 18 points, so let's change target to 6 lines)
        
        # Let's restart the logic: Target 6 disjoint 3-point lines.
        
        if len(selected_lines) == 6:
             # Add 2 extra points
             pts = []
             for mask in selected_lines:
                 m = mask
                 while m:
                     idx = (m & -m).bit_length() - 1
                     pts.append(points_in_grid[idx])
                     m &= m - 1
                     
             available = [i for i in range(len(points_in_grid)) if not (used_points_mask & (1 << i))]
             if len(available) < 2:
                 return
             random.shuffle(available)
             pts.append(points_in_grid[available[0]])
             pts.append(points_in_grid[available[1]])
             
             if not _check_no_four_collinear(pts):
                 return
                 
             score = _max_independent_set(pts)
             if score < best_score:
                 best_score = score
                 best_points = pts[:]
             return

        for i, mask in enumerate(line_masks):
            # Check if line is disjoint from used points
            if mask & used_points_mask:
                continue
                
            # Also check if adding this line creates 4 collinear?
            # Since lines are from the same grid and disjoint, 
            # could two disjoint lines share a collinear 4th point?
            # No, if line A has 3 points and line B has 3 points, and they are disjoint,
            # can there be a 4th point on line A? 
            # The line is defined by the 3 points. If there was a 4th point on that line in the grid,
            # our line generation might have included it or not.
            # In a 7x7 grid, max collinear points is 7.
            # If we pick a subset of 3, there might be other points on the same infinite line.
            # We must ensure NO 4 points are collinear in the FINAL SET.
            # So, when adding a line, we must ensure no OTHER selected point is collinear with any 2 points of this line?
            # Actually, just check the final set.
            
            selected_lines.append(mask)
            backtrack()
            selected_lines.pop()

    backtrack()
    
    if best_points is None:
        # Fallback: Random valid configuration
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in pts:
                # Check 4-collinear
                valid = True
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        count = 2
                        for k in range(len(pts)):
                            if k == i or k == j: continue
                            x3, y3 = pts[k]
                            if dx * (y3 - y1) == dy * (x3 - x1):
                                count += 1
                        if dx * (y - y1) == dy * (x - x1):
                            count += 1
                        if count >= 4:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    pts.append(p)
        return pts
        
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Run the structured search
    pts = _find_best_configuration()
    
    # Optional: Local search refinement on the structured result
    # Move one point slightly to see if score improves
    if time.time() < deadline - 20:
        current_score = _max_independent_set(pts)
        best_score = current_score
        best_pts = pts[:]
        
        while time.time() < deadline - 5:
            # Pick a random point
            idx = random.randint(0, 19)
            old_p = pts[idx]
            
            # Try small perturbations
            improved = False
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0: continue
                    new_p = (old_p[0] + dx, old_p[1] + dy)
                    if new_p in pts: continue
                    if abs(new_p[0]) > 100 or abs(new_p[1]) > 100: continue
                    
                    pts[idx] = new_p
                    if _check_no_four_collinear(pts):
                        new_score = _max_independent_set(pts)
                        if new_score < best_score:
                            best_score = new_score
                            best_pts = pts[:]
                            improved = True
                    else:
                        pts[idx] = old_p
                        continue
                    pts[idx] = old_p # Revert for next try
            
            if not improved and random.random() < 0.1:
                # Restart local search? Or just break
                break
                
        if best_score < current_score:
            return [list(p) for p in best_pts]
            
    return [list(p) for p in pts]

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
    print(pts)

