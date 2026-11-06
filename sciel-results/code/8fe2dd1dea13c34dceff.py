import time
import math
from itertools import combinations

def construct():
    """
    Constructs 32 points on the triangular lattice maximizing the number of edges.
    Uses hexagonal spiral with optimization of outer ring selection.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    n_points = 32
    
    # Generate all points in hexagonal spiral order
    # Ring 0: (0,0)
    # Ring r: hexagonal perimeter
    points_all = [(0, 0)]
    
    directions = [
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (1, 0),
        (0, 1)
    ]
    
    r = 1
    while len(points_all) < 50:  # Generate extra points for optimization
        a, b = r, 0
        for side in range(6):
            da, db = directions[side]
            for step in range(r):
                points_all.append((a, b))
                a += da
                b += db
        r += 1
    
    # For n=32, we have:
    # Ring 0: 1 point
    # Ring 1: 6 points (total: 7)
    # Ring 2: 12 points (total: 19)
    # Ring 3: 18 points (total: 37)
    # So we need 32 - 19 = 13 points from ring 3, or 32 - 7 = 25 from rings 2+3
    
    # Actually, let's be more precise:
    # Ring 0: 1
    # Ring 1: 6 (cumulative: 7)
    # Ring 2: 12 (cumulative: 19)
    # Ring 3: 18 (cumulative: 37)
    # We need 32 points, so 32 - 19 = 13 points from ring 3
    
    # But wait, let me reconsider. The hexagonal spiral order means we take points in order.
    # Points 0-0: ring 0 (1 point)
    # Points 1-6: ring 1 (6 points)
    # Points 7-18: ring 2 (12 points)
    # Points 19-36: ring 3 (18 points)
    # We need 32 points, so points 0-31, which is all of rings 0,1,2 and 13 of ring 3
    
    # The key insight: we should choose WHICH 13 points from ring 3 to include
    # to maximize edges. Points that connect to more included points are better.
    
    # Let's use a greedy approach: start with all inner points, then add outer points
    # in order of how many edges they create with already-selected points.
    
    # First, let's get the inner points (rings 0, 1, 2)
    inner_points = points_all[:19]  # 19 points
    outer_candidates = points_all[19:37]  # 18 points from ring 3
    
    # We need 32 - 19 = 13 points from outer_candidates
    
    def count_edges(point_set):
        """Count number of unit-distance pairs in point_set."""
        ps = list(point_set)
        count = 0
        for i in range(len(ps)):
            for j in range(i+1, len(ps)):
                a1, b1 = ps[i]
                a2, b2 = ps[j]
                da = a1 - a2
                db = b1 - b2
                # Distance squared in triangular lattice: a^2 + b^2 + a*b
                # where the point is a + b*omega, omega = exp(i*pi/3)
                # |a + b*omega - (a' + b'*omega)|^2 = (a-a')^2 + (b-b')^2 + (a-a')(b-b')
                dist_sq = da*da + db*db + da*db
                if dist_sq == 1:
                    count += 1
        return count
    
    # Greedy: start with inner points, add outer points one by one
    # choosing the one that adds the most new edges
    selected = set(inner_points)
    remaining = list(outer_candidates)
    
    while len(selected) < n_points and remaining:
        best_point = None
        best_gain = -1
        for pt in remaining:
            # Count edges this point would add with currently selected points
            gain = 0
            for sp in selected:
                da = pt[0] - sp[0]
                db = pt[1] - sp[1]
                dist_sq = da*da + db*db + da*db
                if dist_sq == 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_point = pt
        
        if best_point is None:
            break
        
        selected.add(best_point)
        remaining.remove(best_point)
        
        if time.time() > deadline - 30:
            break
    
    # If we didn't reach n_points, fill with remaining
    while len(selected) < n_points and remaining:
        selected.add(remaining.pop())
    
    # Local search: try to swap one selected outer point with one unselected outer point
    # to see if we can improve
    selected_list = list(selected)
    unselected = set(outer_candidates) - set(selected_list)
    
    improved = True
    while improved:
        improved = False
        current_edges = count_edges(selected)
        
        for sel_pt in selected_list:
            # Only consider swapping outer points
            if sel_pt in inner_points:
                continue
            for unsel_pt in unselected:
                # Try the swap
                new_selected = set(selected)
                new_selected.remove(sel_pt)
                new_selected.add(unsel_pt)
                
                new_edges = count_edges(new_selected)
                if new_edges > current_edges:
                    selected = new_selected
                    current_edges = new_edges
                    improved = True
                    break
            if improved:
                break
        
        if time.time() > deadline - 20:
            break
    
    result = [[a, b] for a, b in selected]
    
    # Verify we have exactly 32 distinct points
    assert len(result) == n_points, f"Expected {n_points} points, got {len(result)}"
    assert len(set(tuple(p) for p in result)) == n_points, "Points are not distinct"
    
    return result
