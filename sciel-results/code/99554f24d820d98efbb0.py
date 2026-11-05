import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper functions for exact arithmetic
    def is_square(n):
        if n < 0:
            return False
        r = int(n ** 0.5)
        return r * r == n or (r + 1) * (r + 1) == n

    def dist_sq(p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def is_collinear(p1, p2, p3):
        # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def is_concyclic(p1, p2, p3, p4):
        # Determinant of the matrix with rows [x, y, x^2+y^2, 1]
        # Or use the circle equation: x^2 + y^2 + Dx + Ey + F = 0
        # Solve for D, E, F using p1, p2, p3 and check if p4 satisfies it.
        # To avoid division, we can use the determinant form.
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Determinant of:
        # x1^2+y1^2, x1, y1, 1
        # x2^2+y2^2, x2, y2, 1
        # x3^2+y3^2, x3, y3, 1
        # x4^2+y4^2, x4, y4, 1
        a1 = x1*x1 + y1*y1
        a2 = x2*x2 + y2*y2
        a3 = x3*x3 + y3*y3
        a4 = x4*x4 + y4*y4
        
        # Calculate determinant
        # Using expansion along the last column for efficiency?
        # Or just direct formula for 4x4.
        # det = a1*(x2*y3 - x3*y2 - x2*y4 + x4*y2 + x3*y4 - x4*y3)
        #     - x1*(a2*y3 - a3*y2 - a2*y4 + a4*y2 + a3*y4 - a4*y3)
        #     + y1*(a2*x3 - a3*x2 - a2*x4 + a4*x2 + a3*x4 - a4*x3)
        #     - 1*(a2*x3*y4 - a3*x2*y4 - a2*x3*y4 + a4*x2*y4 + a3*x2*y4 - a4*x2*y3 ... wait, easier to use 2x2 minors)
        
        # Let's use the property: 4 points are concyclic iff the determinant is 0.
        # det = | x1^2+y1^2  x1  y1  1 |
        #       | x2^2+y2^2  x2  y2  1 |
        #       | x3^2+y3^2  x3  y3  1 |
        #       | x4^2+y4^2  x4  y4  1 |
        
        # Subtract row 1 from others to simplify?
        # Let's just compute it directly.
        
        def det4(m):
            # m is 4x4
            return (m[0][0] * (m[1][1]*m[2][2]*m[3][3] - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - m[1][3]*m[2][2]*m[3][1])
                  - m[0][1] * (m[1][0]*m[2][2]*m[3][3] - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - m[1][3]*m[2][2]*m[3][0])
                  + m[0][2] * (m[1][0]*m[2][1]*m[3][3] - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - m[1][3]*m[2][1]*m[3][0])
                  - m[0][3] * (m[1][0]*m[2][1]*m[3][2] - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - m[1][2]*m[2][1]*m[3][0]))
                  
        m = [[a1, x1, y1, 1],
             [a2, x2, y2, 1],
             [a3, x3, y3, 1],
             [a4, x4, y4, 1]]
        return det4(m) == 0

    # Generate candidate points
    # We search for integer points first.
    # If no 6-point set is found in a reasonable time, we return the best found.
    
    best_set = []
    
    # Try increasing bounding box sizes
    for N in range(1, 20):
        if time.time() > deadline:
            break
            
        # Generate all integer points in [-N, N] x [-N, N]
        # Exclude duplicates? Points must be distinct.
        candidates = []
        for x in range(-N, N + 1):
            for y in range(-N, N + 1):
                candidates.append((x, y))
        
        # Sort candidates to make backtracking deterministic and potentially faster
        # (e.g., by x then y)
        candidates.sort()
        
        # Precompute distance squared and check if rational (integer squared is always rational)
        # For integer points, dist_sq is integer, so dist is rational iff dist_sq is a perfect square.
        
        # Backtracking search for clique of size 6
        # We want to find a subset of size 6 such that:
        # 1. All pairwise distances are rational (perfect squares for integer points)
        # 2. No three collinear
        # 3. No four concyclic
        
        # Optimization: 
        # If we have a set S, and we add p, we check:
        # - dist(p, q) is rational for all q in S
        # - p is not collinear with any pair in S
        # - p is not concyclic with any triple in S
        
        found = False
        best_size = 0
        best_combo = []
        
        # To speed up, we can prune candidates that are too close or form bad configurations with a fixed anchor?
        # Let's just run backtracking on the full candidate set for small N.
        
        def backtrack(current_set, remaining_candidates):
            nonlocal found, best_size, best_combo, deadline
            
            if time.time() > deadline:
                return
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_combo = current_set.copy()
                
            if len(current_set) == 6:
                found = True
                return
            
            if not remaining_candidates:
                return
                
            # Heuristic: Try to add points that are "far" or "compatible"
            # We iterate through remaining candidates
            for i, p in enumerate(remaining_candidates):
                # Check if p can be added to current_set
                compatible = True
                for q in current_set:
                    d2 = dist_sq(p, q)
                    if not is_square(d2):
                        compatible = False
                        break
                
                if not compatible:
                    continue
                    
                # Check collinearity with any pair in current_set
                if len(current_set) >= 2:
                    for p1, p2 in combinations(current_set, 2):
                        if is_collinear(p1, p2, p):
                            compatible = False
                            break
                            
                if not compatible:
                    continue
                
                # Check concyclicity with any triple in current_set
                if len(current_set) >= 3:
                    for p1, p2, p3 in combinations(current_set, 3):
                        if is_concyclic(p1, p2, p3, p):
                            compatible = False
                            break
                            
                if not compatible:
                    continue
                
                # If compatible, add p and recurse
                backtrack(current_set + [p], remaining_candidates[i+1:])
                
                if found:
                    return

        # Start backtracking
        # We can start with an empty set, but to reduce search space, we can fix the first point to (0,0) due to translation invariance?
        # The problem is invariant under translation. So we can fix one point to (0,0).
        # This reduces the search space significantly.
        # Let's fix the first point to (0,0).
        
        # Filter candidates to exclude (0,0) for the remaining slots, but start with [(0,0)]
        # Note: (0,0) must be in candidates. It is if N>=0.
        
        # However, the set doesn't HAVE to contain (0,0). But any valid set can be translated to contain (0,0).
        # So finding a set containing (0,0) is equivalent to finding any set.
        
        # Let's modify backtrack to start with [(0,0)]
        # But wait, if the optimal set doesn't have an integer point at (0,0) after translation?
        # If the points are integers, translation by integer vector keeps them integers.
        # So yes, we can fix one point to (0,0).
        
        # Let's restart the search with this constraint.
        
        # We need to ensure (0,0) is in the candidate list.
        # If N>=0, it is.
        
        # We'll run a separate backtracking that starts with [(0,0)]
        
        def backtrack_fixed(current_set, remaining_candidates):
            nonlocal found, best_size, best_combo, deadline
            
            if time.time() > deadline:
                return
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_combo = current_set.copy()
                
            if len(current_set) == 6:
                found = True
                return
            
            if not remaining_candidates:
                return
                
            for i, p in enumerate(remaining_candidates):
                # Check if p can be added to current_set
                compatible = True
                for q in current_set:
                    d2 = dist_sq(p, q)
                    if not is_square(d2):
                        compatible = False
                        break
                
                if not compatible:
                    continue
                    
                # Check collinearity with any pair in current_set
                if len(current_set) >= 2:
                    for p1, p2 in combinations(current_set, 2):
                        if is_collinear(p1, p2, p):
                            compatible = False
                            break
                            
                if not compatible:
                    continue
                
                # Check concyclicity with any triple in current_set
                if len(current_set) >= 3:
                    for p1, p2, p3 in combinations(current_set, 3):
                        if is_concyclic(p1, p2, p3, p):
                            compatible = False
                            break
                            
                if not compatible:
                    continue
                
                backtrack_fixed(current_set + [p], remaining_candidates[i+1:])
                
                if found:
                    return

        # Candidates excluding (0,0)
        candidates_no_zero = [p for p in candidates if p != (0,0)]
        
        # Start with (0,0)
        backtrack_fixed([(0,0)], candidates_no_zero)
        
        if found:
            # We found a 6-point set containing (0,0)
            # We need to output the points in the format [[x_num, x_den], [y_num, y_den]]
            # D=1
            points_out = []
            for p in best_combo:
                points_out.append([[p[0], 1], [p[1], 1]])
            return {'D': 1, 'points': points_out}
        
        # If we didn't find 6 points, we might have found a smaller set.
        # But the task asks to maximize. If we found 5, we return 5.
        # However, we want to try larger N if we didn't find 6.
        # But if we found a set of size 5, we should remember it.
        
        if best_size > len(best_set):
            best_set = best_combo

    # If no 6-point set found, return the best set found (likely 5 or fewer)
    # But we should ensure we return a valid set.
    # If best_set is empty, we can return a trivial set like 3 points.
    if not best_set:
        # Fallback: 3 points
        best_set = [(0,0), (1,0), (0,1)]
        
    points_out = []
    for p in best_set:
        points_out.append([[p[0], 1], [p[1], 1]])
    
    return {'D': 1, 'points': points_out}
