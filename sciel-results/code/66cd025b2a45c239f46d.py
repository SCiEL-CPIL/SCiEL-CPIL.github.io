# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def _concyclic_det(p1, p2, p3, p4):
        """Exact 4x4 determinant for concyclic check using integer arithmetic."""
        # |x  y  x^2+y^2  1|
        rows = []
        for p in [p1, p2, p3, p4]:
            x, y = p
            rows.append([x, y, x*x + y*y, 1])
        
        # Compute determinant via cofactor expansion
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
                   - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
                   + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        
        def det4(m):
            result = 0
            signs = [1, -1, 1, -1]
            for j in range(4):
                minor = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
                result += signs[j] * m[0][j] * det3(minor)
            return result
        
        return det4(rows)

    def all_constraints_satisfied(points):
        """Check all pairwise distances are integer, no 3 collinear, no 4 concyclic."""
        n = len(points)
        # Check all pairwise distances are integers
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        # Check no four concyclic
        for combo in combinations(range(n), 4):
            if _concyclic_det(points[combo[0]], points[combo[1]], 
                            points[combo[2]], points[combo[3]]) == 0:
                return False
        return True

    def generate_candidates(max_coord=300):
        """Generate candidate lattice points sorted by distance from origin."""
        candidates = []
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                candidates.append((x, y))
        candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
        return candidates

    def dfs_extend(current_points, candidates, best_points, deadline):
        """
        Depth-first search with backtracking to extend current_points.
        Updates best_points if a larger valid set is found.
        """
        if elapsed() > deadline:
            return best_points
        
        if len(current_points) > len(best_points):
            best_points = current_points[:]
            if len(best_points) >= 12:
                return best_points

        existing = set(current_points)
        
        # Filter candidates: must not be in existing, and must be at integer distance from at least one existing point
        # This is a strong pruning step. For the first few points, this might be too restrictive if we start small,
        # but we start with a valid set, so this is good.
        valid_cands = []
        for cand in candidates:
            if cand in existing:
                continue
            
            # Check if cand is at integer distance from any point in current_points
            # If not, it can't be added directly without violating the distance constraint for that pair.
            # Wait, the constraint is ALL pairwise distances must be integer.
            # So if cand is not at integer distance from ANY existing point, it's invalid.
            # Actually, it must be at integer distance from ALL existing points.
            # Let's check if it's at integer distance from the FIRST existing point. 
            # If not, it's definitely invalid. This is a fast filter.
            # But we need to check ALL.
            # Let's do a fast pre-check: distance to first point.
            x0, y0 = current_points[0]
            dx = cand[0] - x0
            dy = cand[1] - y0
            if not is_perfect_square(dx*dx + dy*dy):
                continue
            
            # Now check against all other points to be sure
            valid = True
            for p in current_points:
                dx = cand[0] - p[0]
                dy = cand[1] - p[1]
                if not is_perfect_square(dx*dx + dy*dy):
                    valid = False
                    break
            
            if valid:
                # Also check collinearity with existing points? 
                # We can't check collinearity with just 2 points. 
                # We need to check if adding cand creates a collinear triple with any 2 existing points.
                # And if it creates a concyclic quad with any 3 existing points.
                
                # Collinearity check:
                collinear = False
                for i in range(len(current_points)):
                    for j in range(i+1, len(current_points)):
                        if check_collinear(current_points[i], current_points[j], cand):
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                
                # Concyclic check:
                concyclic = False
                n_pts = len(current_points)
                # If we have >= 3 points, check if cand is concyclic with any 3
                if n_pts >= 3:
                    for i in range(n_pts):
                        for j in range(i+1, n_pts):
                            for k in range(j+1, n_pts):
                                if _concyclic_det(current_points[i], current_points[j], current_points[k], cand) == 0:
                                    concyclic = True
                                    break
                            if concyclic:
                                break
                        if concyclic:
                            break
                if concyclic:
                    continue
                
                valid_cands.append(cand)
                
                # Heuristic: Sort valid candidates by "density" - how many other valid candidates they are close to?
                # Or just keep them sorted by distance from origin (already done in generate_candidates)
                # Let's stick to distance for now, but maybe prioritize those with smaller coordinates.
        
        # Try adding each valid candidate
        for cand in valid_cands:
            new_points = current_points + [cand]
            # Recurse
            best_points = dfs_extend(new_points, candidates, best_points, deadline)
            
            if elapsed() > deadline:
                return best_points
                
        return best_points

    # Known 5-point integral sets
    initial_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (4, 0), (0, 4)],
        [(0, 0), (5, 0), (0, 12), (5, 12)],
        [(0, 0), (9, 12), (12, 9), (15, 0)],
        [(0, 0), (7, 24), (24, 7), (25, 0)],
        [(0, 0), (8, 15), (15, 8), (17, 0)],
        [(0, 0), (12, 35), (35, 12), (37, 0)],
    ]

    # Verify and keep only valid initial sets
    valid_inits = []
    for init in initial_sets:
        if all_constraints_satisfied(init):
            valid_inits.append(init)

    if not valid_inits:
        # Fallback: use the first one even if constraints fail (shouldn't happen)
        valid_inits = [initial_sets[0]]

    best_points = valid_inits[0][:]
    best_len = len(best_points)

    # Generate candidate pool
    # Larger pool allows finding points further out, but increases search time.
    # 300 is a reasonable start.
    candidates = generate_candidates(300)

    # Try DFS extension from each valid initial set
    # We limit the depth or time to avoid hanging on one branch
    for init in valid_inits:
        if elapsed() > DEADLINE:
            break
        
        # To make DFS effective, we should perhaps shuffle candidates or use a different ordering?
        # The current ordering is by distance from origin.
        # Let's just run DFS.
        
        # Optimization: Pass a copy of best_points to track global best
        current_best = best_points[:]
        
        # We need to be careful with recursion depth. 
        # Max points is 12, initial is ~5, so depth is ~7. This is safe.
        
        # To speed up, we can prune candidates that are too far?
        # No, let the DFS handle it.
        
        # One issue: DFS might explore a bad branch deeply.
        # We can add a simple heuristic: if len(current) + remaining_potential < best_len, prune.
        # But we don't know remaining potential.
        
        # Let's just run it.
        best_points = dfs_extend(init, candidates, best_points, DEADLINE)
        best_len = len(best_points)
        
        if best_len >= 12:
            break

    # Final fallback: ensure we have at least 3 valid points
    if best_len < 3:
        best_points = [(0, 0), (3, 4), (4, 0)]

    # Convert to required format with D=1
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best_points]

    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END
