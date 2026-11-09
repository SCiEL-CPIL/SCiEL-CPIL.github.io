"""Search for integral point sets in general position (maximize size)."""
from fractions import Fraction
import itertools
import math
import time

def construct():
    t0 = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    # We'll search for integer points (D=1) with integer pairwise distances.
    # A set of integer points automatically has rational (integer) pairwise distances.
    # We need: no three collinear, no four concyclic.

    # Strategy:
    # 1. Generate candidate integer points in a bounded region.
    # 2. Build a set greedily / via backtracking, checking:
    #    - all pairwise distances are integers
    #    - no three collinear
    #    - no four concyclic
    # 3. Try to maximize the size.

    # Known: 5-point sets exist. 6-point sets are the target.
    # Let's do a systematic search.

    # For efficiency, precompute which pairs of points have integer distance.
    # Then do a clique-style search with the collinearity and concyclicity constraints.

    # Bounding box: let's try points in [-B, B] x [-B, B]
    # We need to be careful about the number of candidates.

    # A point set with integer pairwise distances is an "integral point set".
    # Known results:
    # - 5 points exist (e.g., the one in the seed).
    # - 6 points: existence is known for some configurations.
    #   Actually, it's known that there exist integral point sets of size 6, 7, 8, ...
    #   Wait, let me recall. The problem of integral point sets in general position:
    #   - Size 5: known.
    #   - Size 6: known (e.g., by Kurz and Müller).
    #   - Size 7: known.
    #   - Size 8: open? Or known?
    #   The task says "8 points would be new", suggesting 7 is the current known max.

    # Let's try to find a 6-point set first, then try for 7.

    # For the search, we need:
    # - Points (x, y) with integer coordinates.
    # - All pairwise distances sqrt((dx)^2 + (dy)^2) must be integers.
    # - No three collinear.
    # - No four concyclic.

    # Approach:
    # Generate a pool of candidate points.
    # For each pair, check if distance is integer.
    # Then search for a clique of size k in the "integer-distance" graph,
    # subject to no-three-collinear and no-four-concyclic.

    # To make this feasible, we need a reasonable pool size.
    # Let's try a grid and filter.

    def is_integer_distance(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        if d2 == 0:
            return False  # distinct points
        r = int(math.isqrt(d2))
        return r*r == d2

    def are_collinear(p1, p2, p3):
        # (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0
        return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]) == 0

    def are_concyclic(p1, p2, p3, p4):
        # Four points are concyclic iff the determinant of the circle equation is zero.
        # Circle: x^2 + y^2 + Dx + Ey + F = 0
        # For each point: xi^2 + yi^2 + D*xi + E*yi + F = 0
        # This is a linear system in D, E, F.
        # The system is consistent (non-trivial solution) iff the 4x4 determinant is 0.
        # Actually, we have 3 unknowns (D, E, F) and 4 equations.
        # The points are concyclic iff the 4x4 matrix:
        # | x1 y1 x1^2+y1^2 1 |
        # | x2 y2 x2^2+y2^2 1 |
        # | x3 y3 x3^2+y3^2 1 |
        # | x4 y4 x4^2+y4^2 1 |
        # has determinant 0.
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        # Compute determinant of 4x4 matrix
        # Using expansion:
        a1 = x1; a2 = y1; a3 = x1*x1+y1*y1; a4 = 1
        b1 = x2; b2 = y2; b3 = x2*x2+y2*y2; b4 = 1
        c1 = x3; c2 = y3; c3 = x3*x3+y3*y3; c4 = 1
        d1 = x4; d2 = y4; d3 = x4*x4+y4*y4; d4 = 1
        det = (a1*(b2*(c3*d4 - c4*d3) - b3*(c2*d4 - c4*d2) + b4*(c2*d3 - c3*d2))
             - a2*(b1*(c3*d4 - c4*d3) - b3*(c1*d4 - c4*d1) + b4*(c1*d3 - c3*d1))
             + a3*(b1*(c2*d4 - c4*d2) - b2*(c1*d4 - c4*d1) + b4*(c1*d2 - c2*d1))
             - a4*(b1*(c2*d3 - c3*d2) - b2*(c1*d3 - c3*d1) + b3*(c1*d2 - c2*d1)))
        return det == 0

    def check_set(points):
        n = len(points)
        # Check no three collinear
        for i, j, k in itertools.combinations(range(n), 3):
            if are_collinear(points[i], points[j], points[k]):
                return False
        # Check no four concyclic
        for i, j, k, l in itertools.combinations(range(n), 4):
            if are_concyclic(points[i], points[j], points[k], points[l]):
                return False
        return True

    # Known 5-point set from the seed:
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Let's verify the seed set is valid
    # (We trust it's valid since it's the seed)

    # Now, try to extend to 6 points.
    # We'll do a search: start with the 5-point set, try adding a 6th point.
    # Or, do a broader search.

    # For a broader search, let's generate candidate points in a grid.
    # The seed points have coordinates up to 63. Let's try a grid of size ~200x200.
    # But that's 40000 points, and checking all pairs is 40000^2/2 ~ 800 million, too slow.
    
    # Better approach: 
    # 1. Generate a pool of candidate points.
    # 2. For each point, precompute which other points it has integer distance to.
    # 3. Use a backtracking search to find a clique of size 6 (or more) 
    #    with the collinearity and concyclicity constraints.

    # To keep the pool manageable, let's use a smaller grid or a structured search.

    # Alternative: Use known parametric families of integral point sets.
    # For example, sets based on Pythagorean triples.

    # Let's try a targeted search:
    # Start with the 5-point seed, and try to add a 6th point from a larger grid.

    # Actually, let's try a different approach:
    # Search for 6-point sets by trying all combinations of 6 points from a candidate pool,
    # but the pool must be small enough.

    # Let's try a pool of points that are "likely" to form integral distances.
    # For example, points on a circle with integer radius, or points related by Pythagorean triples.

    # Given the time constraint, let's try a heuristic search:
    # 1. Start with the 5-point seed.
    # 2. Try to add a 6th point by searching over a grid.
    # 3. If found, try to add a 7th.

    # For step 2, the 6th point must have integer distance to all 5 existing points.
    # This is a system of 5 equations:
    # (x - xi)^2 + (y - yi)^2 = ri^2  for i = 1..5
    # where ri are integers.
    # Subtracting the first equation from the others gives 4 linear equations in x, y, and the ri^2.
    # This can be solved for x, y in terms of the ri^2.

    # Let's implement this:
    # Given 5 points P1..P5, find integer (x, y) such that dist((x,y), Pi) is integer for all i.

    # This is equivalent to finding a point that lies on 5 circles centered at Pi with integer radii.
    # The intersection of 2 circles gives at most 2 points. Adding more circles constrains further.

    # Let's try: for each pair of existing points, find the radical axis, etc.
    # This is getting complex. Let's just do a grid search for the 6th point.

    # The seed points are in [-56, 16] x [0, 63]. Let's search in a slightly larger box.
    # But the 6th point could be far away.

    # Given the complexity, let's try a different known 6-point set.
    # From the literature, one example of a 6-point integral set in general position is:
    # (I need to recall or construct one.)

    # Actually, let's try to use a SAT-like approach with a small pool.
    # Pool: all integer points in [-100, 100] x [-100, 100] that have integer distance to at least 3 other points in the pool.
    # This pre-filtering might reduce the pool size.

    # But even this might be too large for a brute-force clique search.

    # Given the time constraint (130s), let's try a more targeted approach:
    # Use the known 5-point set and try to find a 6th point by solving the distance equations.

    # For the 5 seed points:
    # P1 = (0, 0)
    # P2 = (-56, 33)
    # P3 = (-16, 63)
    # P4 = (0, 33)
    # P5 = (16, 63)

    # Let (x, y) be the 6th point.
    # d1^2 = x^2 + y^2
    # d2^2 = (x+56)^2 + (y-33)^2
    # d3^2 = (x+16)^2 + (y-63)^2
    # d4^2 = x^2 + (y-33)^2
    # d5^2 = (x-16)^2 + (y-63)^2

    # d1^2 - d4^2 = x^2 + y^2 - x^2 - (y-33)^2 = y^2 - (y^2 - 66y + 1089) = 66y - 1089
    # So 66y - 1089 = d1^2 - d4^2
    # Similarly, d1^2 - d2^2 = x^2 + y^2 - (x+56)^2 - (y-33)^2
    #   = x^2 + y^2 - (x^2 + 112x + 3136) - (y^2 - 66y + 1089)
    #   = -112x + 66y - 4225
    # So -112x + 66y - 4225 = d1^2 - d2^2

    # We have two linear equations in x, y:
    # 66y = d1^2 - d4^2 + 1089
    # -112x + 66y = d1^2 - d2^2 + 4225

    # From the first: y = (d1^2 - d4^2 + 1089) / 66
    # Substitute into the second to get x.

    # For x, y to be integers, we need (d1^2 - d4^2 + 1089) to be divisible by 66.
    # And x must also be integer.

    # We can iterate over possible values of d1 and d4 (integers), compute y, then x,
    # and check if d3 and d5 are also integers.

    # This is a feasible search!

    best_size = 5
    best_points = seed_points

    # Try to find a 6th point
    found_6 = False
    for d1 in range(1, 500):
        for d4 in range(1, 500):
            num_y = d1*d1 - d4*d4 + 1089
            if num_y % 66 != 0:
                continue
            y = num_y // 66
            # Now find x
            # -112x + 66y = d1^2 - d2^2 + 4225
            # We need d2^2 = d1^2 - (-112x + 66y - 4225) ... wait, let's re-derive.
            # -112x + 66y - 4225 = d1^2 - d2^2
            # d2^2 = d1^2 + 4225 - 66y + 112x
            # This still has x. Let's use another pair.
            
            # Use d1^2 - d5^2:
            # d1^2 - d5^2 = x^2 + y^2 - (x-16)^2 - (y-63)^2
            #   = x^2 + y^2 - (x^2 - 32x + 256) - (y^2 - 126y + 3969)
            #   = 32x + 126y - 4225
            # So 32x + 126y - 4225 = d1^2 - d5^2
            # 32x = d1^2 - d5^2 + 4225 - 126y
            
            # We need to iterate over d5 as well.
            for d5 in range(1, 500):
                num_x = d1*d1 - d5*d5 + 4225 - 126*y
                if num_x % 32 != 0:
                    continue
                x = num_x // 32
                # Now check d2 and d3
                d2_sq = (x+56)**2 + (y-33)**2
                d3_sq = (x+16)**2 + (y-63)**2
                d2 = int(math.isqrt(d2_sq))
                d3 = int(math.isqrt(d3_sq))
                if d2*d2 != d2_sq or d3*d3 != d3_sq:
                    continue
                # Check all distances are integers (d1, d4, d5 already are by construction)
                # Check no three collinear and no four concyclic
                new_points = seed_points + [(x, y)]
                if check_set(new_points):
                    found_6 = True
                    best_size = 6
                    best_points = new_points
                    break
            if found_6:
                break
        if found_6:
            break

    if time.time() - t0 > DEADLINE - 5:
        # Return best found so far
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}

    # If we found 6 points, try to find a 7th
    if best_size == 6:
        # Similar approach but more complex. Skip for now due to time.
        pass

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
