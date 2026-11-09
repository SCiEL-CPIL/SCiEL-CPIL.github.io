"""Search for integral-point sets in general position (rational distances, no 3 collinear, no 4 concyclic)."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 130.0  # safety margin under 150s

    # ---- Exact helpers ----
    def is_square(n):
        if n < 0:
            return False
        if n == 0:
            return True
        r = isqrt(n)
        return r * r == n

    def isqrt(n):
        if n < 0:
            raise ValueError
        if n == 0:
            return 0
        x = n
        y = (x + 1) // 2
        while y < x:
            x = y
            y = (x + n // x) // 2
        return x

    def dist2(p, q):
        """Squared distance between two points (x1,y1),(x2,y2) as Fraction."""
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx * dx + dy * dy

    def all_rational_distances(points):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d2 = dist2(points[i], points[j])
                if d2 <= 0:
                    return False
                if not is_square(d2.numerator) or not is_square(d2.denominator):
                    return False
        return True

    def no_three_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # cross product (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
                    cr = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                    if cr == 0:
                        return False
        return True

    def no_four_concyclic(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        # General circle through (x1,y1),(x2,y2),(x3,y3):
                        # A x^2 + A y^2 + B x + C y + D = 0
                        # Solve for A,B,C,D (up to scale)
                        # (x1^2+y1^2)A + x1 B + y1 C + D = 0
                        # (x2^2+y2^2)A + x2 B + y2 C + D = 0
                        # (x3^2+y3^2)A + x3 B + y3 C + D = 0
                        # Check if (x4,y4) lies on it.
                        a1 = x1*x1 + y1*y1
                        a2 = x2*x2 + y2*y2
                        a3 = x3*x3 + y3*y3
                        a4 = x4*x4 + y4*y4
                        # Use determinants of 4x4 matrix
                        # |a1 x1 y1 1|
                        # |a2 x2 y2 1|
                        # |a3 x3 y3 1| = 0
                        # |a4 x4 y4 1|
                        det = (a1*(x2*(y3-y4) - x3*(y2-y4)) - x1*(a2*(y3-y4) - a3*(y2-y4))
                               + y1*(a2*(x3-x4) - a3*(x2-x4)) - 1*(a2*(x2*y3-x3*y2) - a3*(x2*y3-x3*y2)))
                        # Let me redo this more carefully using expansion
                        # det = a1*(x2*(y3-1) - y2*(x3-1) + 1*(x3*y2-y3*x2))  ... no
                        # Let me just compute the 4x4 determinant directly
                        # Rows: [a1,x1,y1,1], [a2,x2,y2,1], [a3,x3,y3,1], [a4,x4,y4,1]
                        # Using cofactor expansion along row 1:
                        # a1 * det[[x2,y2,1],[x3,y3,1],[x4,y4,1]]
                        # - x1 * det[[a2,y2,1],[a3,y3,1],[a4,y4,1]]
                        # + y1 * det[[a2,x2,1],[a3,x3,1],[a4,x4,1]]
                        # - 1 * det[[a2,x2,y2],[a3,x3,y3],[a4,x4,y4]]
                        def det3(r1, r2, r3):
                            return (r1[0]*(r2[1]*r3[2]-r2[2]*r3[1])
                                    - r1[1]*(r2[0]*r3[2]-r2[2]*r3[0])
                                    + r1[2]*(r2[0]*r3[1]-r2[1]*r3[0]))
                        M1 = (x2*(y3-y4) - y2*(x3-x4) + 1*(x3*y4-y3*x4))
                        M2 = (a2*(y3-y4) - y2*(a3-a4) + 1*(a3*y4-a3*y4))  # wrong, let me redo
                        # M2 = det[[a2,y2,1],[a3,y3,1],[a4,y4,1]]
                        M2 = a2*(y3*1 - 1*y4) - y2*(a3*1 - 1*a4) + 1*(a3*y4 - y3*a4)
                        # M3 = det[[a2,x2,1],[a3,x3,1],[a4,x4,1]]
                        M3 = a2*(x3*1 - 1*x4) - x2*(a3*1 - 1*a4) + 1*(a3*x4 - x3*a4)
                        # M4 = det[[a2,x2,y2],[a3,x3,y3],[a4,x4,y4]]
                        M4 = a2*(x3*y4 - y3*x4) - x2*(a3*y4 - y3*a4) + y2*(a3*x4 - x3*a4)
                        det_val = a1*M1 - x1*M2 + y1*M3 - 1*M4
                        if det_val == 0:
                            return False
        return True

    def validate(points):
        if len(points) < 3:
            return False
        # Check distinctness
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if points[i] == points[j]:
                    return False
        if not all_rational_distances(points):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True

    def to_output(points, D=1):
        return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]}

    # ---- Known 5-point set (fallback) ----
    known5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)), (Fraction(-16), Fraction(63)),
              (Fraction(0), Fraction(33)), (Fraction(16), Fraction(63))]
    best = known5

    # ---- Search for 6-point sets ----
    # Strategy: Start from known 5-point sets and try to add a 6th point,
    # or do a local search over rational lattice points.
    # Also try systematic search over small rational coordinates.

    # Generate candidate rational points on a grid
    # For D=1, we need integer distances.
    # Let's search over points with rational coordinates where all pairwise distances are rational.

    # Approach: Build up sets incrementally.
    # Start with pairs of points at rational distance, then add more.

    # For efficiency, work with integer-scaled coordinates.
    # If we scale all coordinates by a common denominator, we get integer coordinates,
    # and rational distances become integer distances after scaling.

    # Let's try a direct search: pick 6 points from a bounded grid of rational coords.
    # To keep it tractable, use small denominators.

    # Try: points with coordinates in {0, 1, 2, 3, 4, 5} / {1, 2, 3}
    # But that's too many combos. Instead, use a greedy/local search.

    # Better: Use known constructions.
    # A 6-point rational distance set: take a rectangle and add center + one more?
    # Rectangle (0,0),(a,0),(a,b),(0,b): distances a, b, sqrt(a^2+b^2).
    # Need a, b, sqrt(a^2+b^2) all rational -> Pythagorean triple.
    # E.g., (3,0),(3,4),(0,4),(0,0): distances 3,4,5. Good.
    # Add (3/2, 0) and (0, 4/2)=(0,2)? Check all distances.
    # Let's try: (0,0),(3,0),(3,4),(0,4) + (1.5, 0) + (0, 2)
    # Distances from (1.5,0) to (0,0)=1.5, to (3,0)=1.5, to (3,4)=sqrt(1.5^2+4^2)=sqrt(2.25+16)=sqrt(18.25)=sqrt(73/4)=sqrt(73)/2. Not rational.

    # Try (0,0),(4,0),(4,3),(0,3) + (2,0) + (0,1.5)
    # (2,0) to (4,3): sqrt(4+9)=sqrt(13). No.

    # Let me try a different approach: use the fact that we can search over
    # integer points with integer distances (Erdos-type problem).
    # Known: max number of integer-distance points in plane is at most 5 (open for 6+).
    # But we allow rational coordinates, not just integers.

    # For rational coordinates, we can scale to integers. So we're looking for
    # integer points with integer pairwise distances. The max known is 5.
    # But with rational (non-integer) coordinates, we might do better.

    # Actually, any rational-distance set with rational coordinates can be scaled to
    # integer coordinates with integer distances (multiply by LCM of denominators).
    # So the question reduces to: max number of integer points with integer pairwise distances.
    # This is known to be at most 5 (Erdos, 1950s). So 6 might not exist for D=1!

    # Therefore, we need D > 1. With D > 1, the y-coordinates are y*sqrt(D),
    # and distances involve sqrt(D). We need all pairwise distances to be rational.

    # For two points (x1, y1*sqrt(D)) and (x2, y2*sqrt(D)):
    # dist^2 = (x1-x2)^2 + D*(y1-y2)^2
    # We need this to be a perfect square of a rational.

    # Let's try D=3 and search for 6 points.
    # Or try a parametric family.

    # Given time constraints, let's do a targeted search:
    # Fix D, and search over small rational x,y values.

    def search_for_6(D, max_coord_num, max_coord_den, time_limit):
        """Search for 6 points with rational coordinates and rational distances for given D."""
        t_start = time.time()
        # Generate candidate points: (x_num/x_den, y_num/y_den) where |num|<=max_coord_num, den<=max_coord_den
        # To keep tractable, use small ranges.
        # For D=3, we need (x1-x2)^2 + 3*(y1-y2)^2 to be a rational square.

        # Let's try integer x and y first (den=1) for simplicity, then expand.
        candidates = []
        for x_num in range(-max_coord_num, max_coord_num + 1):
            for y_num in range(-max_coord_num, max_coord_num + 1):
                candidates.append((Fraction(x_num), Fraction(y_num)))

        # For each candidate, precompute distance squared to all others
        n_cand = len(candidates)
        # dist2_matrix[i][j] = (xi-xj)^2 + D*(yi-yj)^2
        dist2_mat = [[None]*n_cand for _ in range(n_cand)]
        for i in range(n_cand):
            xi, yi = candidates[i]
            for j in range(i+1, n_cand):
                xj, yj = candidates[j]
                dx = xi - xj
                dy = yi - yj
                d2 = dx*dx + D*dy*dy
                dist2_mat[i][j] = d2
                dist2_mat[j][i] = d2

        # Check which pairs have rational distance
        rational_pair = [[False]*n_cand for _ in range(n_cand)]
        for i in range(n_cand):
            for j in range(i+1, n_cand):
                d2 = dist2_mat[i][j]
                if d2 > 0 and is_square(d2.numerator) and is_square(d2.denominator):
                    rational_pair[i][j] = True
                    rational_pair[j][i] = True

        # Now find a 6-clique in the graph where edges = rational distance
        # Then check no 3 collinear, no 4 concyclic.
        # Use backtracking to find a 6-clique.

        # Build adjacency
        adj = [set() for _ in range(n_cand)]
        for i in range(n_cand):
            for j in range(i+1, n_cand):
                if rational_pair[i][j]:
                    adj[i].add(j)
                    adj[j].add(i)

        # Backtracking clique search
        found = None
        def backtrack(clique, candidates_set):
            nonlocal found
            if len(clique) == 6:
                # Validate: no 3 collinear, no 4 concyclic
                pts = [candidates[i] for i in clique]
                if no_three_collinear(pts) and no_four_concyclic(pts):
                    found = pts
                    return True
                return False
            if time.time() - t_start > time_limit:
                return False
            # Try adding each candidate from candidates_set
            for c in list(candidates_set):
                if all(c in adj[clique[i]] for i in range(len(clique))):
                    new_clique = clique + [c]
                    new_cands = candidates_set & adj[c]
                    if backtrack(new_clique, new_cands):
                        return True
                    candidates_set.discard(c)
            return False

        # Start with all candidates
        all_cands = set(range(n_cand))
        backtrack([], all_cands)
        return found

    # Try D=3 with small coordinate range
    # Keep range small to stay within time
    result6 = search_for_6(3, 5, 1, 10)
    if result6 is not None:
        best = result6

    if time.time() > DEADLINE:
        return to_output(best)

    # Try D=2
    result6_2 = search_for_6(2, 5, 1, 10)
    if result6_2 is not None:
        best = result6_2

    if time.time() > DEADLINE:
        return to_output(best)

    # Try larger coordinate ranges for D=3
    result6_3 = search_for_6(3, 8, 1, 15)
    if result6_3 is not None:
        best = result6_3

    if time.time() > DEADLINE:
        return to_output(best)

    # Try with denominator 2 for more flexibility
    def search_for_6_frac(D, max_num, max_den, time_limit):
        t_start = time.time()
        candidates = []
        for x_num in range(-max_num, max_num + 1):
            for x_den in range(1, max_den + 1):
                for y_num in range(-max_num, max_num + 1):
                    for y_den in range(1, max_den + 1):
                        candidates.append((Fraction(x_num, x_den), Fraction(y_num, y_den)))
        # Remove duplicates
        candidates = list(set(candidates))
        n_cand = len(candidates)
        if n_cand > 500:
            return None  # Too many
        dist2_mat = [[None]*n_cand for _ in range(n_cand)]
        for i in range(n_cand):
            xi, yi = candidates[i]
            for j in range(i+1, n_cand):
                xj, yj = candidates[j]
                dx = xi - xj
                dy = yi - yj
                d2 = dx*dx + D*dy*dy
                dist2_mat[i][j] = d2
                dist2_mat[j][i] = d2
        rational_pair = [[False]*n_cand for _ in range(n_cand)]
        for i in range(n_cand):
            for j in range(i+1, n_cand):
                d2 = dist2_mat[i][j]
                if d2 > 0 and is_square(d2.numerator) and is_square(d2.denominator):
                    rational_pair[i][j] = True
                    rational_pair[j][i] = True
        adj = [set() for _ in range(n_cand)]
        for i in range(n_cand):
            for j in range(i+1, n_cand):
                if rational_pair[i][j]:
                    adj[i].add(j)
                    adj[j].add(i)
        found = None
        def backtrack(clique, candidates_set):
            nonlocal found
            if len(clique) == 6:
                pts = [candidates[i] for i in clique]
                if no_three_collinear(pts) and no_four_concyclic(pts):
                    found = pts
                    return True
                return False
            if time.time() - t_start > time_limit:
                return False
            for c in list(candidates_set):
                if all(c in adj[clique[i]] for i in range(len(clique))):
                    new_clique = clique + [c]
                    new_cands = candidates_set & adj[c]
                    if backtrack(new_clique, new_cands):
                        return True
                    candidates_set.discard(c)
            return False
        all_cands = set(range(n_cand))
        backtrack([], all_cands)
        return found

    if time.time() < DEADLINE - 20:
        r = search_for_6_frac(3, 4, 2, 10)
        if r is not None:
            best = r

    if time.time() > DEADLINE:
        return to_output(best)

    return to_output(best)

# EVOLVE-BLOCK-END
