"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # Keep 50s margin for 330s limit

    all_points = [(x, y) for x in range(N) for y in range(N)]
    num_points = N * N

    # We'll use a greedy + local search approach
    # First, build an initial solution greedily
    # Then, improve with local search (swap, add, remove operations)

    def is_isosceles(p1, p2, p3):
        """Check if three points form an isosceles triangle."""
        d12 = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        d23 = (p2[0] - p3[0])**2 + (p2[1] - p3[1])**2
        d13 = (p1[0] - p3[0])**2 + (p1[1] - p3[1])**2
        return d12 == d23 or d12 == d13 or d23 == d13

    def can_add(S_set, dist_from, p):
        """Check if point p can be added to set S without creating an isosceles triangle."""
        px, py = p
        for q in S_set:
            q2 = (q[0] - px)**2 + (q[1] - py)**2
            if q2 == 0:
                return False
        # For each existing point q, check if adding p creates an isosceles triangle with any other point r
        # i.e., for each pair (q, r) in S, check if triangle (p, q, r) is isosceles
        # This is O(|S|^2) which is slow. Let's use a smarter approach.
        
        # Actually, let's think about this differently.
        # When we add p to S, we need to check all pairs (q, r) in S to see if (p, q, r) is isosceles.
        # That's O(|S|^2) per candidate, which is too slow for |S| ~ 100+.
        
        # Better approach: for each existing point q, the new point p must not be equidistant from q and any other point r in S.
        # So for each q in S, compute d(p, q). Then check if d(p, q) equals d(p, r) for any r != q in S.
        # But that's still O(|S|^2) overall.
        
        # Alternative: For each pair (q, r) in S, the locus of points p such that (p, q, r) is isosceles is:
        # - The perpendicular bisector of qr (for pq = pr)
        # - The circle centered at q with radius qr (for q is apex)
        # - The circle centered at r with radius qr (for r is apex)
        # We can precompute forbidden positions, but that's complex.
        
        # Let's just do the O(|S|^2) check but only when |S| is small, and use heuristics when large.
        
        # Actually, for the greedy phase, let's use a different strategy:
        # Try to add points in a specific order and check incrementally.
        
        # For now, let's do a direct check but limit it
        S_list = list(S_set)
        n = len(S_list)
        if n == 0:
            return True
        if n == 1:
            return True  # Can't form triangle with 2 points
        # Check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                q = S_list[i]
                r = S_list[j]
                d12 = (p[0] - q[0])**2 + (p[1] - q[1])**2
                d23 = (q[0] - r[0])**2 + (q[1] - r[1])**2
                d13 = (p[0] - r[0])**2 + (p[1] - r[1])**2
                if d12 == d23 or d12 == d13 or d23 == d13:
                    return False
        return True

    # Greedy construction
    S = set()
    random.shuffle(all_points)
    for p in all_points:
        if time.time() > deadline - 60:
            break
        if can_add(S, None, p):
            S.add(p)

    best_set = S
    best_size = len(S)

    # Local search: try to improve
    # Operations: add a point, remove a point, swap (remove one, add another)
    # We'll do a simple hill climbing with random restarts

    # Precompute all squared distances between points for speed
    # Actually, let's just do direct checks

    # Local search phase
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 30:
        improved = False
        
        # Try to add a random point
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            p = random.choice(all_points)
            if p not in S:
                if can_add(S, None, p):
                    S.add(p)
                    improved = True
                    no_improve_count = 0
                    break
        
        # Try to swap: remove one point, add another
        if not improved and len(S) > 0:
            for _ in range(20):
                if time.time() > deadline - 30:
                    break
                remove_p = random.choice(list(S))
                add_p = random.choice(all_points)
                if add_p in S or add_p == remove_p:
                    continue
                S_temp = S.copy()
                S_temp.remove(remove_p)
                if can_add(S_temp, None, add_p):
                    S_temp.add(add_p)
                    S = S_temp
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart: rebuild with different randomization
                if time.time() > deadline - 30:
                    break
                S = set()
                random.shuffle(all_points)
                for p in all_points:
                    if time.time() > deadline - 30:
                        break
                    if can_add(S, None, p):
                        S.add(p)
                no_improve_count = 0

        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)

    return list(best_set)
# EVOLVE-BLOCK-END
