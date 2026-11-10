"""No-isosceles set: greedy + local improvement with deadline guard."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 315  # safety margin before 330s kill
    rng = random.Random(12345)

    all_points = [(x, y) for x in range(N) for y in range(N)]

    def sq(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def try_add(pts, dist_sets, p):
        """Return True if p can be added without creating an isosceles triangle."""
        # Check apex at p: all distances from p to existing must be distinct
        seen = set()
        for u in pts:
            d = sq(p, u)
            if d in seen:
                return False
            seen.add(d)
        # Check apex at each existing u: dist(p,u) not already a distance from u
        for i, u in enumerate(pts):
            d = sq(p, u)
            if d in dist_sets[i]:
                return False
        return True

    def commit(pts, dist_sets, p):
        """Add p to pts and update dist_sets."""
        new_set = set()
        for u in pts:
            d = sq(p, u)
            new_set.add(d)
            dist_sets[pts.index(u)]  # placeholder
        # More efficient: append
        pts.append(p)
        dist_sets.append(new_set)
        for i, u in enumerate(pts[:-1]):
            dist_sets[i].add(sq(p, u))

    def greedy(order):
        pts = []
        dist_sets = []
        for p in order:
            if try_add(pts, dist_sets, p):
                commit(pts, dist_sets, p)
            if time.time() > deadline:
                break
        return pts, dist_sets

    def local_improve(pts, dist_sets):
        """Try to add more points by scanning; also try swap moves."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any point
            for p in all_points:
                if p in pts:
                    continue
                if try_add(pts, dist_sets, p):
                    commit(pts, dist_sets, p)
                    improved = True
            # Try swap: remove one, add two? Keep simple: single add pass above.
        return pts, dist_sets

    best = []
    # Multiple random orderings
    for restart in range(200):
        if time.time() > deadline:
            break
        order = list(all_points)
        rng.shuffle(order)
        pts, dist_sets = greedy(order)
        pts, dist_sets = local_improve(pts, dist_sets)
        if len(pts) > len(best):
            best = pts
        if time.time() > deadline:
            break

    # Final attempt: try to add more to best
    if time.time() < deadline:
        pts = list(best)
        dist_sets = [set() for _ in pts]
        for i, p in enumerate(pts):
            for j, q in enumerate(pts):
                if i != j:
                    dist_sets[i].add(sq(p, q))
        local_improve(pts, dist_sets)
        if len(pts) > len(best):
            best = pts

    return best
# EVOLVE-BLOCK-END
