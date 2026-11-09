import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    def elapsed():
        return time.time() - t0

    def full_score(pts):
        """Compute max multiplicity of any positive doubled area. Returns (score, area_counts)."""
        n = len(pts)
        if n < 3:
            return 0, {}
        P = np.array(pts, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)
        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu, cv = dx[u], dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)
        if not area_counts:
            return 0, area_counts
        return max(area_counts.values()), area_counts

    def incremental_score(pts, area_counts, moved_idx):
        """
        After moving point moved_idx, update area_counts incrementally.
        Only areas involving moved_idx change.
        Returns (new_max_score, updated_area_counts).
        """
        n = len(pts)
        # Remove old contributions involving moved_idx
        # A triple (i,j,k) involves moved_idx if moved_idx is one of them
        # The doubled area = |cross(P[j]-P[i], P[k]-P[i])| for i<j<k
        # We need to figure out which pairs involve moved_idx
        # Simpler: recompute only the areas that involve the moved point
        # Areas involving point p involve pairs (p, a) and (p, b) with a < b or a < p < b etc.
        # Actually, the area of triangle (i,j,k) = |det(P[j]-P[i], P[k]-P[i])|
        # If moved_idx = i: area = |det(P[j]-P[i], P[k]-P[i])| for all j,k > i
        # If moved_idx = j: area = |det(P[j]-P[i], P[k]-P[i])| for all i < j, k > j
        # If moved_idx = k: area = |det(P[j]-P[i], P[k]-P[i])| for all i < k, j < k
        
        # For each other pair of points (a,b) where a != moved_idx and b != moved_idx,
        # the triangle (moved_idx, a, b) has an area.
        # Let's just recompute all areas involving moved_idx.
        
        mi = moved_idx
        # Find all triples involving mi
        old_areas = []
        for a in range(n):
            if a == mi:
                continue
            for b in range(a+1, n):
                if b == mi:
                    continue
                # triple is (min(mi,a,b), mid, max)
                # area = |det(P[mid]-P[min], P[max]-P[min])|
                idxs = sorted([mi, a, b])
                i, j, k = idxs
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area > 0:
                    old_areas.append(area)
        
        # Subtract old areas from counts
        for area in old_areas:
            if area in area_counts:
                area_counts[area] -= 1
                if area_counts[area] <= 0:
                    del area_counts[area]
        
        # Add new areas
        new_areas = []
        for a in range(n):
            if a == mi:
                continue
            for b in range(a+1, n):
                if b == mi:
                    continue
                idxs = sorted([mi, a, b])
                i, j, k = idxs
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area > 0:
                    new_areas.append(area)
                    area_counts[area] = area_counts.get(area, 0) + 1
        
        if not area_counts:
            return 0, area_counts
        return max(area_counts.values()), area_counts

    # 1. Initialize with strong structured candidates
    cand1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    cand2 = [(x, y) for x in range(6) for y in range(5)]
    cand3 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    random.seed(42)
    cand4 = []
    used = set()
    while len(cand4) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            cand4.append((x, y))
    cand5 = [(x, 0) for x in range(8)] + [(x, 2) for x in range(8)] + \
            [(x, 4) for x in range(7)] + [(x, 6) for x in range(7)]
    cand6 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + \
            [(x, -1) for x in range(5)] + [(x, -2) for x in range(5)]
    cand7 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + \
            [(x, 3) for x in range(5)] + [(x, 4) for x in range(5)]
    cand8 = [(x, 0) for x in range(12)] + [(x, 1) for x in range(12)] + \
            [(x, 2) for x in range(6)]

    candidates = [cand1, cand2, cand3, cand4, cand5, cand6, cand7, cand8]

    best_points = None
    best_score = 0
    best_area_counts = {}

    for cand in candidates:
        s, ac = full_score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            best_area_counts = dict(ac)

    # 2. Local search with incremental scoring
    current_points = list(best_points)
    current_score = best_score
    current_area_counts = dict(best_area_counts)

    random.seed(123)
    step_size = 3
    no_improve_count = 0
    iter_count = 0

    # Precompute for speed: for each point index, list of (other1, other2) pairs
    # where the triple involves this point
    n = 30
    triples_involving = [[] for _ in range(n)]
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for p in (a, b, c):
                    triples_involving[p].append((a, b, c))

    while elapsed() < DEADLINE:
        iter_count += 1

        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]

        r = random.random()
        if r < 0.015:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.10:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = old_x + random.randint(-step_size, step_size)
            new_y = old_y + random.randint(-step_size, step_size)

        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue

        new_pt = (new_x, new_y)
        if new_pt == (old_x, old_y):
            continue

        is_valid = True
        for j in range(30):
            if j == idx:
                continue
            if current_points[j] == new_pt:
                is_valid = False
                break

        if not is_valid:
            continue

        # Try the move with incremental scoring
        current_points[idx] = new_pt
        new_score, current_area_counts = incremental_score(current_points, current_area_counts, idx)

        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                best_area_counts = dict(current_area_counts)
                if step_size > 1 and random.random() < 0.25:
                    step_size = max(1, step_size - 1)
        else:
            # Revert
            current_points[idx] = (old_x, old_y)
            # Recompute area counts after revert
            current_score, current_area_counts = incremental_score(current_points, current_area_counts, idx)
            no_improve_count += 1

        if no_improve_count > 3000:
            current_points = list(best_points)
            current_score = best_score
            current_area_counts = dict(best_area_counts)
            no_improve_count = 0
            step_size = random.randint(4, 10)

        if random.random() < 0.002:
            step_size = random.randint(1, 6)

    return best_points
