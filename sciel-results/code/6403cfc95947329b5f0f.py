import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def hex_dist(a, b):
        return max(abs(a), abs(b), abs(a + b))

    def point_dist_sq(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        da = a1 - a2
        db = b1 - b2
        return da*da + da*db + db*db

    # Generate all points with hex_dist <= 3 (37 points total)
    all_pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) <= 3:
                all_pts.append((a, b))

    outer_pts = [p for p in all_pts if hex_dist(p[0], p[1]) == 3]
    inner_pts = [p for p in all_pts if hex_dist(p[0], p[1]) < 3]

    # Precompute adjacency
    n_all = len(all_pts)
    adj = [[False] * n_all for _ in range(n_all)]
    pt_to_idx = {p: i for i, p in enumerate(all_pts)}
    for i in range(n_all):
        for j in range(i + 1, n_all):
            if point_dist_sq(all_pts[i], all_pts[j]) == 1:
                adj[i][j] = True
                adj[j][i] = True

    # Precompute edge counts
    inner_inner_edges = 0
    for i in range(len(inner_pts)):
        for j in range(i + 1, len(inner_pts)):
            if point_dist_sq(inner_pts[i], inner_pts[j]) == 1:
                inner_inner_edges += 1

    outer_to_inner = {}
    outer_to_outer = {}
    for p in outer_pts:
        cnt_inner = 0
        cnt_outer = 0
        for q in inner_pts:
            if point_dist_sq(p, q) == 1:
                cnt_inner += 1
        for q in outer_pts:
            if q != p and point_dist_sq(p, q) == 1:
                cnt_outer += 1
        outer_to_inner[p] = cnt_inner
        outer_to_outer[p] = cnt_outer

    best_score = 0
    best_remove = None

    # Search all combinations of 5 outer points to remove
    for remove_combo in combinations(outer_pts, 5):
        if time.time() > deadline:
            break
        remove_set = set(remove_combo)
        kept_outer = [p for p in outer_pts if p not in remove_set]

        # Count edges among kept_outer
        edges_among_kept = 0
        for i in range(len(kept_outer)):
            for j in range(i + 1, len(kept_outer)):
                if point_dist_sq(kept_outer[i], kept_outer[j]) == 1:
                    edges_among_kept += 1

        # Sum of outer-to-inner edges for kept outer
        outer_inner_sum = sum(outer_to_inner[p] for p in kept_outer)

        total = inner_inner_edges + outer_inner_sum + edges_among_kept

        if total > best_score:
            best_score = total
            best_remove = remove_combo

        # Early termination if we hit the Harborth bound
        if best_score >= 76:
            break

    if best_remove is None:
        # Fallback: just use first 32 points
        best_result = all_pts[:32]
    else:
        remove_set = set(best_remove)
        best_result = [p for p in all_pts if p not in remove_set]

    return [[a, b] for a, b in best_result]
