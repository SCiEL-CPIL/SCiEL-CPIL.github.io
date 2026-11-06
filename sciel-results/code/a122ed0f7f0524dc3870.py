import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Lattice neighbors in (a,b) coords
    NDIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + da * db + db * db

    # Candidate pool (large enough for any optimal 32-point shape)
    L = 10
    candidates = [(a, b) for a in range(-L, L + 1) for b in range(-L, L + 1)]
    cand_set = set(candidates)

    # Precompute neighbor lists within candidate pool
    nbrs = {}
    for p in candidates:
        a, b = p
        lst = []
        for da, db in NDIRS:
            q = (a + da, b + db)
            if q in cand_set:
                lst.append(q)
        nbrs[p] = lst

    # --- Seeds ---
    # Hexagon radius 3 minus 5 corners -> 32 points, 75 edges
    hex_points = [(a, b) for a in range(-3, 4) for b in range(-3, 4)
                  if max(abs(a), abs(b), abs(a + b)) <= 3]
    corners = [(3, 0), (0, 3), (-3, 3), (-3, 0), (0, -3), (3, -3)]
    hex_seed = set(p for p in hex_points if p not in corners[:5])

    # Greedy construction: add point with most neighbors already in set
    def greedy_construct():
        pts = set()
        pts.add((0, 0))
        for _ in range(31):
            if time.time() > deadline:
                break
            best = None
            best_score = -1
            for q in candidates:
                if q in pts:
                    continue
                score = sum(1 for r in nbrs[q] if r in pts)
                if score > best_score:
                    best_score = score
                    best = q
            if best is not None:
                pts.add(best)
        return pts

    greedy_seed = greedy_construct()
    if len(greedy_seed) != 32:
        greedy_seed = hex_seed

    seeds = [hex_seed, greedy_seed]
    # Add a couple of random seeds for diversity
    for _ in range(3):
        rs = set(random.sample(candidates, 32))
        seeds.append(rs)

    def count_edges(pts):
        cnt = 0
        for p in pts:
            for q in nbrs[p]:
                if q in pts:
                    cnt += 1
        return cnt // 2

    best_points = set()
    best_edges = 0

    for seed in seeds:
        if time.time() > deadline:
            break
        points = set(seed)
        edges = count_edges(points)
        if edges > best_edges:
            best_edges = edges
            best_points = set(points)

        # Local search: moves (relocate to neighbor) and swaps (exchange with outside)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            point_list = list(points)
            random.shuffle(point_list)

            for p in point_list:
                if time.time() > deadline:
                    break

                # --- Try MOVE: relocate p to one of its neighbors ---
                lost = sum(1 for q in nbrs[p] if q in points)
                best_move_gain = 0
                best_move_q = None
                for q in nbrs[p]:
                    if q in points:
                        continue
                    gained = sum(1 for r in nbrs[q] if r in points)
                    gain = gained - lost
                    if gain > best_move_gain:
                        best_move_gain = gain
                        best_move_q = q
                if best_move_q is not None:
                    points.remove(p)
                    points.add(best_move_q)
                    edges += best_move_gain
                    if edges > best_edges:
                        best_edges = edges
                        best_points = set(points)
                    improved = True
                    break  # restart outer loop

                # --- Try SWAP: exchange p with an outside candidate ---
                best_swap_gain = 0
                best_swap_q = None
                for q in candidates:
                    if q in points:
                        continue
                    gained = sum(1 for r in nbrs[q] if r in points)
                    gain = gained - lost
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_q = q
                if best_swap_q is not None:
                    points.remove(p)
                    points.add(best_swap_q)
                    edges += best_swap_gain
                    if edges > best_edges:
                        best_edges = edges
                        best_points = set(points)
                    improved = True
                    break  # restart outer loop

            if best_edges == 76:
                break

        if best_edges == 76:
            break

    # Fallback: if local search somehow failed, return the 75-edge hex seed
    if best_edges < count_edges(hex_seed):
        best_points = hex_seed

    return [[a, b] for a, b in best_points]
