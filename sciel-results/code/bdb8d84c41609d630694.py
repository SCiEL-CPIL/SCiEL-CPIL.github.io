import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    BOUND = 1000

    def gcd(a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a

    def line_key(p, q):
        x1, y1 = p
        x2, y2 = q
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(dx, dy)
        if g:
            dx //= g
            dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = x1 * dy - y1 * dx
        return (dx, dy, c)

    def score_config(points):
        """Returns (exact4_count, invalid_flag)."""
        n = len(points)
        counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                k = line_key(points[i], points[j])
                counts[k] = counts.get(k, 0) + 1
        exact4 = 0
        for c in counts.values():
            if c == 6:  # 4 points -> 6 pairs
                exact4 += 1
            elif c >= 10:  # 5 points -> 10 pairs
                return (0, True)
        return (exact4, False)

    def is_valid(points):
        n = len(points)
        counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                k = line_key(points[i], points[j])
                counts[k] = counts.get(k, 0) + 1
                if counts[k] >= 10:
                    return False
        return True

    # Generate diverse seed configurations
    def seed_grid():
        return [[x, y] for x in range(4) for y in range(4)]

    def seed_staggered():
        pts = []
        for row in range(4):
            offset = row
            for col in range(4):
                pts.append([col + offset, row])
        return pts

    def seed_diagonal():
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        # Perturb some points to create diagonal 4-point lines
        pts[5] = [1, 5]
        pts[10] = [2, -1]
        pts[15] = [5, 3]
        return pts

    def seed_affine_shear():
        pts = []
        for x in range(4):
            for y in range(4):
                pts.append([x + y, y])
        return pts

    def seed_affine_shear2():
        pts = []
        for x in range(4):
            for y in range(4):
                pts.append([x, x + y])
        return pts

    def seed_perturbed_grid():
        pts = [[x, y] for x in range(4) for y in range(4)]
        pts[0] = [0, 0]
        pts[3] = [3, 3]
        pts[4] = [0, 5]
        pts[15] = [4, 3]
        return pts

    def seed_hexagonal():
        pts = [[0, 0]]
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]:
            pts.append([dx, dy])
        ring2 = [(2,0),(-2,0),(0,2),(0,-2),(2,1),(-2,-1),(1,2),(-1,-2)]
        for p in ring2:
            pts.append(list(p))
        return pts[:16]

    seeds = [
        seed_grid(),
        seed_staggered(),
        seed_diagonal(),
        seed_affine_shear(),
        seed_affine_shear2(),
        seed_perturbed_grid(),
        seed_hexagonal(),
    ]

    best_points = None
    best_score = -1

    rng = random.Random(42)

    for seed_idx, seed_pts in enumerate(seeds):
        if time.time() >= deadline:
            break

        # Ensure distinctness and bounds
        valid_seed = True
        seen = set()
        for p in seed_pts:
            if abs(p[0]) > BOUND or abs(p[1]) > BOUND:
                valid_seed = False
                break
            key = (p[0], p[1])
            if key in seen:
                valid_seed = False
                break
            seen.add(key)
        if not valid_seed:
            continue

        cur = [list(p) for p in seed_pts]
        cur_score, invalid = score_config(cur)
        if invalid:
            continue

        # Phase 1: Local search (hill climbing)
        local_deadline = min(deadline, start_time + 15.0)
        improved = True
        while improved and time.time() < local_deadline:
            improved = False
            for i in range(16):
                if time.time() >= local_deadline:
                    break
                x0, y0 = cur[i]
                best_local_move = None
                best_local_score = cur_score

                # Try small moves
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x0 + dx, y0 + dy
                        if abs(nx) > BOUND or abs(ny) > BOUND:
                            continue
                        if (nx, ny) in [(p[0], p[1]) for j, p in enumerate(cur) if j != i]:
                            continue
                        cand = [list(p) for p in cur]
                        cand[i] = [nx, ny]
                        s, inv = score_config(cand)
                        if inv:
                            continue
                        if s > best_local_score:
                            best_local_score = s
                            best_local_move = [nx, ny]

                if best_local_move is not None:
                    cur[i] = best_local_move
                    cur_score = best_local_score
                    improved = True

        if cur_score > best_score:
            best_score = cur_score
            best_points = [list(p) for p in cur]

        # Phase 2: Simulated annealing for escape
        if time.time() < deadline:
            T = 5.0
            T_min = 0.1
            cooling = 0.995
            sa_deadline = min(deadline, start_time + 35.0)

            while time.time() < sa_deadline:
                idx = rng.randint(0, 15)
                old_p = cur[idx][:]

                if rng.random() < 0.7:
                    new_x = old_p[0] + rng.randint(-4, 4)
                    new_y = old_p[1] + rng.randint(-4, 4)
                else:
                    new_x = old_p[0] + rng.randint(-15, 15)
                    new_y = old_p[1] + rng.randint(-15, 15)

                if abs(new_x) > BOUND or abs(new_y) > BOUND:
                    continue
                if (new_x, new_y) in [(p[0], p[1]) for j, p in enumerate(cur) if j != idx]:
                    continue

                cur[idx] = [new_x, new_y]
                new_score, invalid = score_config(cur)
                if invalid:
                    cur[idx] = old_p
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    continue

                delta = new_score - cur_score
                if delta >= 0 or rng.random() < math.exp(delta / max(T, 0.01)):
                    cur_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [list(p) for p in cur]
                else:
                    cur[idx] = old_p

                T *= cooling
                if T < T_min:
                    T = T_min

            # After SA, do a final local polish
            if time.time() < deadline:
                polish_deadline = min(deadline, start_time + 40.0)
                improved = True
                while improved and time.time() < polish_deadline:
                    improved = False
                    for i in range(16):
                        if time.time() >= polish_deadline:
                            break
                        x0, y0 = cur[i]
                        best_local_move = None
                        best_local_score = cur_score
                        for dx in range(-3, 4):
                            for dy in range(-3, 4):
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = x0 + dx, y0 + dy
                                if abs(nx) > BOUND or abs(ny) > BOUND:
                                    continue
                                if (nx, ny) in [(p[0], p[1]) for j, p in enumerate(cur) if j != i]:
                                    continue
                                cand = [list(p) for p in cur]
                                cand[i] = [nx, ny]
                                s, inv = score_config(cand)
                                if inv:
                                    continue
                                if s > best_local_score:
                                    best_local_score = s
                                    best_local_move = [nx, ny]
                        if best_local_move is not None:
                            cur[i] = best_local_move
                            cur_score = best_local_score
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_points = [list(p) for p in cur]

    if best_points is None:
        # Fallback: return the plain grid
        best_points = seed_grid()

    return best_points
