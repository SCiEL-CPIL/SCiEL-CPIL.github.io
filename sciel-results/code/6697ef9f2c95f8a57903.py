import random
import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin before the 120s kill

    # ---- collinearity helpers ----
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 4:
                    return True
        return False

    def collinear_triples(points):
        """All triples of indices that are collinear (sorted i<j<k)."""
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    # ---- exact score via MILP (independence number of the 3-uniform hypergraph) ----
    def calculate_score(points):
        if has_4_collinear(points):
            return None
        triples = collinear_triples(points)
        if not triples:
            return 20
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            A = [[0] * 20 for _ in triples]
            for r, (i, j, k) in enumerate(triples):
                A[r][i] = 1
                A[r][j] = 1
                A[r][k] = 1
            c = [-1.0] * 20
            cons = LinearConstraint(A, ub=2, lb=0)
            bounds = Bounds(0, 1)
            res = milp(c=c, constraints=cons, integrality=[1] * 20, bounds=bounds)
            if res.status != 0:
                return None
            return int(round(-res.fun))
        except Exception:
            return None

    # ---- configuration generators ----
    def dedupe_fill(pts):
        seen = set()
        out = []
        for p in pts:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                out.append([p[0], p[1]])
        while len(out) < 20:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in seen:
                seen.add((x, y))
                out.append([x, y])
        return out[:20]

    def gen_grid():
        pts = []
        for i in range(4):
            for j in range(5):
                pts.append([-10 + i * 5, -10 + j * 5])
        return dedupe_fill(pts)

    def gen_lines():
        pts = []
        pts.extend([[-10, 0], [0, 0], [10, 0]])
        pts.extend([[-9, 1], [1, 1], [11, 1]])
        pts.extend([[-8, -1], [2, -1], [12, -1]])
        pts.extend([[0, -5], [0, 5], [0, 15]])
        pts.extend([[1, -4], [1, 6], [1, 16]])
        pts.extend([[-1, 2], [-2, -2], [3, 3], [4, -3], [5, 4]])
        return dedupe_fill(pts)

    def gen_random():
        return dedupe_fill([[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)])

    def random_perturb(points, radius=3):
        new_pts = []
        for p in points:
            dx = random.randint(-radius, radius)
            dy = random.randint(-radius, radius)
            nx = p[0] + dx
            ny = p[1] + dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                new_pts.append([nx, ny])
            else:
                new_pts.append([p[0], p[1]])
        return dedupe_fill(new_pts)

    # ---- main search ----
    best_score = 20
    best_points = None

    # Evaluate a set of initial configurations and keep the best valid one.
    candidates = []
    candidates.append(gen_grid())
    candidates.append(gen_lines())
    for _ in range(8):
        candidates.append(gen_random())

    for config in candidates:
        if len(set((p[0], p[1]) for p in config)) != 20:
            continue
        if has_4_collinear(config):
            continue
        score = calculate_score(config)
        if score is not None and score < best_score:
            best_score = score
            best_points = [p[:] for p in config]

    if best_points is None:
        for _ in range(500):
            pts = gen_random()
            if not has_4_collinear(pts):
                best_points = pts
                best_score = calculate_score(pts)
                break

    current_points = best_points
    current_score = best_score
    if current_score is None:
        current_score = best_score

    # ---- local search refinement ----
    while time.time() - start_time < time_limit:
        improved = False

        # 1) single-point moves (small neighborhood)
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
            original = current_points[i]
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx = original[0] + dx
                    ny = original[1] + dy
                    if nx < -100 or nx > 100 or ny < -100 or ny > 100:
                        continue
                    new_points = current_points.copy()
                    new_points[i] = [nx, ny]
                    if len(set((p[0], p[1]) for p in new_points)) < 20:
                        continue
                    if has_4_collinear(new_points):
                        continue
                    score = calculate_score(new_points)
                    if score is not None:
                        if score < current_score:
                            current_points = new_points
                            current_score = score
                            if current_score < best_score:
                                best_score = current_score
                                best_points = current_points
                            improved = True
                            break
                        elif score == current_score and random.random() < 0.05:
                            current_points = new_points
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

        # 2) swap-style random relocation of one point
        if not improved and time.time() - start_time < time_limit:
            for i in range(20):
                if time.time() - start_time > time_limit:
                    break
                for _ in range(4):
                    j = random.randint(0, 19)
                    nx = random.randint(current_points[j][0] - 5, current_points[j][0] + 5)
                    ny = random.randint(current_points[j][1] - 5, current_points[j][1] + 5)
                    if nx < -100 or nx > 100 or ny < -100 or ny > 100:
                        continue
                    new_points = current_points.copy()
                    new_points[i] = [nx, ny]
                    if len(set((p[0], p[1]) for p in new_points)) < 20:
                        continue
                    if has_4_collinear(new_points):
                        continue
                    score = calculate_score(new_points)
                    if score is not None:
                        if score < current_score:
                            current_points = new_points
                            current_score = score
                            if current_score < best_score:
                                best_score = current_score
                                best_points = current_points
                            improved = True
                            break
                        elif score == current_score and random.random() < 0.05:
                            current_points = new_points
                            improved = True
                            break
                if improved:
                    break

        # 3) larger random restart / perturbation when stuck
        if not improved:
            new_points = current_points.copy()
            for _ in range(4):
                idx = random.randint(0, 19)
                dx = random.randint(-8, 8)
                dy = random.randint(-8, 8)
                nx = new_points[idx][0] + dx
                ny = new_points[idx][1] + dy
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_points[idx] = [nx, ny]
            if len(set((p[0], p[1]) for p in new_points)) == 20 and not has_4_collinear(new_points):
                score = calculate_score(new_points)
                if score is not None:
                    if score < current_score:
                        current_points = new_points
                        current_score = score
                        if current_score < best_score:
                            best_score = current_score
                            best_points = current_points
                    elif score == current_score and random.random() < 0.1:
                        current_points = new_points

    return best_points if best_points else current_points

# EVOLVE-BLOCK-END
