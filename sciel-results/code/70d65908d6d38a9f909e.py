import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit

    # Initial configuration: 4x4 grid
    initial_points = [[x, y] for x in range(4) for y in range(4)]

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def evaluate(points):
        n = len(points)
        # Quick distinctness check
        seen = set()
        for p in points:
            t = (p[0], p[1])
            if t in seen:
                return -10000
            seen.add(t)

        line_counts = {}
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                c = dy * xi - dx * yi
                key = (dx, dy, c)
                line_counts[key] = line_counts.get(key, 0) + 1

        score = 0
        invalid = False
        for count in line_counts.values():
            if count == 6:  # C(4,2) = 6 pairs -> 4 points
                score += 1
            elif count >= 10:  # C(5,2) = 10 pairs -> 5+ points
                invalid = True
                break

        if invalid:
            return -1000 + score
        return score

    def find_violating_points(points):
        """Find points that are part of 5+ collinear lines."""
        n = len(points)
        line_counts = {}
        line_points = {}
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                c = dy * xi - dx * yi
                key = (dx, dy, c)
                line_counts[key] = line_counts.get(key, 0) + 1
                if key not in line_points:
                    line_points[key] = []
                line_points[key].append(i)

        violating = set()
        for key, count in line_counts.items():
            if count >= 10:  # 5+ points
                for idx in line_points[key]:
                    violating.add(idx)
        return violating

    best_points = [p[:] for p in initial_points]
    best_score = evaluate(best_points)
    current_points = [p[:] for p in initial_points]
    current_score = best_score

    T = 2.0
    T_min = 0.05
    cooling_rate = 0.9995

    def make_move(points, temperature, violating_set):
        """Generate a candidate move, biased to break violations if present."""
        idx = random.randint(0, 15)

        # If there are violations, bias toward moving violating points
        if violating_set and random.random() < 0.7:
            violating_list = list(violating_set)
            idx = random.choice(violating_list)

        old_x, old_y = points[idx]

        # Choose move scale based on temperature and violation status
        if violating_set:
            # Larger moves to break violations
            scale = random.choice([5, 10, 15, 20])
        elif temperature > 1.0:
            scale = random.choice([3, 5, 8])
        else:
            scale = random.choice([1, 2, 3, 5])

        new_x = old_x + random.randint(-scale, scale)
        new_y = old_y + random.randint(-scale, scale)

        if abs(new_x) > 1000 or abs(new_y) > 1000:
            return None

        new_point = [new_x, new_y]
        # Check distinctness
        for k in range(16):
            if k != idx and points[k] == new_point:
                return None

        return idx, new_point

    no_improve_count = 0
    max_no_improve = 500
    restart_count = 0
    max_restarts = 20

    while time.time() - start_time < time_limit:
        # Find current violations for biasing
        violating_set = set()
        if current_score < 0:  # Has violations
            violating_set = find_violating_points(current_points)

        move = make_move(current_points, T, violating_set)
        if move is None:
            continue

        idx, new_point = move
        old_point = current_points[idx][:]
        current_points[idx] = new_point
        new_score = evaluate(current_points)

        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Simulated annealing acceptance
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / max(T, 1e-10))

            if accept:
                # Keep the move
                pass
            else:
                # Revert
                current_points[idx] = old_point
                no_improve_count += 1

        T *= cooling_rate
        if T < T_min:
            T = T_min

        if no_improve_count > max_no_improve:
            # Kick: larger perturbation
            restart_count += 1
            if restart_count > max_restarts:
                # Full restart with new random configuration
                restart_count = 0
                current_points = [p[:] for p in initial_points]
                # Add random jitter to escape
                for i in range(16):
                    if random.random() < 0.3:
                        current_points[i][0] += random.randint(-10, 10)
                        current_points[i][1] += random.randint(-10, 10)
                        if abs(current_points[i][0]) > 1000:
                            current_points[i][0] = 0
                        if abs(current_points[i][1]) > 1000:
                            current_points[i][1] = 0
                # Ensure distinctness
                for i in range(16):
                    for j in range(i + 1, 16):
                        if current_points[i] == current_points[j]:
                            current_points[j][0] += random.randint(1, 5)
                            if abs(current_points[j][0]) > 1000:
                                current_points[j][0] = 0
                current_score = evaluate(current_points)
                T = 2.0
            else:
                # Partial kick: move a few points to random locations
                for _ in range(random.randint(2, 5)):
                    idx2 = random.randint(0, 15)
                    new_x = random.randint(-50, 50)
                    new_y = random.randint(-50, 50)
                    if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                        is_distinct = True
                        for k in range(16):
                            if k != idx2 and current_points[k] == [new_x, new_y]:
                                is_distinct = False
                                break
                        if is_distinct:
                            current_points[idx2] = [new_x, new_y]
                current_score = evaluate(current_points)
                T = max(T, 1.0)
            no_improve_count = 0

    return best_points
