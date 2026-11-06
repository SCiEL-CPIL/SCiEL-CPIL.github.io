import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                if count >= 4:
                    return True
        return False

    def build_coll_mask(points):
        """Precompute collinearity masks: coll_mask[i][j] = bitmask of k collinear with i,j (k!=i,j)."""
        n = len(points)
        coll_mask = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                coll_mask[i][j] = mask
                coll_mask[j][i] = mask
        return coll_mask

    def count_triples_fast(coll_mask):
        """Count collinear triples from precomputed mask."""
        n = len(coll_mask)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                count += bin(coll_mask[i][j]).count('1')
        return count

    def exact_independence_number(points):
        """
        Exact branch-and-bound for maximum subset with no 3 collinear.
        Uses bitmask-based backtracking with candidate-set pruning.
        """
        n = len(points)
        coll_mask = build_coll_mask(points)

        # For each point p, precompute: for each a, coll_mask[p][a] gives forbidden points
        # When we add p to selected set A, any q such that exists a1,a2 in A with p,a1,a2 collinear
        # is forbidden. Equivalently: for each a in A, coll_mask[p][a] & A != 0 means conflict.

        best = [0]

        def backtrack(A, C, depth):
            if depth + bin(C).count('1') <= best[0]:
                return
            if C == 0:
                if depth > best[0]:
                    best[0] = depth
                return

            # Pick a candidate: use the one with fewest conflicts to prune early
            # Try each candidate in C, pick one that maximizes pruning
            # For speed, just pick the first set bit
            p_bit = C & -C
            p = p_bit.bit_length() - 1

            # Option 1: exclude p
            backtrack(A, C ^ p_bit, depth)

            # Option 2: include p
            # Check compatibility: for each a in A, if coll_mask[p][a] & A != 0, conflict
            compatible = True
            temp_A = A
            while temp_A:
                a_bit = temp_A & -temp_A
                a = a_bit.bit_length() - 1
                if coll_mask[p][a] & A:
                    compatible = False
                    break
                temp_A ^= a_bit

            if compatible:
                # New candidates: remove p and all q that would form triple with p and some a in A
                new_C = C ^ p_bit
                temp_A = A
                while temp_A:
                    a_bit = temp_A & -temp_A
                    a = a_bit.bit_length() - 1
                    new_C &= ~coll_mask[p][a]
                    temp_A ^= a_bit

                backtrack(A | p_bit, new_C, depth + 1)

        backtrack(0, (1 << n) - 1, 0)
        return best[0]

    def evaluate_score(points):
        return exact_independence_number(points)

    def generate_perturbed_grid(seed=0):
        """Generate a 5x4 grid with small random perturbations to break 4-collinearities."""
        random.seed(seed)
        points = []
        seen = set()
        for i in range(5):
            for j in range(4):
                di = random.choice([-1, 0, 0, 0, 1])
                dj = random.choice([-1, 0, 0, 0, 1])
                px = i + di
                py = j + dj
                if (px, py) not in seen:
                    points.append([px, py])
                    seen.add((px, py))

        # Fill if needed
        while len(points) < 20:
            px = random.randint(0, 10)
            py = random.randint(0, 10)
            if (px, py) not in seen:
                points.append([px, py])
                seen.add((px, py))

        if len(set(map(tuple, points))) < 20 or has_4_collinear(points):
            return None
        return points

    def generate_staggered_grid(seed=0):
        """Generate a staggered grid pattern."""
        random.seed(seed)
        points = []
        seen = set()
        for j in range(4):
            offset = j % 2
            for i in range(5):
                px = i + offset + random.choice([-1, 0, 0, 1])
                py = j * 2 + random.choice([-1, 0, 0, 1])
                if (px, py) not in seen:
                    points.append([px, py])
                    seen.add((px, py))

        while len(points) < 20:
            px = random.randint(-2, 10)
            py = random.randint(-2, 10)
            if (px, py) not in seen:
                points.append([px, py])
                seen.add((px, py))

        if len(set(map(tuple, points))) < 20 or has_4_collinear(points):
            return None
        return points

    def generate_line_cluster(seed=0):
        """Generate points clustered near a few lines with controlled perturbations."""
        random.seed(seed)
        points = []
        seen = set()

        # Create 5-6 lines, each with 3-4 points
        num_lines = random.randint(5, 6)
        for _ in range(num_lines):
            if random.random() < 0.4:
                # Horizontal line
                y = random.randint(-5, 5)
                x_base = random.randint(-5, 5)
                for i in range(random.randint(3, 4)):
                    x = x_base + i * random.randint(3, 5)
                    y_off = y + random.choice([-1, 0, 0, 1])
                    if (x, y_off) not in seen:
                        points.append([x, y_off])
                        seen.add((x, y_off))
            elif random.random() < 0.7:
                # Vertical line
                x = random.randint(-5, 5)
                y_base = random.randint(-5, 5)
                for i in range(random.randint(3, 4)):
                    y = y_base + i * random.randint(3, 5)
                    x_off = x + random.choice([-1, 0, 0, 1])
                    if (x_off, y) not in seen:
                        points.append([x_off, y])
                        seen.add((x_off, y))
            else:
                # Diagonal
                m = random.choice([1, -1])
                b = random.randint(-10, 10)
                x_base = random.randint(-5, 5)
                for i in range(random.randint(3, 4)):
                    x = x_base + i * random.randint(2, 4)
                    y = m * x + b + random.choice([-1, 0, 0, 1])
                    if (x, y) not in seen:
                        points.append([x, y])
                        seen.add((x, y))

        while len(points) < 20:
            px = random.randint(-10, 10)
            py = random.randint(-10, 10)
            if (px, py) not in seen:
                points.append([px, py])
                seen.add((px, py))

        points = points[:20]
        if len(set(map(tuple, points))) < 20 or has_4_collinear(points):
            return None
        return points

    def local_search(points, coll_mask, max_steps=200):
        """Local search to minimize independence number using triple count as proxy."""
        current_points = [p[:] for p in points]
        current_coll = build_coll_mask(current_points)
        current_triples = count_triples_fast(current_coll)
        current_score = evaluate_score(current_points)

        best_points = [p[:] for p in current_points]
        best_score = current_score

        for step in range(max_steps):
            if time.time() > deadline - 15:
                break

            improved = False
            for i in range(20):
                if time.time() > deadline - 15:
                    break

                orig = current_points[i][:]

                # Try nearby positions
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_p = [orig[0] + dx, orig[1] + dy]
                        if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                            continue

                        # Check duplicates
                        dup = False
                        for j in range(20):
                            if j != i and current_points[j][0] == new_p[0] and current_points[j][1] == new_p[1]:
                                dup = True
                                break
                        if dup:
                            continue

                        current_points[i] = new_p

                        if has_4_collinear(current_points):
                            current_points[i] = orig
                            continue

                        new_coll = build_coll_mask(current_points)
                        new_triples = count_triples_fast(new_coll)

                        if new_triples > current_triples:
                            current_triples = new_triples
                            current_coll = new_coll
                            current_score = evaluate_score(current_points)
                            if current_score < best_score:
                                best_score = current_score
                                best_points = [p[:] for p in current_points]
                            improved = True
                            break
                        else:
                            current_points[i] = orig

                    if improved:
                        break

            if not improved:
                break

        return best_points, best_score

    def simulated_annealing(points, max_iters=500):
        """Simulated annealing to minimize independence number."""
        current_points = [p[:] for p in points]
        current_coll = build_coll_mask(current_points)
        current_triples = count_triples_fast(current_coll)
        current_score = evaluate_score(current_points)

        best_points = [p[:] for p in current_points]
        best_score = current_score

        T = 5.0
        cooling = 0.995

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            idx = random.randint(0, 19)
            old = current_points[idx][:]

            if random.random() < 0.7:
                new_p = [old[0] + random.randint(-4, 4), old[1] + random.randint(-4, 4)]
            else:
                new_p = [random.randint(-15, 15), random.randint(-15, 15)]

            new_p[0] = max(-100, min(100, new_p[0]))
            new_p[1] = max(-100, min(100, new_p[1]))

            dup = False
            for j in range(20):
                if j != idx and current_points[j][0] == new_p[0] and current_points[j][1] == new_p[1]:
                    dup = True
                    break
            if dup:
                continue

            current_points[idx] = new_p

            if has_4_collinear(current_points):
                current_points[idx] = old
                continue

            new_coll = build_coll_mask(current_points)
            new_triples = count_triples_fast(new_coll)
            delta = new_triples - current_triples

            if delta > 0:
                current_triples = new_triples
                current_coll = new_coll
                if it % 20 == 0:
                    current_score = evaluate_score(current_points)
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
            else:
                if T > 0.1:
                    prob = min(1.0, max(0.0, 1.0 - abs(delta) / (T * 5)))
                    if random.random() < prob:
                        current_triples = new_triples
                        current_coll = new_coll
                        if it % 50 == 0:
                            current_score = evaluate_score(current_points)
                            if current_score < best_score:
                                best_score = current_score
                                best_points = [p[:] for p in current_points]
                    else:
                        current_points[idx] = old

            T *= cooling

        return best_points, best_score

    # Main algorithm
    best_points = None
    best_score = 21

    # Generate diverse initial configurations
    initial_configs = []

    # Perturbed grids
    for seed in range(20):
        if time.time() > deadline - 60:
            break
        pts = generate_perturbed_grid(seed)
        if pts:
            initial_configs.append(pts)

    # Staggered grids
    for seed in range(10):
        if time.time() > deadline - 60:
            break
        pts = generate_staggered_grid(seed)
        if pts:
            initial_configs.append(pts)

    # Line clusters
    for seed in range(10):
        if time.time() > deadline - 60:
            break
        pts = generate_line_cluster(seed)
        if pts:
            initial_configs.append(pts)

    # Evaluate initial configs
    for pts in initial_configs:
        if time.time() > deadline - 30:
            break
        score = evaluate_score(pts)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]

    if best_points is None:
        # Fallback: random valid config
        while True:
            pts = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(20)]
            if len(set(map(tuple, pts))) == 20 and not has_4_collinear(pts):
                best_points = pts
                best_score = evaluate_score(pts)
                break

    # Phase 2: Local search from best
    if time.time() < deadline - 25:
        ls_pts, ls_score = local_search(best_points)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_pts

    # Phase 3: Simulated annealing
    if time.time() < deadline - 20:
        sa_pts, sa_score = simulated_annealing(best_points, max_iters=300)
        if sa_score < best_score:
            best_score = sa_score
            best_points = sa_pts

    # Phase 4: Final local search polish
    if time.time() < deadline - 10:
        ls_pts2, ls_score2 = local_search(best_points, max_steps=50)
        if ls_score2 < best_score:
            best_score = ls_score2
            best_points = ls_pts2

    # Final validation
    final_score = evaluate_score(best_points)
    if final_score < best_score:
        best_score = final_score

    return best_points
