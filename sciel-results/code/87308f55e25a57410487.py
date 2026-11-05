import time
import random
import math
from collections import Counter

def construct():
    deadline = time.time() + 45.0
    random.seed(42)
    N = 30

    # Precompute all triplets
    triplets = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                triplets.append((i, j, k))

    # Precompute for each index which triplets it participates in
    # This allows faster delta computation
    idx_triplets = [[] for _ in range(N)]
    for t_idx, (i, j, k) in enumerate(triplets):
        idx_triplets[i].append(t_idx)
        idx_triplets[j].append(t_idx)
        idx_triplets[k].append(t_idx)

    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

    def full_count(points):
        areas = []
        for i, j, k in triplets:
            a = doubled_area(points[i], points[j], points[k])
            if a > 0:
                areas.append(a)
        if not areas:
            return Counter(), 0, 0
        c = Counter(areas)
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count

    def count_target(points, target):
        count = 0
        for i, j, k in triplets:
            if doubled_area(points[i], points[j], points[k]) == target:
                count += 1
        return count

    def generate_hex_config():
        """Generate a hexagonal lattice subset of 30 points."""
        points = []
        # Use axial coordinates for hex grid
        # Convert to cartesian: x = q + r/2, y = r*sqrt(3)/2
        # But we need integer coords, so use a rotated/scaled version
        # Simple approach: use a rectangular grid with hexagonal-like spacing
        # Actually, let's use a proper hexagonal grid mapping to integers
        # For hex grid: x = 2*q + r, y = r (this gives integer coords)
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = 2*q + r
                y = r
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
        # Shuffle and take 30
        random.shuffle(points)
        return points[:30]

    def generate_triangular_config():
        """Generate a triangular lattice subset."""
        points = []
        for x in range(-4, 5):
            for y in range(-4, 5):
                # Triangular lattice: offset rows
                if (x + y) % 2 == 0:
                    points.append((x, y))
        random.shuffle(points)
        return points[:30]

    def generate_grid_config():
        """Generate a 6x5 grid."""
        return [(x, y) for x in range(-2, 4) for y in range(-2, 3)]

    def generate_dense_config():
        """Generate a dense 5x6 grid centered at origin."""
        return [(x, y) for x in range(-2, 3) for y in range(-3, 3)]

    # Try multiple initial configurations
    initial_configs = [
        generate_grid_config(),
        generate_hex_config(),
        generate_triangular_config(),
        generate_dense_config(),
        # Another variant
        [(x, y) for x in range(-3, 3) for y in range(-3, 3) if abs(x) + abs(y) <= 4][:30],
    ]

    best_initial = None
    best_initial_count = 0

    for config in initial_configs:
        seen = set()
        filtered = []
        for p in config:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                filtered.append(p)
        while len(filtered) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                filtered.append(p)
        filtered = filtered[:30]

        counter, ba, bc = full_count(filtered)
        if bc > best_initial_count:
            best_initial_count = bc
            best_initial = filtered

    coords = list(best_initial)
    counter, target, current_count = full_count(coords)
    best_overall = current_count
    best_points = list(coords)
    best_target = target

    # Simulated Annealing with target-focused optimization
    T = 10.0
    cooling_rate = 0.99

    no_improve = 0
    max_no_improve = 150
    restarts = 0
    max_restarts = 8

    while time.time() < deadline:
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break

            # Re-evaluate current state
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba

            # Switch to the most common area in current configuration
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 50
            T = 10.0

            # Random restart with perturbation
            for _ in range(3):
                idx = random.randint(0, N-1)
                new_pt = (random.randint(-30, 30), random.randint(-30, 30))
                if new_pt not in coords:
                    coords[idx] = new_pt
                    current_count = count_target(coords, target)
                    break

            if current_count < best_overall:
                coords = list(best_points)
                target = best_target
                current_count = best_overall
            continue

        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]

        # Generate candidates
        candidates = []
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                    candidates.append(new_pt)
        for _ in range(5):
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt != old_pt and new_pt not in coords:
                candidates.append(new_pt)

        if not candidates:
            no_improve += 1
            T *= cooling_rate
            continue

        best_delta = -999
        best_new_pt = None

        # Precompute old matches for this idx
        old_matches = 0
        for t_idx in idx_triplets[idx]:
            i, j, k = triplets[t_idx]
            if doubled_area(coords[i], coords[j], coords[k]) == target:
                old_matches += 1

        for new_pt in candidates:
            new_matches = 0
            for t_idx in idx_triplets[idx]:
                i, j, k = triplets[t_idx]
                if i == idx:
                    p1, p2, p3 = new_pt, coords[j], coords[k]
                elif j == idx:
                    p1, p2, p3 = coords[i], new_pt, coords[k]
                else:
                    p1, p2, p3 = coords[i], coords[j], new_pt
                if doubled_area(p1, p2, p3) == target:
                    new_matches += 1

            delta = new_matches - old_matches
            if delta > best_delta:
                best_delta = delta
                best_new_pt = new_pt

        if best_new_pt is not None:
            if best_delta > 0:
                accept = True
            else:
                if T > 0.1:
                    accept_prob = math.exp(best_delta / T)
                    accept = random.random() < accept_prob
                else:
                    accept = False

            if accept:
                coords[idx] = best_new_pt
                current_count += best_delta
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_points = list(coords)
            else:
                no_improve += 1

        T *= cooling_rate

    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x, y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x, y) for x in range(6) for y in range(5)]

    return best_points
