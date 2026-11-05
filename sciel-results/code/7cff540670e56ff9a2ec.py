import time
import random
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 30
    RANGE = 30

    def gen_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for i in range(w):
            for j in range(h):
                pts.append((i + offset_x, j + offset_y))
        return pts

    inits = []
    inits.append(gen_grid(6, 5, -2, -2))
    inits.append(gen_grid(5, 6, -2, -2))
    inits.append(gen_grid(10, 3, -4, -1))
    inits.append(gen_grid(3, 10, -1, -4))

    random.seed(42)
    for _ in range(3):
        pts = []
        while len(pts) < N:
            p = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
            if p not in pts:
                pts.append(p)
        inits.append(pts)

    best_score = 0
    best_points = inits[0]

    target_areas = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]

    def get_max_multiplicity(points):
        n = len(points)
        counts = Counter()
        for i in range(n-2):
            x1, y1 = points[i]
            for j in range(i+1, n-1):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    d = dx1 * (y3 - y1) - dy1 * (x3 - x1)
                    if d != 0:
                        counts[abs(d)] += 1
        if not counts:
            return 0
        return max(counts.values())

    def compute_initial_count(points, A):
        n = len(points)
        count = 0
        for i in range(n-2):
            x1, y1 = points[i]
            for j in range(i+1, n-1):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    d = dx1 * (y3 - y1) - dy1 * (x3 - x1)
                    if abs(d) == A:
                        count += 1
        return count

    def hypergraph_prune(points, target_A, min_density=0.3):
        """
        Global combinatorial pruning: for target area A, find which points
        participate in the densest cluster of triangles with area A.
        Remove points that don't contribute to high-density clusters.
        Returns pruned list of (index, new_position) pairs.
        """
        n = len(points)
        # Count how many area-A triangles each point participates in
        point_counts = np.zeros(n, dtype=int)

        for i in range(n - 2):
            x1, y1 = points[i]
            for j in range(i + 1, n - 1):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    d = dx1 * (y3 - y1) - dy1 * (x3 - x1)
                    if abs(d) == target_A:
                        point_counts[i] += 1
                        point_counts[j] += 1
                        point_counts[k] += 1

        # Find the threshold: keep points with count >= min_density * max_count
        max_count = point_counts.max() if n > 0 else 0
        if max_count == 0:
            return list(range(n))

        threshold = max(1, int(min_density * max_count))
        # Keep points that meet the threshold
        kept = [i for i in range(n) if point_counts[i] >= threshold]

        # If we'd keep too few, relax
        if len(kept) < N:
            # Sort by count descending, take top N
            sorted_indices = sorted(range(n), key=lambda i: -point_counts[i])
            kept = sorted_indices[:N]

        return kept

    def vectorized_delta(points, idx, old_pt, new_pt, A):
        """
        Compute delta in count of area-A triangles when replacing point idx.
        Uses vectorized numpy for speed.
        """
        n = len(points)
        # Get all other points as array
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        others = np.array([points[i] for i in range(n) if i != idx], dtype=np.int64)

        old_x, old_y = old_pt
        new_x, new_y = new_pt

        # For each pair of others, compute area with old_pt and new_pt
        # Number of others = n-1
        m = n - 1
        # Compute all pair areas with old point
        # Area = |(x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)| for points 0,1,2
        # With old_pt as reference:
        ox = others[:, 0] - old_x
        oy = others[:, 1] - old_y

        # For each pair (i,j) from others, area with old_pt
        # d = ox[i]*oy[j] - oy[i]*ox[j]
        # Vectorize: create all pairs
        # This is O(m^2) which for m=29 is 841 pairs - fine

        old_areas = np.zeros((m, m), dtype=np.int64)
        new_areas = np.zeros((m, m), dtype=np.int64)

        nx = others[:, 0] - new_x
        ny = others[:, 1] - new_y

        # d[i,j] = ox[i]*oy[j] - oy[i]*ox[j]
        # This is the cross product of vector i and vector j from old_pt
        old_areas = ox[:, None] * oy[None, :] - oy[:, None] * ox[None, :]
        new_areas = nx[:, None] * ny[None, :] - ny[:, None] * nx[None, :]

        # Count triangles with area A (each triangle counted twice, once as (i,j) and (j,i))
        # But we only count i<j pairs, so count upper triangle
        old_count = np.sum(np.abs(old_areas) == A) // 2
        new_count = np.sum(np.abs(new_areas) == A) // 2

        return new_count - old_count

    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 2.0:
            break

        current_points = list(init_pts)
        if len(set(current_points)) != N:
            continue
        valid = True
        for p in current_points:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                valid = False
                break
        if not valid:
            continue

        for A in target_areas:
            if time.time() > deadline - 2.0:
                break

            # Phase 1: Hypergraph pruning - identify good starting configuration
            # Try to prune to find a dense cluster
            pruned_indices = hypergraph_prune(current_points, A, min_density=0.2)

            # SA to maximize count of area A
            T = 5.0
            min_T = 0.01
            cooling = 0.999
            current_score = compute_initial_count(current_points, A)
            best_local_score = current_score
            best_local_points = list(current_points)

            no_improve = 0
            max_no_improve = 8000

            while time.time() < deadline - 2.0 and no_improve < max_no_improve:
                idx = random.randint(0, N - 1)
                old_pt = current_points[idx]

                if random.random() < 0.8:
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                    if dx == 0 and dy == 0:
                        dx = 1
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if not (-RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE):
                        continue
                    if new_pt in current_points:
                        continue
                else:
                    new_pt = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
                    if new_pt in current_points:
                        continue

                # Use vectorized delta computation
                delta = vectorized_delta(current_points, idx, old_pt, new_pt, A)

                new_score = current_score + delta

                if delta > 0 or (T > min_T and random.random() < np.exp(delta / T)):
                    current_points[idx] = new_pt
                    current_score = new_score
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_local_points = list(current_points)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

                T *= cooling
                if T < min_T:
                    T = min_T

            # Phase 2: After SA, do a final hypergraph-guided refinement
            # Try to further concentrate points around the densest area-A cluster
            if time.time() < deadline - 3.0:
                pruned = hypergraph_prune(best_local_points, A, min_density=0.4)
                # For points not in pruned set, try moving them to positions
                # that increase area-A count
                for idx in range(N):
                    if time.time() > deadline - 2.0:
                        break
                    if idx not in pruned:
                        old_pt = best_local_points[idx]
                        # Try a few candidate positions
                        for _ in range(5):
                            if random.random() < 0.7:
                                dx = random.randint(-3, 3)
                                dy = random.randint(-3, 3)
                                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                            else:
                                new_pt = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
                            if not (-RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE):
                                continue
                            if new_pt in best_local_points:
                                continue

                            delta = vectorized_delta(best_local_points, idx, old_pt, new_pt, A)
                            if delta > 0:
                                best_local_points[idx] = new_pt
                                break

            final_score = get_max_multiplicity(best_local_points)

            if final_score > best_score:
                best_score = final_score
                best_points = list(best_local_points)

    # Final validation
    if len(best_points) != N or len(set(best_points)) != N:
        return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)

    return best_points
