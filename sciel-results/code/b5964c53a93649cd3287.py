"""Improved program for the no-isosceles set task."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 310  # Keep 20s margin for 330s limit

    num_points = N * N

    def idx_to_point(idx):
        return (idx // N, idx % N)

    # Precompute squared distances using numpy for speed
    try:
        import numpy as np
        coords = np.arange(num_points)
        xs = (coords // N).astype(np.int32)
        ys = (coords % N).astype(np.int32)
        # Compute distance matrix
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_matrix = (dx * dx + dy * dy).astype(np.int32)
    except ImportError:
        dist_matrix = [[0] * num_points for _ in range(num_points)]
        for i in range(num_points):
            x1, y1 = i // N, i % N
            for j in range(i + 1, num_points):
                x2, y2 = j // N, j % N
                d = (x1 - x2)**2 + (y1 - y2)**2
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d

    # For each point, precompute the list of (distance, point_idx) pairs
    # This helps with the can_add check
    # dist_from[p] = list of (d, q) for all q != p, sorted by d
    # Actually, we need: for each p, a dict from distance to bitmask/set of points at that distance
    # Let's precompute: dist_groups[p] = dict mapping d -> list of points at distance d from p
    # This is useful for checking if p is apex of an isosceles triangle with two points in S

    # Alternative approach: use the bitset method but with numpy for distance lookups
    # For can_add(p):
    # 1. Check if any two points in S are equidistant from p (p is apex)
    #    -> compute all d(p,s) for s in S, check for duplicates
    # 2. For each s in S, check if d(p,s) equals d(s,t) for some t in S (s is apex, p and t are base)
    #    -> for each s, we need to know if d(p,s) is in the set of distances from s to other points in S

    # Let's use a hybrid approach:
    # - Maintain S as a list of indices
    # - For each point in S, maintain a set of distances to other points in S
    # - For can_add(p):
    #   a. Compute d(p,s) for all s in S, check for duplicates (p as apex)
    #   b. For each s in S, check if d(p,s) is in dist_set[s] (s as apex)

    # To speed up, we can precompute for each pair (p, s) the distance
    # and use numpy for batch distance computation

    def can_add(S_list, dist_sets, p_idx):
        """
        S_list: list of point indices in S
        dist_sets: dict mapping point index to set of distances to other points in S
        p_idx: candidate point
        Returns True if p_idx can be added without creating an isosceles triangle.
        """
        k = len(S_list)
        if k < 2:
            return True

        # Case 1: p is apex. Check for duplicate distances from p to points in S.
        seen_dists = set()
        dists_from_p = []
        for s_idx in S_list:
            d = dist_matrix[p_idx][s_idx]
            if d in seen_dists:
                return False
            seen_dists.add(d)
            dists_from_p.append((s_idx, d))

        # Cases 2 & 3: For each s in S, check if d(p,s) is in dist_sets[s]
        for s_idx, d_ps in dists_from_p:
            if d_ps in dist_sets[s_idx]:
                return False

        return True

    def add_point(S_list, dist_sets, p_idx):
        """Add p_idx to S_list and update dist_sets."""
        S_list.append(p_idx)
        new_dist_set = set()
        for s_idx in S_list:
            if s_idx != p_idx:
                d = dist_matrix[p_idx][s_idx]
                new_dist_set.add(d)
                dist_sets[s_idx].add(d)
        dist_sets[p_idx] = new_dist_set

    def remove_point(S_list, dist_sets, p_idx):
        """Remove p_idx from S_list and update dist_sets."""
        for s_idx in S_list:
            if s_idx != p_idx:
                d = dist_matrix[p_idx][s_idx]
                dist_sets[s_idx].discard(d)
        del dist_sets[p_idx]
        S_list.remove(p_idx)

    def greedy_construct(order_indices):
        """Greedy construction given an order of point indices to try."""
        S_list = []
        dist_sets = {}
        S_set = set()
        for p_idx in order_indices:
            if time.time() > deadline - 30:
                break
            if p_idx in S_set:
                continue
            if can_add(S_list, dist_sets, p_idx):
                S_set.add(p_idx)
                add_point(S_list, dist_sets, p_idx)
        return S_list, dist_sets, S_set

    def generate_diagonal_order():
        """Generate points in diagonal sweep order."""
        order = []
        for s in range(2 * N - 1):
            for x in range(max(0, s - N + 1), min(N, s + 1)):
                y = s - x
                order.append(x * N + y)
        return order

    def generate_spiral_order():
        """Generate points in spiral order from center."""
        cx, cy = N // 2, N // 2
        order = [cx * N + cy]
        visited = set([cx * N + cy])
        dx, dy = 1, 0
        steps = 1
        count = 1
        while count < num_points:
            for _ in range(2):
                for _ in range(steps):
                    cx += dx
                    cy += dy
                    if 0 <= cx < N and 0 <= cy < N:
                        idx = cx * N + cy
                        if idx not in visited:
                            visited.add(idx)
                            order.append(idx)
                            count += 1
                dx, dy = -dy, dx
            steps += 1
        for i in range(num_points):
            if i not in visited:
                order.append(i)
        return order

    def generate_random_order(seed):
        """Generate a random order with given seed."""
        rng = random.Random(seed)
        order = list(range(num_points))
        rng.shuffle(order)
        return order

    # New strategy: structured construction using modular arithmetic
    # Points where x + y ≡ c (mod m) for small m might work
    def generate_modular_order(m):
        """Generate points grouped by (x+y) mod m."""
        order = []
        for c in range(m):
            for x in range(N):
                for y in range(N):
                    if (x + y) % m == c:
                        order.append(x * N + y)
        return order

    def generate_row_order():
        """Generate points in row-major order."""
        return list(range(num_points))

    def generate_col_order():
        """Generate points in column-major order."""
        order = []
        for y in range(N):
            for x in range(N):
                order.append(x * N + y)
        return order

    # Multiple greedy restarts with different orderings
    best_S_list = []
    best_dist_sets = {}
    best_S_set = set()
    best_size = 0

    strategies = [
        generate_diagonal_order(),
        generate_spiral_order(),
        generate_row_order(),
        generate_col_order(),
        generate_modular_order(2),
        generate_modular_order(3),
        generate_modular_order(5),
    ]

    num_random_restarts = 40
    for i in range(num_random_restarts):
        strategies.append(generate_random_order(i))

    for order_indices in strategies:
        if time.time() > deadline - 30:
            break

        S_list, dist_sets, S_set = greedy_construct(order_indices)

        if len(S_list) > best_size:
            best_size = len(S_list)
            best_S_list = S_list.copy()
            best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
            best_S_set = S_set.copy()

    # Local search on the best solution
    S_list = best_S_list
    dist_sets = best_dist_sets
    S_set = best_S_set
    no_improve_count = 0
    max_no_improve = 30
    rng = random.Random(42)

    while time.time() < deadline - 10:
        improved = False

        # Try to add a random point
        for _ in range(300):
            if time.time() > deadline - 10:
                break
            p_idx = rng.randint(0, num_points - 1)
            if p_idx not in S_set:
                if can_add(S_list, dist_sets, p_idx):
                    S_set.add(p_idx)
                    add_point(S_list, dist_sets, p_idx)
                    improved = True
                    no_improve_count = 0
                    break

        # Try to swap: remove one point, add another
        if not improved and len(S_list) > 0:
            for _ in range(300):
                if time.time() > deadline - 10:
                    break
                remove_p = rng.choice(S_list)
                add_p = rng.randint(0, num_points - 1)
                if add_p in S_set or add_p == remove_p:
                    continue
                remove_point(S_list, dist_sets, remove_p)
                S_set.discard(remove_p)
                if can_add(S_list, dist_sets, add_p):
                    S_set.add(add_p)
                    add_point(S_list, dist_sets, add_p)
                    improved = True
                    no_improve_count = 0
                else:
                    S_set.add(remove_p)
                    add_point(S_list, dist_sets, remove_p)
                break

        # Try double-swap
        if not improved and len(S_list) > 1:
            for _ in range(150):
                if time.time() > deadline - 10:
                    break
                remove_p1 = rng.choice(S_list)
                remaining = [x for x in S_list if x != remove_p1]
                if not remaining:
                    continue
                remove_p2 = rng.choice(remaining)
                add_p1 = rng.randint(0, num_points - 1)
                add_p2 = rng.randint(0, num_points - 1)
                if add_p1 in S_set or add_p2 in S_set or add_p1 == add_p2:
                    continue
                remove_point(S_list, dist_sets, remove_p1)
                S_set.discard(remove_p1)
                remove_point(S_list, dist_sets, remove_p2)
                S_set.discard(remove_p2)
                if can_add(S_list, dist_sets, add_p1) and can_add(S_list, dist_sets, add_p2):
                    S_set.add(add_p1)
                    add_point(S_list, dist_sets, add_p1)
                    S_set.add(add_p2)
                    add_point(S_list, dist_sets, add_p2)
                    improved = True
                    no_improve_count = 0
                else:
                    S_set.add(remove_p1)
                    add_point(S_list, dist_sets, remove_p1)
                    S_set.add(remove_p2)
                    add_point(S_list, dist_sets, remove_p2)
                break

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                if time.time() > deadline - 30:
                    break
                order = generate_random_order(rng.randint(0, 100000))
                S_list, dist_sets, S_set = greedy_construct(order)
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S_list = S_list.copy()
                    best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
                    best_S_set = S_set.copy()
                no_improve_count = 0
        else:
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_S_list = S_list.copy()
                best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
                best_S_set = S_set.copy()

    result = [idx_to_point(idx) for idx in best_S_list]
    return result
# EVOLVE-BLOCK-END
