import time
import itertools

def construct():
    deadline = time.time() + 140

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + db * db + da * db

    # Generate all points in the radius-3 hexagon (37 points)
    R = 3
    all_points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if abs(a + b) <= R:
                all_points.append((a, b))

    n = len(all_points)  # 37
    assert n == 37

    # Build adjacency matrix and degrees
    adj = [[0] * n for _ in range(n)]
    deg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if dist_sq(all_points[i], all_points[j]) == 1:
                adj[i][j] = 1
                adj[j][i] = 1
                deg[i] += 1
                deg[j] += 1

    total_edges = sum(deg) // 2

    # We need to remove 5 points from 37 to get 32.
    # Edges lost = sum(deg[i] for i in removed) - edges_within_removed
    # Minimize edges lost to maximize remaining edges.

    # Pre-compute: for each point, its neighbors as a bitmask for fast intersection
    # Since n=37, we can use Python ints as bitmasks
    neighbor_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        neighbor_mask[i] = mask

    # Sort points by degree descending - try removing high-degree points first
    # but we need to check all combinations for optimality
    # Instead, pre-compute deg_sum for quick cost estimation

    best_cost = float('inf')
    best_removed = None

    remove_count = 5

    # Pre-compute pair adjacencies for the 5-combo check
    # Use a lookup: for a combo (i,j,k,l,m), edges_within = adj[i][j]+adj[i][k]+...
    # To speed up, pre-compute for each pair (i,j) whether they're adjacent

    for combo in itertools.combinations(range(n), remove_count):
        if time.time() > deadline - 5:
            break

        i, j, k, l, m = combo
        # Cost = sum of degrees - edges within combo
        cost = deg[i] + deg[j] + deg[k] + deg[l] + deg[m]
        # Subtract edges within the combo (6 pairs)
        if adj[i][j]:
            cost -= 1
        if adj[i][k]:
            cost -= 1
        if adj[i][l]:
            cost -= 1
        if adj[i][m]:
            cost -= 1
        if adj[j][k]:
            cost -= 1
        if adj[j][l]:
            cost -= 1
        if adj[j][m]:
            cost -= 1
        if adj[k][l]:
            cost -= 1
        if adj[k][m]:
            cost -= 1
        if adj[l][m]:
            cost -= 1

        if cost < best_cost:
            best_cost = cost
            best_removed = combo

    if best_removed is None:
        return [list(p) for p in all_points[:32]]

    # Build result: all points except those in best_removed
    removed_set = set(best_removed)
    result = [list(p) for i, p in enumerate(all_points) if i not in removed_set]

    # Verify: exactly 32 distinct points, |a|,|b| <= 64, mutual distance >= 1
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
    for i in range(32):
        for j in range(i + 1, 32):
            d = dist_sq(result[i], result[j])
            assert d >= 1, f"Points {result[i]} and {result[j]} too close: {d}"

    return result
