import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Try to extend a deterministic geometric construction of n=25 to n=26
    #    This gives a guaranteed valid baseline of n=25.
    n25_colors = geometric_construct_25()
    if n25_colors is not None:
        r26 = try_extend_focused({'n': 25, 'colors': n25_colors}, time_limit=30.0)
        if r26 is not None:
            # Try to extend to 27 if time permits
            if time.time() < deadline - 30:
                r27 = try_extend_focused(r26, time_limit=25.0)
                if r27 is not None:
                    return r27
            return r26
        else:
            # Fallback to n=25 if extension fails
            return {'n': 25, 'colors': n25_colors}

    # 2. Fallback: Randomized local search for n=26
    r = focused_local_search(26, time_limit=40.0)
    if r is not None:
        if time.time() < deadline - 30:
            r27 = try_extend_focused(r, time_limit=25.0)
            if r27 is not None:
                return r27
        return r

    # 3. Fallback: Smaller n
    for n in [25, 24, 23]:
        if time.time() > deadline - 5:
            break
        r = focused_local_search(n, time_limit=10.0)
        if r is not None:
            return r

    # Absolute fallback
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}


def geometric_construct_25():
    """
    Deterministic construction of a valid 5-coloring of K_25.
    Vertices are points in AG(2, 5) = Z_5 x Z_5.
    Edge (u, v) is colored by the slope of the line uv, where:
    - Vertical lines get color 4.
    - Non-vertical lines get color = slope mod 5 (0, 1, 2, 3).
    """
    n = 25
    colors = []
    for i in range(n):
        x1, y1 = divmod(i, 5)
        for j in range(i + 1, n):
            x2, y2 = divmod(j, 5)
            if x1 == x2:
                c = 4
            else:
                dx = (x2 - x1) % 5
                dy = (y2 - y1) % 5
                inv_dx = pow(dx, -1, 5)  # Modular inverse
                c = (dy * inv_dx) % 5
            colors.append(c)
    return colors


def try_extend_focused(base_result, time_limit=40.0):
    """
    Extend a valid n-vertex coloring to n+1 vertices using focused search.
    This is much faster than full local search because we only need to check
    violations involving the new vertex.
    """
    base_colors = base_result['colors']
    n_base = base_result['n']
    n_new = n_base + 1

    # Build adjacency masks for base
    adj_mask = [[0] * n_base for _ in range(5)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj_mask[c][i] |= (1 << j)
            adj_mask[c][j] |= (1 << i)
            idx += 1

    def count_violations_new(new_colors):
        """Count C4 violations involving the new vertex."""
        violations = 0
        for c in range(5):
            # Get neighbors of new vertex in color c
            neighbors = [i for i in range(n_base) if new_colors[i] == c]
            # For any pair of neighbors (u, v), if they share a common neighbor in color c,
            # that forms a C4 with the new vertex.
            for a in range(len(neighbors)):
                u = neighbors[a]
                for b in range(a + 1, len(neighbors)):
                    v = neighbors[b]
                    common = adj_mask[c][u] & adj_mask[c][v]
                    k = common.bit_count()
                    if k >= 1:
                        violations += k
        return violations

    # Initialize
    new_colors = [random.randint(0, 4) for _ in range(n_base)]
    current_violations = count_violations_new(new_colors)

    if current_violations == 0:
        return build_full_coloring(base_colors, new_colors, n_base)

    best_violations = current_violations
    best_new_colors = new_colors[:]

    start_time = time.time()
    temperature = 1.0
    no_improve = 0

    while time.time() - start_time < time_limit:
        if best_violations == 0:
            break
        if no_improve > 2000:
            new_colors = [random.randint(0, 4) for _ in range(n_base)]
            current_violations = count_violations_new(new_colors)
            temperature = 1.0
            no_improve = 0
            continue

        # Pick a variable: prefer those in violating pairs
        # For simplicity, pick random
        e_idx = random.randint(0, n_base - 1)
        old_c = new_colors[e_idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue

        new_colors[e_idx] = new_c
        new_violations = count_violations_new(new_colors)

        delta = new_violations - current_violations
        if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_new_colors = new_colors[:]
                no_improve = 0
                if best_violations == 0:
                    break
            else:
                no_improve += 1
        else:
            new_colors[e_idx] = old_c
            no_improve += 1

        temperature *= 0.999
        if temperature < 0.05:
            temperature = 0.05

    if best_violations == 0:
        return build_full_coloring(base_colors, best_new_colors, n_base)

    return None


def build_full_coloring(base_colors, new_colors, n_base):
    """Build the full (n_base+1)-vertex coloring from base and new vertex colors."""
    n_new = n_base + 1
    final_colors = [0] * (n_new * (n_new - 1) // 2)

    idx = 0
    for i in range(n_new):
        for j in range(i + 1, n_new):
            if j < n_base:
                # Edge in base graph
                base_idx = i * (n_base - 1) - i * (i - 1) // 2 + (j - i - 1)
                final_colors[idx] = base_colors[base_idx]
            else:
                # Edge to new vertex (j == n_base)
                final_colors[idx] = new_colors[i]
            idx += 1

    return {'n': n_new, 'colors': final_colors}


def focused_local_search(n, time_limit=40.0):
    """
    Fast local search with delta calculation.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    # Initialize colors randomly
    colors = [random.randint(0, 4) for _ in range(num_edges)]

    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]

    def update_neighbors(idx, old_color, new_color):
        i, j = edges[idx]
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)

    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)

    # Precompute common neighbor counts for each color and pair of vertices
    k = [[[0] * n for _ in range(n)] for _ in range(5)]

    def calc_k_for_color(c):
        for u in range(n):
            for v in range(u + 1, n):
                common = neighbors[c][u] & neighbors[c][v]
                k[c][u][v] = common.bit_count()
                k[c][v][u] = common.bit_count()

    for c in range(5):
        calc_k_for_color(c)

    def total_violations():
        v = 0
        for c in range(5):
            for u in range(n):
                for v_idx in range(u + 1, n):
                    kc = k[c][u][v_idx]
                    if kc >= 2:
                        v += kc * (kc - 1) // 2
        return v

    current_violations = total_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}

    best_violations = current_violations
    best_colors = colors[:]

    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999

    # Local search loop
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]

        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue

        i, j = edges[e_idx]

        # Calculate delta violations
        delta = 0

        # Effect on old_color
        for w in range(n):
            if w == i or w == j:
                continue
            if (neighbors[old_color][w] >> j) & 1:
                old_k = k[old_color][i][w]
                new_k = old_k - 1
                delta -= (old_k * (old_k - 1) // 2) - (new_k * (new_k - 1) // 2)
                k[old_color][i][w] = new_k
                k[old_color][w][i] = new_k

            if (neighbors[old_color][w] >> i) & 1:
                old_k = k[old_color][j][w]
                new_k = old_k - 1
                delta -= (old_k * (old_k - 1) // 2) - (new_k * (new_k - 1) // 2)
                k[old_color][j][w] = new_k
                k[old_color][w][j] = new_k

        # Effect on new_color
        for w in range(n):
            if w == i or w == j:
                continue
            if (neighbors[new_color][w] >> j) & 1:
                old_k = k[new_color][i][w]
                new_k = old_k + 1
                delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                k[new_color][i][w] = new_k
                k[new_color][w][i] = new_k

            if (neighbors[new_color][w] >> i) & 1:
                old_k = k[new_color][j][w]
                new_k = old_k + 1
                delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                k[new_color][j][w] = new_k
                k[new_color][w][j] = new_k

        # Apply change if accepted
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations += delta
            colors[e_idx] = new_color
            update_neighbors(e_idx, old_color, new_color)

            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert k changes
            for w in range(n):
                if w == i or w == j:
                    continue
                common = neighbors[old_color][i] & neighbors[old_color][w]
                k[old_color][i][w] = common.bit_count()
                k[old_color][w][i] = common.bit_count()

                common = neighbors[old_color][j] & neighbors[old_color][w]
                k[old_color][j][w] = common.bit_count()
                k[old_color][w][j] = common.bit_count()

                common = neighbors[new_color][i] & neighbors[new_color][w]
                k[new_color][i][w] = common.bit_count()
                k[new_color][w][i] = common.bit_count()

                common = neighbors[new_color][j] & neighbors[new_color][w]
                k[new_color][j][w] = common.bit_count()
                k[new_color][w][j] = common.bit_count()

        temperature *= cooling_rate
        if temperature < 0.05:
            temperature = 0.05

    return None
