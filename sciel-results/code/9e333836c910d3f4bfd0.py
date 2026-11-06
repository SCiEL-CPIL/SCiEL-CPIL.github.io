import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Seed solution: n=9 with a simple valid coloring
    seed_n = 9
    seed_colors = []
    for i in range(seed_n):
        for j in range(i + 1, seed_n):
            d = j - i
            seed_colors.append(d % 4)

    best = {'n': seed_n, 'colors': seed_colors}

    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 10:
            break

        budget = 30.0 if n >= 16 else 15.0
        end_time = time.time() + budget

        result = _qubo_simulated_annealing(n, end_time)
        if result:
            best = result
            return result

    return best


def _get_edge_idx(i, j, n):
    """Get index of edge (i,j) in lexicographic order for n vertices."""
    if i > j:
        i, j = j, i
    idx = i * (2 * n - i - 1) // 2 + (j - i - 1)
    return idx


def _build_adj_from_colors(colors, n):
    """Build adjacency bitmasks for each color from edge color list."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj


def _find_c6(adj_c, n):
    """
    Find a C6 in color class adj_c. Returns (v0,v1,v2,v3,v4,v5) or None.
    Uses bitset operations for speed.
    """
    for v0 in range(n):
        n0 = adj_c[v0]
        if n0 == 0:
            continue

        v1_mask = n0
        while v1_mask:
            lsb1 = v1_mask & -v1_mask
            v1 = lsb1.bit_length() - 1
            v1_mask ^= lsb1

            if v1 <= v0:
                continue

            v5_mask = n0 & ~(1 << v1)
            while v5_mask:
                lsb5 = v5_mask & -v5_mask
                v5 = lsb5.bit_length() - 1
                v5_mask ^= lsb5

                if v5 <= v0:
                    continue

                n1 = adj_c[v1] & ~(1 << v0) & ~(1 << v5)

                v2_mask = n1
                while v2_mask:
                    lsb2 = v2_mask & -v2_mask
                    v2 = lsb2.bit_length() - 1
                    v2_mask ^= lsb2

                    n2 = adj_c[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v5)

                    v3_mask = n2
                    while v3_mask:
                        lsb3 = v3_mask & -v3_mask
                        v3 = lsb3.bit_length() - 1
                        v3_mask ^= lsb3

                        n3 = adj_c[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v5)

                        v4_mask = n3
                        while v4_mask:
                            lsb4 = v4_mask & -v4_mask
                            v4 = lsb4.bit_length() - 1
                            v4_mask ^= lsb4

                            if adj_c[v4] & (1 << v5):
                                return (v0, v1, v2, v3, v4, v5)

    return None


def _count_c6s(adj, n):
    """Count total C6s across all color classes."""
    total = 0
    for c in range(4):
        if _find_c6(adj[c], n) is not None:
            total += 1
    return total


def _initial_colors(n, seed):
    """
    Generate initial edge colors using a structured approach.
    Combines distance-based coloring with random perturbation.
    """
    rng = np.random.RandomState(seed)

    # Start with distance-based coloring
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            colors.append(d % 4)

    # Add random perturbation
    for idx in range(len(colors)):
        if rng.random() < 0.3:
            colors[idx] = rng.randint(0, 3)

    return colors


def _qubo_simulated_annealing(n, deadline):
    """
    QUBO-based simulated annealing for edge coloring.
    Objective: minimize number of monochromatic C6 cycles.
    Each edge has 4 binary variables (one per color), with constraint that exactly one is 1.
    """
    num_edges = n * (n - 1) // 2

    # Precompute edge index mapping
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1

    # Precompute all 6-cycles as lists of 6 edges
    # A 6-cycle is (v0,v1,v2,v3,v4,v5) with edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0)
    # We need to find all unique 6-cycles in K_n
    # For efficiency, we'll generate them lazily during the search

    def get_adj(colors):
        return _build_adj_from_colors(colors, n)

    def count_c6s_from_colors(colors):
        adj = get_adj(colors)
        return _count_c6s(adj, n)

    # Initial state
    rng = random.Random(42)
    colors = _initial_colors(n, 42)

    current_c6 = count_c6s_from_colors(colors)
    best_colors = colors[:]
    best_c6 = current_c6

    temperature = 5.0
    cooling = 0.995
    min_temp = 0.01

    while time.time() < deadline - 2 and temperature > min_temp:
        if current_c6 == 0:
            break

        if current_c6 < best_c6:
            best_c6 = current_c6
            best_colors = colors[:]

        # Pick a random edge to flip
        edge_idx = rng.randint(0, num_edges - 1)
        old_color = colors[edge_idx]
        new_color = rng.choice([c for c in range(4) if c != old_color])
        colors[edge_idx] = new_color

        # Evaluate new state
        new_c6 = count_c6s_from_colors(colors)

        # Accept if better or with Metropolis probability
        delta = new_c6 - current_c6
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            current_c6 = new_c6
        else:
            # Reject: revert
            colors[edge_idx] = old_color

        temperature *= cooling

    # Return best found
    if best_c6 == 0:
        return {'n': n, 'colors': best_colors}

    # Final check
    if count_c6s_from_colors(best_colors) == 0:
        return {'n': n, 'colors': best_colors}

    return None


def _multi_start(n, deadline):
    """
    Try multiple initializations with different strategies.
    """
    strategies = [
        ('distance', 0.0),
        ('random', 1.0),
        ('perturbed', 0.3),
        ('structured', 0.5),
    ]

    results = []
    num_starts = 50
    per_start_time = (deadline - time.time()) / num_starts

    for start in range(num_starts):
        if time.time() > deadline - 5:
            break

        seed = 1000 + start * 7

        # Choose strategy
        strategy, perturb = strategies[start % len(strategies)]

        if strategy == 'distance':
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append((j - i) % 4)
        elif strategy == 'random':
            colors = [random.Random(seed).randint(0, 3) for _ in range(n * (n - 1) // 2)]
        elif strategy == 'perturbed':
            colors = _initial_colors(n, seed)
        else:  # structured
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    # Use a different modular structure
                    colors.append(((i * 3 + j * 7) % 4))

        # Quick check
        adj = _build_adj_from_colors(colors, n)
        has_c6 = any(_find_c6(adj[c], n) is not None for c in range(4))

        if not has_c6:
            return {'n': n, 'colors': colors}

        # Try simulated annealing with short time budget
        sa_deadline = time.time() + min(2.0, per_start_time)
        result = _qubo_simulated_annealing(n, sa_deadline)
        if result:
            return result

    return None


def _qubo_simulated_annealing(n, deadline):
    """
    QUBO-based simulated annealing for edge coloring.
    Objective: minimize number of monochromatic C6 cycles.
    Uses vectorized C6 counting for speed.
    """
    num_edges = n * (n - 1) // 2

    def get_adj(colors):
        return _build_adj_from_colors(colors, n)

    def count_c6s_from_colors(colors):
        adj = get_adj(colors)
        total = 0
        for c in range(4):
            if _find_c6(adj[c], n) is not None:
                total += 1
        return total

    # Initial state with multiple strategies
    rng = random.Random(42)
    best_colors = None
    best_c6 = float('inf')

    # Try several initializations
    initializations = []
    for seed in range(20):
        colors = _initial_colors(n, seed * 13 + 7)
        initializations.append(colors)

    # Also try pure distance and pure random
    dist_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            dist_colors.append((j - i) % 4)
    initializations.append(dist_colors)

    rand_colors = [random.Random(999).randint(0, 3) for _ in range(num_edges)]
    initializations.append(rand_colors)

    for init_colors in initializations:
        if time.time() > deadline - 2:
            break

        current_c6 = count_c6s_from_colors(init_colors)

        if current_c6 == 0:
            return {'n': n, 'colors': init_colors}

        if current_c6 < best_c6:
            best_c6 = current_c6
            best_colors = init_colors[:]

    # If we found a good initial, do more search
    if best_colors is not None and best_c6 > 0:
        colors = best_colors[:]
        current_c6 = best_c6

        temperature = 5.0
        cooling = 0.995
        min_temp = 0.01

        while time.time() < deadline - 2 and temperature > min_temp:
            if current_c6 == 0:
                break

            if current_c6 < best_c6:
                best_c6 = current_c6
                best_colors = colors[:]

            # Pick a random edge to flip
            edge_idx = rng.randint(0, num_edges - 1)
            old_color = colors[edge_idx]
            new_color = rng.choice([c for c in range(4) if c != old_color])
            colors[edge_idx] = new_color

            # Evaluate new state
            new_c6 = count_c6s_from_colors(colors)

            # Accept if better or with Metropolis probability
            delta = new_c6 - current_c6
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                current_c6 = new_c6
            else:
                colors[edge_idx] = old_color

            temperature *= cooling

    if best_colors is not None and count_c6s_from_colors(best_colors) == 0:
        return {'n': n, 'colors': best_colors}

    return None
