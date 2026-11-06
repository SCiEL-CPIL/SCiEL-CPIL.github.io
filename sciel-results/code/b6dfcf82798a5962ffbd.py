"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        """Check if graph contains a K4."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                # Need two more vertices c, d both connected to a and b and to each other
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    # Now need d connected to a, b, c
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    while remaining:
                        d_bit = remaining & (-remaining)
                        d = d_bit.bit_length() - 1
                        return True
                        remaining &= remaining - 1
        return False

    def max_triangle_free_size(adj, time_limit=3.0):
        """Find the largest induced triangle-free subgraph using branch and bound."""
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda v: -deg[v])
        t0 = time.time()
        timeout = [False]

        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            # Pruning: even if we include all remaining, can't beat best
            if current_size + (n - idx) <= best[0]:
                return

            v = order[idx]

            # Option 1: Don't include v
            backtrack(idx + 1, current_mask, current_size)

            # Option 2: Include v (if it doesn't create a triangle with existing vertices)
            if not timeout[0]:
                neighbors_in_S = current_mask & adj[v]
                has_edge = False
                tmp = neighbors_in_S
                while tmp:
                    bit = tmp & (-tmp)
                    u = bit.bit_length() - 1
                    rest = tmp & ~bit
                    if (adj[u] & rest) != 0:
                        has_edge = True
                        break
                    tmp = rest
                if not has_edge:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)

        backtrack(0, 0, 0)
        return best[0]

    def remove_k4(adj):
        """Remove edges to make graph K4-free. Returns True if modified."""
        for _ in range(100):
            # Find a K4
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if not ((common >> c) & 1):
                            continue
                        remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                        if remaining:
                            d_bit = remaining & (-remaining)
                            d = d_bit.bit_length() - 1
                            # Remove a random edge from the K4 {a,b,c,d}
                            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                            x, y = pair
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return not has_k4(adj)

    random.seed(42)
    candidates = []

    # K_{6,6,6} baseline
    adj1 = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)

    # Random dense K4-free graphs with various densities
    for trial in range(80):
        if time.time() > deadline - 25:
            break
        density = 0.4 + random.random() * 0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Also try structured graphs: random partitions into 3 parts with high cross-density
    for trial in range(30):
        if time.time() > deadline - 25:
            break
        # Random partition into 3 parts of roughly equal size
        vertices = list(range(n))
        random.shuffle(vertices)
        part_sizes = [6, 6, 6]
        parts = []
        idx = 0
        for size in part_sizes:
            parts.append(vertices[idx:idx + size])
            idx += size

        adj = [0] * n
        cross_density = 0.7 + random.random() * 0.3
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        if random.random() < cross_density:
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
        # Add some within-part edges (these create triangles within parts)
        within_density = random.random() * 0.3
        for i in range(3):
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < within_density:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        if remove_k4(adj):
            candidates.append(adj)

    # Evaluate candidates
    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 20:
            break
        score = max_triangle_free_size(adj, time_limit=2.5)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=2.5)

    # Local search / simulated annealing to improve
    adj = best_adj[:]
    current_score = best_score
    best_adj = adj[:]
    best_score = current_score

    temperature = 1.0
    cooling = 0.995

    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        if best_score <= 5:
            break

        # Pick a random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i

        old_adj = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

        if has_k4(adj):
            adj = old_adj
            continue

        score = max_triangle_free_size(adj, time_limit=1.5)
        delta = score - current_score

        if delta <= 0 or random.random() < min(1.0, exp(-delta / max(temperature, 0.01))):
            current_score = score
            if score < best_score:
                best_score = score
                best_adj = adj[:]
        else:
            adj = old_adj

        temperature *= cooling

    return edges_from_adj(best_adj)

def exp(x):
    """Simple exponential function to avoid import."""
    import math
    return math.exp(x)
# EVOLVE-BLOCK-END
