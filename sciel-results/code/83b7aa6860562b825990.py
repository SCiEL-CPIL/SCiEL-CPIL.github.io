"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4_fast(adj):
        """Fast K4 detection using bitmask operations."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        if common & adj[c] & ~((1 << (c + 1)) - 1):
                            return True
        return False

    def max_triangle_free_size(adj, time_limit=3.0):
        """Find maximum triangle-free induced subgraph size using branch and bound."""
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        t0 = time.time()
        timeout = [False]
        
        # Precompute triangle-neighborhood info for each vertex
        # tri_adj[v] = set of vertices u such that u is adjacent to v and there exists w
        # Actually, we need: for a set S, is there a triangle in S?
        # A triangle exists if there are u,v,w in S all mutually adjacent.
        # When adding v to S, check if any two neighbors of v in S are adjacent.
        
        def has_triangle_with_v(v, mask):
            """Check if adding v to mask creates a triangle."""
            neighbors_in_mask = mask & adj[v]
            # Check if there's an edge within neighbors_in_mask
            tmp = neighbors_in_mask
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                rest = tmp & ~bit
                if (adj[u] & rest) != 0:
                    return True
                tmp = rest
            return False

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
            if current_size + (n - idx) <= best[0]:
                return

            v = order[idx]

            # Try including v first (finds large solutions faster for pruning)
            if not has_triangle_with_v(v, current_mask):
                backtrack(idx + 1, current_mask | (1 << v), current_size + 1)

            if not timeout[0]:
                backtrack(idx + 1, current_mask, current_size)

        backtrack(0, 0, 0)
        return best[0]

    def remove_k4(adj):
        """Remove edges to make graph K4-free."""
        for _ in range(500):
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                    for c in range(b + 1, n):
                        if (common >> c) & 1:
                            remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                            if remaining:
                                d_bit = remaining & (-remaining)
                                d = d_bit.bit_length() - 1
                                # Remove an edge from the K4 to break it
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
        return not has_k4_fast(adj)

    def make_partition_graph(parts, cross_density, within_density):
        adj = [0] * n
        num_parts = len(parts)
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                for a in parts[i]:
                    for b in parts[j]:
                        if random.random() < cross_density:
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
        for i in range(num_parts):
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < within_density:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    random.seed(42)
    candidates = []

    # Strategy 1: Complete tripartite K_{6,6,6}
    adj1 = [0] * n
    parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts3[i]:
                for b in parts3[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)

    # Strategy 1b: Complete tripartite with shuffled partitions
    for trial in range(10):
        vertices = list(range(n))
        random.shuffle(vertices)
        parts = [vertices[0:6], vertices[6:12], vertices[12:18]]
        adj = [0] * n
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        candidates.append(adj)

    # Strategy 2: Turan T(18,3) with internal edges (shuffled)
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        vertices = list(range(n))
        random.shuffle(vertices)
        parts = [vertices[0:6], vertices[6:12], vertices[12:18]]
        adj = [0] * n
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        for i in range(3):
            wd = 0.2 + random.random() * 0.5
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < wd:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 3: Random 4-partition
    for trial in range(25):
        if time.time() > deadline - 30:
            break
        vertices = list(range(n))
        random.shuffle(vertices)
        sizes = [5, 5, 4, 4]
        parts = []
        idx = 0
        for size in sizes:
            parts.append(vertices[idx:idx + size])
            idx += size
        adj = make_partition_graph(parts, 0.5 + random.random() * 0.5, random.random() * 0.35)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 4: 5-partition
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        vertices = list(range(n))
        random.shuffle(vertices)
        sizes = [4, 4, 4, 3, 3]
        parts = []
        idx = 0
        for size in sizes:
            parts.append(vertices[idx:idx + size])
            idx += size
        adj = make_partition_graph(parts, 0.4 + random.random() * 0.5, random.random() * 0.25)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 5: Dense random graphs
    for trial in range(40):
        if time.time() > deadline - 30:
            break
        density = 0.35 + random.random() * 0.45
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 6: Very dense random
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        density = 0.55 + random.random() * 0.3
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Evaluate candidates with short time limit
    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 20:
            break
        score = max_triangle_free_size(adj, time_limit=1.5)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=1.5)

    # Simulated annealing with improved neighborhood
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score

    temperature = 2.0
    cooling = 0.997
    max_iter = 20000
    restart_threshold = 500
    steps_since_improve = 0

    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        if best_score_sa <= 6:
            break

        # Choose move type: single edge (60%), double edge (25%), triple edge (15%)
        move_type = random.random()
        
        if move_type < 0.60:
            num_flips = 1
        elif move_type < 0.85:
            num_flips = 2
        else:
            num_flips = 3

        # Restart mechanism: if stuck, perturb more aggressively
        if steps_since_improve > restart_threshold and random.random() < 0.2:
            num_flips = random.randint(4, 8)

        old_adj = adj[:]
        flipped = []
        for _ in range(num_flips):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            flipped.append((i, j))

        if not flipped:
            continue

        if has_k4_fast(adj):
            adj = old_adj
            steps_since_improve += 1
            continue

        score = max_triangle_free_size(adj, time_limit=0.8)
        delta = score - current_score

        if delta <= 0 or random.random() < min(1.0, math.exp(-delta / max(temperature, 0.01))):
            current_score = score
            if score < best_score_sa:
                best_score_sa = score
                best_adj_sa = adj[:]
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        else:
            adj = old_adj
            steps_since_improve += 1

        temperature *= cooling

    # Final polish: try all single-edge flips from best solution
    adj = best_adj_sa[:]
    for i in range(n):
        if time.time() > deadline - 2:
            break
        for j in range(i + 1, n):
            if time.time() > deadline - 2:
                break
            old_adj = adj[:]
            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            if has_k4_fast(adj):
                adj = old_adj
                continue
            score = max_triangle_free_size(adj, time_limit=0.4)
            if score < best_score_sa:
                best_score_sa = score
                best_adj_sa = adj[:]
            else:
                adj = old_adj

    # Second polish pass if time permits
    if time.time() < deadline - 3:
        adj = best_adj_sa[:]
        for i in range(n):
            if time.time() > deadline - 1:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 1:
                    break
                old_adj = adj[:]
                if (adj[i] >> j) & 1:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                if has_k4_fast(adj):
                    adj = old_adj
                    continue
                score = max_triangle_free_size(adj, time_limit=0.3)
                if score < best_score_sa:
                    best_score_sa = score
                    best_adj_sa = adj[:]
                else:
                    adj = old_adj

    return edges_from_adj(best_adj_sa)
# EVOLVE-BLOCK-END
