import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin (150s kill, leave buffer)
    n = 18
    FULL = (1 << n) - 1

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
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
                        return True
        return False

    def get_triangles(adj):
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        triangles.append((1 << a) | (1 << b) | (1 << c))
        return triangles

    # Precompute suffix popcount: popcount(mask & suffix_mask[i])
    # where suffix_mask[i] = bits i..n-1
    suffix_mask = [0] * (n + 1)
    for i in range(n, -1, -1):
        if i == n:
            suffix_mask[i] = 0
        else:
            suffix_mask[i] = suffix_mask[i + 1] | (1 << i)

    # popcount for every mask (2^18 = 262144, small)
    POPC = [0] * (1 << n)
    for m in range(1, 1 << n):
        POPC[m] = POPC[m >> 1] + (m & 1)

    def max_triangle_free_size(triangles):
        """Branch and bound for largest triangle-free induced subgraph."""
        if not triangles:
            return n
        # Sort triangles by lowest vertex, then size
        # Precompute: for each vertex v, the set of triangles that contain v
        tri_by_vertex = [[] for _ in range(n)]
        for t in triangles:
            # find vertices in t
            for v in range(n):
                if (t >> v) & 1:
                    tri_by_vertex[v].append(t)

        # Precompute "required" mask for a triangle: all 3 vertices must be in any
        # triangle-free set that excludes the triangle.
        best = [0]

        # Iterative DFS to avoid recursion depth issues (depth 18, fine either way)
        # We'll use recursion with early pruning.
        def dfs(i, chosen_mask):
            # chosen_mask: vertices we've decided to INCLUDE among 0..i-1
            # i: next vertex to consider (0..n)
            if i == n:
                pc = POPC[chosen_mask]
                if pc > best[0]:
                    best[0] = pc
                return
            # Upper bound: current chosen + all remaining vertices
            remaining = POPC[suffix_mask[i]]
            cur = POPC[chosen_mask]
            if cur + remaining <= best[0]:
                return
            # Pruning: for each triangle fully in remaining (i..n-1), at least one of
            # its vertices must be excluded. If all 3 are "forced in" (i.e., we already
            # committed to them), that's a contradiction. But we don't force-in yet.
            # Stronger: if we include vertex i, check whether there's a triangle {i, a, b}
            # with a,b >= i+1 that we'd be forced to complete. We handle by: after deciding
            # to include i, if there exists a triangle containing i and two other vertices
            # both of which we will be forced to include (because excluding them would drop
            # below bound) -- too complex. Use simpler: just recurse.
            # Branch 1: exclude i
            dfs(i + 1, chosen_mask)
            # Branch 2: include i -- but only if no triangle is "completed" by this
            # inclusion together with already-included vertices.
            new_mask = chosen_mask | (1 << i)
            # Check: does new_mask contain any triangle? (triangles among already-chosen
            # plus i)
            ok = True
            for t in tri_by_vertex[i]:
                if (new_mask & t) == t:
                    ok = False
                    break
            if ok:
                dfs(i + 1, new_mask)

        dfs(0, 0)
        return best[0]

    def remove_k4(adj):
        for _ in range(200):
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

    def local_search(adj, current_score, max_iterations=2000, deadline=None):
        best_adj = adj[:]
        best_score = current_score
        no_improve = 0
        temperature = 1.0
        cooling = 0.995

        for _ in range(max_iterations):
            if deadline and time.time() > deadline:
                break
            if best_score <= 5:
                break
            if no_improve > 600:
                no_improve = 0
                temperature = max(temperature, 1.0)
                if random.random() < 0.3:
                    adj = best_adj[:]
                    current_score = best_score

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

            new_triangles = get_triangles(adj)
            new_score = max_triangle_free_size(new_triangles)
            delta = new_score - current_score

            if delta <= 0 or random.random() < min(1.0, math.exp(-delta / max(temperature, 0.01))):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                adj = old_adj
                no_improve += 1

            temperature *= cooling

        return best_adj, best_score

    random.seed(42)

    candidates = []

    # 1. K_{6,6,6}
    adj1 = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)

    # 2. Random dense graphs
    for _ in range(30):
        if time.time() > deadline - 60:
            break
        density = 0.5 + random.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # 3. Cyclic structured
    adj_s = [0] * n
    for i in range(n):
        for off in [2, 3, 5, 7, 11, 13]:
            j = (i + off) % n
            if j != i and not ((adj_s[i] >> j) & 1):
                adj_s[i] |= (1 << j)
                adj_s[j] |= (1 << i)
    if remove_k4(adj_s):
        candidates.append(adj_s)

    # 4. Quadratic structured
    adj_q = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (i * j + i + j) % 3 == 0:
                adj_q[i] |= (1 << j)
                adj_q[j] |= (1 << i)
    if remove_k4(adj_q):
        candidates.append(adj_q)

    best_adj = None
    best_score = 18

    for adj in candidates:
        if time.time() > deadline - 40:
            break
        triangles = get_triangles(adj)
        score = max_triangle_free_size(triangles)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(get_triangles(best_adj))

    # Local search from best
    adj = best_adj[:]
    current_score = best_score
    best_adj, best_score = local_search(adj, current_score, max_iterations=4000, deadline=deadline - 15)

    # Multiple restarts
    for restart in range(6):
        if time.time() > deadline - 15:
            break
        if best_score <= 6:
            break

        adj_p = best_adj[:]
        for _ in range(6):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (adj_p[i] >> j) & 1:
                adj_p[i] &= ~(1 << j)
                adj_p[j] &= ~(1 << i)
            else:
                adj_p[i] |= (1 << j)
                adj_p[j] |= (1 << i)
            if has_k4(adj_p):
                if (adj_p[i] >> j) & 1:
                    adj_p[i] &= ~(1 << j)
                    adj_p[j] &= ~(1 << i)
                else:
                    adj_p[i] |= (1 << j)
                    adj_p[j] |= (1 << i)

        triangles = get_triangles(adj_p)
        score = max_triangle_free_size(triangles)
        adj_r, score_r = local_search(adj_p, score, max_iterations=1500, deadline=deadline - 5)
        if score_r < best_score:
            best_score = score_r
            best_adj = adj_r

    return edges_from_adj(best_adj)
