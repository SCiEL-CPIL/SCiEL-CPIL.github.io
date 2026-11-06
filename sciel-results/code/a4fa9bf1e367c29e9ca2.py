import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for output
    rng = random.Random(42)

    # ---------- Bitmask helpers ----------

    def build_adj(n, edges_set):
        """Build adjacency as list of bitmasks from edge set."""
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edges_from_adj(n, adj):
        """Convert adjacency bitmask list to sorted edge list."""
        edges = []
        for u in range(n):
            mask = adj[u] & ~((1 << (u + 1)) - 1)  # only vertices > u
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                edges.append([u, v])
                mask ^= lsb
        return edges

    def has_k4(n, adj):
        """Check if graph contains a K4 using bitmask operations."""
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Only consider vertices > v to avoid duplicates
                    common &= ~((1 << (v + 1)) - 1)
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        if adj[w] & common:
                            return True
                        c ^= lsb
        return False

    def find_k4(n, adj):
        """Return a K4 as tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] & (1 << b)):
                    continue
                common = adj[a] & adj[b]
                common &= ~((1 << (b + 1)) - 1)
                c_mask = common
                while c_mask:
                    lsb = c_mask & -c_mask
                    c = lsb.bit_length() - 1
                    # Check if any vertex > c is connected to a, b, c
                    rest = common & ~((1 << (c + 1)) - 1)
                    r = rest
                    while r:
                        lsb2 = r & -r
                        d = lsb2.bit_length() - 1
                        if adj[d] & (1 << a) and adj[d] & (1 << b) and adj[d] & (1 << c):
                            return (a, b, c, d)
                        r ^= lsb2
                    c_mask ^= lsb
        return None

    def has_independent_set_6(n, adj):
        """Check if graph has an independent set of size 6 using DFS on complement."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(adj[i] | (1 << i))

        def dfs(candidates, size):
            if size == 0:
                return True
            if candidates == 0:
                return False
            if bin(candidates).count('1') < size:
                return False
            if size == 1:
                return True
            # Pick lowest set bit
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            # Include v
            if dfs(candidates & comp[v], size - 1):
                return True
            # Exclude v
            if dfs(candidates ^ lsb, size):
                return True
            return False

        return dfs(all_mask, 6)

    def find_independent_set_6(n, adj):
        """Find an independent set of size 6 or return None."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(adj[i] | (1 << i))

        found = [None]

        def dfs(candidates, size, current):
            if found[0] is not None:
                return
            if size == 0:
                found[0] = list(current)
                return
            if candidates == 0:
                return
            if bin(candidates).count('1') < size:
                return
            if size == 1:
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                found[0] = list(current) + [v]
                return
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            current.append(v)
            dfs(candidates & comp[v], size - 1, current)
            current.pop()
            dfs(candidates ^ lsb, size, current)

        dfs(all_mask, 6, [])
        return found[0]

    def max_independent_set_size(n, adj):
        """Compute maximum independent set size using Bron-Kerbosch with coloring."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(adj[i] | (1 << i))
        best = [0]

        def bk(R, P, X):
            if not P:
                if len(R) > best[0]:
                    best[0] = len(R)
                return
            if len(R) + bin(P).count('1') <= best[0]:
                return

            # Simple coloring bound
            colors = {}
            uncolored = P
            color = 0
            while uncolored:
                color += 1
                available = uncolored
                while available:
                    lsb = available & -available
                    v = lsb.bit_length() - 1
                    colors[v] = color
                    available ^= lsb
                    uncolored &= ~comp[v]
                    uncolored &= ~lsb

            max_color = max(colors.values()) if colors else 0
            if len(R) + max_color <= best[0]:
                return

            # Pivot
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                candidates = P

            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                bk(R + [v], P & comp[v], X & comp[v])
                P &= ~lsb
                X |= lsb
                c &= ~lsb

        bk([], all_mask, 0)
        return best[0]

    # ---------- Local search ----------

    def local_search(n, adj, time_limit):
        """Iteratively fix K4 and Alpha6 violations. Returns True if valid."""
        max_steps = 20000
        for step in range(max_steps):
            if time.time() > time_limit:
                return False

            # Check K4
            k4 = find_k4(n, adj)
            if k4:
                a, b, c, d = k4
                # Remove a random edge from the K4
                edges_in_k4 = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                u, v = rng.choice(edges_in_k4)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                continue

            # Check Alpha6
            if has_independent_set_6(n, adj):
                alpha6 = find_independent_set_6(n, adj)
                if alpha6 is None:
                    break
                # Add a random edge within the alpha6 set
                pairs = list(itertools.combinations(alpha6, 2))
                u, v = rng.choice(pairs)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                continue

            return True

        return False

    # ---------- Vertex extension ----------

    def try_extend(n, adj, time_limit):
        """Try to add one vertex. Returns new adj or None."""
        # For each candidate vertex, determine which edges to it create violations
        # We want to add vertex n with a set of edges such that:
        # 1. No K4 is created
        # 2. No independent set of 6 is created

        best_new_adj = None
        best_score = -1

        # Try several random edge patterns for the new vertex
        for attempt in range(50):
            if time.time() > time_limit:
                break

            # Randomly decide edges to existing vertices
            new_adj = [0] * (n + 1)
            for i in range(n):
                new_adj[i] = adj[i]
            new_adj[n] = 0

            # Start with no edges to new vertex, then add some
            for i in range(n):
                if rng.random() < 0.5:
                    new_adj[n] |= (1 << i)
                    new_adj[i] |= (1 << n)

            if not has_k4(n + 1, new_adj) and not has_independent_set_6(n + 1, new_adj):
                return new_adj

        # Try smarter: start from a valid configuration
        # Compute which subsets of neighbors are allowed
        # A vertex v's neighborhood S must satisfy:
        # - For any triangle in G containing vertices from S, S cannot contain all 3
        #   (actually, S can contain at most 2 vertices from any triangle, else K4)
        #   Wait: if S contains {a,b,c} and {a,b,c} form a triangle, then {a,b,c,n} is K4
        # - For any independent set I of size 5 in G, S must contain at least 1 vertex from I
        #   (else I ∪ {n} is independent set of size 6)

        # This is a constraint satisfaction problem. Let's try greedy:
        # Start with S = all vertices, remove vertices that cause K4, then add back
        # those needed to break alpha-6.

        # Actually, let's just try more random attempts with bias
        for attempt in range(100):
            if time.time() > time_limit:
                break

            new_adj = [0] * (n + 1)
            for i in range(n):
                new_adj[i] = adj[i]
            new_adj[n] = 0

            # Choose degree between n//3 and 2*n//3
            target_deg = rng.randint(n // 3, 2 * n // 3)
            neighbors = rng.sample(range(n), target_deg)
            for i in neighbors:
                new_adj[n] |= (1 << i)
                new_adj[i] |= (1 << n)

            if not has_k4(n + 1, new_adj) and not has_independent_set_6(n + 1, new_adj):
                return new_adj

        return None

    # ---------- Main search ----------

    # Start with Paley(17) as a known valid graph
    squares_17 = {x * x % 17 for x in range(1, 17)}
    adj_17 = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares_17:
                adj_17[i] |= (1 << j)
                adj_17[j] |= (1 << i)

    best_n = 17
    best_adj = adj_17[:]

    # Try to extend from Paley(17) up to 35
    for target_n in range(18, 36):
        if time.time() > deadline - 10.0:
            break

        extended = try_extend(target_n - 1, best_adj, deadline - 5.0)
        if extended is not None:
            best_n = target_n
            best_adj = extended
            # Try local search to see if we can stabilize
            if local_search(target_n, best_adj, deadline - 5.0):
                continue
            else:
                # Revert
                best_n = target_n - 1
                break
        else:
            break

    # If we haven't reached 35, try random restarts with local search
    for target_n in range(min(35, best_n + 1), 36):
        if time.time() > deadline - 15.0:
            break

        for attempt in range(30):
            if time.time() > deadline - 10.0:
                break

            # Random graph
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            if local_search(target_n, adj, deadline - 5.0):
                best_n = target_n
                best_adj = adj
                break

        if best_n >= target_n:
            break

    # Also try circulant graphs for larger n
    for target_n in range(35, 29, -1):
        if time.time() > deadline - 10.0:
            break
        if best_n >= target_n:
            continue

        for attempt in range(2000):
            if time.time() > deadline - 5.0:
                break
            max_d = target_n // 2
            distances = set()
            for d in range(1, max_d + 1):
                if rng.random() < 0.5:
                    distances.add(d)

            adj = [0] * target_n
            for d in distances:
                for i in range(target_n):
                    j = (i + d) % target_n
                    if i != j:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            for i in range(target_n):
                adj[i] &= ~(1 << i)

            if not has_k4(target_n, adj) and not has_independent_set_6(target_n, adj):
                best_n = target_n
                best_adj = adj
                break

        if best_n >= target_n:
            break

    # Final validation
    if not has_k4(best_n, best_adj) and not has_independent_set_6(best_n, best_adj):
        edges = edges_from_adj(best_n, best_adj)
        return {'n': best_n, 'edges': edges}

    # Fallback: return smallest valid graph found
    # n=1 is always valid
    return {'n': 1, 'edges': []}
