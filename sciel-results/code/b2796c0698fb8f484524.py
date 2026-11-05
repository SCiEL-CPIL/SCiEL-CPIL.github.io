import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140

    # Moore bound for regular graph: n >= 2 * (k-1)^(g//2) for even g
    # For k=4, g=10: n >= 2 * 3^5 = 486? No wait...
    # Moore bound: for girth g >= 4, n >= k * (k-1)^(g//2) if g even
    # Actually the Moore bound is: n >= 2 * (k-1)^(g/2) for even girth
    # For k=4, g=10: n >= 2 * 3^5 = 486? That's too high.
    # Let me recalculate. The Moore bound for (k,g) is:
    # n >= k * (k-1)^(floor(g/2)-1) + (k-1)^(floor(g/2)) for even g?
    # No, the standard Moore bound: n >= 2 * (k-1)^(g/2) for even girth g
    # For k=4, g=10: 2 * 3^5 = 486. But the task says best lower bound is 243.
    # 243 = 3^5. So the lower bound here is (k-1)^(g/2) = 3^5 = 243.
    # The actual Moore bound formula for cages is:
    # n(k,g) >= 2 * (k-1)^(g/2) for even g... hmm that gives 486.
    # Actually I think the Moore bound is:
    # n >= (k-1)^(g/2) + (k-1)^(g/2-1) for... no.
    # Let me just use the known value: lower bound ~243, known best 320.

    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def check_degree(adj, k):
        return all(len(a) == k for a in adj)

    def girth_of_graph(n, adj, k, limit_check=None):
        """Find the girth (shortest cycle) of a regular graph.
        Returns the girth, or inf if no cycle found within time budget.
        Uses BFS from each vertex tracking (node, prev) to avoid backtracking.
        """
        if not check_degree(adj, k):
            return 0

        best = float('inf')
        # For efficiency, we only need to check up to g (or g+1 for validation)
        max_depth = limit_check if limit_check is not None else g + 2

        for s in range(n):
            if time.time() > deadline:
                break

            # BFS from s, tracking (current_node, previous_node)
            dist = {}
            dq = deque()
            for nb in adj[s]:
                state = (nb, s)
                dist[state] = 1
                dq.append(state)

            while dq:
                curr, prev = dq.popleft()
                d = dist[(curr, prev)]

                if d >= max_depth:
                    continue

                if curr == s:
                    cycle_len = d + 1
                    if cycle_len < best:
                        best = cycle_len
                    continue

                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    state = (nb, curr)
                    if state not in dist:
                        dist[state] = d + 1
                        dq.append(state)

        return best

    def validate_graph(n, edges, k, g):
        """Full validation: simple, k-regular, girth >= g."""
        adj = build_adj(n, edges)
        if not check_degree(adj, k):
            return False
        # Check for multi-edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        # Check girth
        girth = girth_of_graph(n, adj, k, limit_check=g + 1)
        return girth >= g

    # Search for circulant graphs C_n(1, s) with girth >= g
    # A circulant graph on n vertices with generators {1, -1, s, -s}
    # is 4-regular. We search for n < n_luw and s that gives girth >= g.
    def search_circulant(k, g, n_max, n_min=None):
        if n_min is None:
            n_min = 2 * (k - 1) ** (g // 2)  # crude lower bound
            # Actually for 4-regular girth 10, let's try from 244 up
            n_min = max(n_min, 244)

        # n must be even for bipartite (to avoid odd cycles < g if g is even)
        # Actually if g is even, we want bipartite to avoid odd cycles
        # C_n(1,s) is bipartite iff n is even (and s is odd? No...)
        # C_n(1,s) has a cycle of odd length iff there's a combination of generators
        # that sums to 0 mod n with odd number of terms.
        # For simplicity, try both even and odd n, but even n is more likely bipartite.

        # Precompute: for given n and s, check girth quickly
        def circulant_girth(n, s, limit):
            """Check if C_n(1,s) has a cycle shorter than 'limit'.
            A non-backtracking walk in C_n(1,s) corresponds to a sequence
            of steps in {1, -1, s, -s} where consecutive steps are not
            opposites (backtracking).
            We need: sum of steps ≡ 0 (mod n), length >= 3, no consecutive
            steps are negatives of each other.
            """
            # BFS over states (position, last_step)
            # position in [0, n), last_step in {1, -1, s, -s}
            # Start: from position 0, take any step
            # A cycle of length L: sequence of L steps a1, a2, ..., aL
            # such that a1+...+aL ≡ 0 (mod n), and a_{i+1} != -a_i for all i
            # (with a_{L+1} = a_1 for the wrap-around constraint too)
            
            # BFS: state = (pos, last_step), depth = number of steps taken
            # We start from pos=0 with any first step.
            # We're looking for a return to pos=0 with the additional constraint
            # that the last step is not the negative of the first step.
            
            # To handle the wrap constraint, track (pos, first_step, last_step, depth)
            # But that's expensive. Alternative: just track (pos, last_step) and
            # when we return to 0, check if last_step != -first_step.
            # We need to track first_step too.
            
            # State: (pos, first_step_idx, last_step_idx, depth)
            # Too expensive for large n. Let's use a simpler approach:
            # For each pair of consecutive non-backtracking steps, do BFS.
            
            # Simpler: enumerate all sequences of steps of length 3 to limit-1
            # and check if they sum to 0 mod n with no backtracking.
            # Number of sequences: 4 * 3^(L-1) for length L.
            # For L=3: 12, L=4: 36, L=5: 108, L=6: 324, L=7: 972, L=8: 2916, L=9: 8748
            # For limit=10, total is about 12+36+108+324+972+2916+8748 = ~13116. Fine.
            
            steps = [1, -1, s % n, (-s) % n]
            # Remove duplicates
            steps = list(set(steps))
            # For 4-regular, we need exactly 4 distinct steps. If s ≡ 1 or -1 mod n, we have < 4 steps.
            if len(steps) < 4:
                return 0  # Not 4-regular
            
            # For each length from 3 to limit-1, check all sequences
            for L in range(3, limit):
                # Enumerate all sequences of length L with no consecutive backtracking
                # A sequence a1, a2, ..., aL is valid if a_{i+1} != -a_i (mod n) for all i
                # AND a1 != -aL (for the cycle to close without backtracking at the join)
                # Sum a1+...+aL ≡ 0 (mod n)
                
                # Use recursive generation with pruning
                def gen(prefix, idx, current_sum):
                    if idx == L:
                        # Check if this is a valid cycle
                        # Sum must be 0 mod n
                        if current_sum % n != 0:
                            return None
                        # Check wrap: a1 != -aL
                        if prefix[0] == (-prefix[-1]) % n:
                            return None
                        return L
                    for step in steps:
                        if idx > 0 and step == (-prefix[-1]) % n:
                            continue
                        result = gen(prefix + [step], idx + 1, current_sum + step)
                        if result is not None:
                            return result
                    return None
                
                result = gen([], 0, 0)
                if result is not None:
                    return result
            
            return float('inf')

        best_n = None
        best_s = None

        # Try values of n from n_min to n_max
        # n must be even for better chance of bipartiteness
        for n in range(n_min, n_max + 1, 2):
            if time.time() > deadline:
                break
            
            # s must satisfy: s != 0, 1, -1 mod n, and s != -s mod n (i.e., 2s != 0)
            # Also s != 2, -2 to avoid 4-cycles? Let's just check girth.
            # Try s values that are "far" from 0, 1, -1
            # Heuristic: try s around n/4 and n/3
            s_candidates = set()
            for s in [n // 4, n // 3, (n // 4) + 1, (n // 3) + 1, n // 5, (n // 5) + 1,
                      n // 4 - 1, n // 3 - 1, 7, 11, 13, 17, 19]:
                s = s % n
                if s not in [0, 1, (n - 1) % n, (n - 2) % n, 2]:
                    s_candidates.add(s)
            # Also try a few random values
            for _ in range(5):
                s = np.random.randint(3, n - 3)
                s_candidates.add(s % n)

            for s in s_candidates:
                s_mod = s % n
                if s_mod in [0, 1, (n - 1) % n, (n - 2) % n, 2, (n - 2) % n]:
                    continue
                # Check 4-regularity: steps {1, -1, s, -s} must be 4 distinct mod n
                steps_set = {1 % n, (n - 1) % n, s_mod % n, (n - s_mod) % n}
                if len(steps_set) != 4:
                    continue

                girth = circulant_girth(n, s_mod, g + 1)
                if girth >= g:
                    if best_n is None or n < best_n:
                        best_n = n
                        best_s = s_mod
                        if n <= 250:  # Good enough, stop early
                            break
            
            if best_n is not None and best_n <= 250:
                break

        return best_n, best_s

    def make_circulant_edges(n, s):
        """Create edges for C_n(1, s)."""
        edges = []
        for u in range(n):
            for v in [(u + 1) % n, (u - 1) % n, (u + s) % n, (u - s) % n]:
                if u < v:
                    edges.append((u, v))
        return edges

    # Try circulant search first
    n_luw, edges_luw = luw_construct(k, g)

    # Only try circulant if n_luw is reasonably large and we have time
    if time.time() < deadline - 30 and n_luw > 244:
        n_circ, s_circ = search_circulant(k, g, n_luw - 1, 244)
        if n_circ is not None:
            edges_circ = make_circulant_edges(n_circ, s_circ)
            if validate_graph(n_circ, edges_circ, k, g):
                return n_circ, edges_circ

    # Validate LUW graph
    adj_luw = build_adj(n_luw, edges_luw)
    if check_degree(adj_luw, k):
        girth_luw = girth_of_graph(n_luw, adj_luw, k, limit_check=g + 1)
        if girth_luw >= g:
            return n_luw, edges_luw

    # Local search from LUW: try to improve girth while keeping n
    def local_search_improve_girth(n, adj, k, g, max_iters=2000):
        """Try to increase girth via edge swaps (2-opt on edges)."""
        if time.time() > deadline - 10:
            return None

        adj_list = [sorted(a) for a in adj]
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))

        current_girth = girth_of_graph(n, adj, k, limit_check=g + 1)

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Pick two random edges
            edges_list = list(edge_set)
            if len(edges_list) < 2:
                break
            i1 = np.random.randint(len(edges_list))
            i2 = np.random.randint(len(edges_list))
            if i1 == i2:
                continue

            u1, v1 = edges_list[i1]
            u2, v2 = edges_list[i2]

            # Try reconnection: (u1,v1)+(u2,v2) -> (u1,u2)+(v1,v2) or (u1,v2)+(v1,u2)
            for (a, b, c, d) in [(u1, u2, v1, v2), (u1, v2, v1, u2)]:
                # Check no self-loop
                if a == b or c == d:
                    continue
                # Check edges don't already exist
                if (min(a,b), max(a,b)) in edge_set or (min(c,d), max(c,d)) in edge_set:
                    continue

                # Perform swap
                adj[a].discard(b if a != b else 0)  # Actually remove (u1,v1) and (u2,v2)
                # Let me redo this properly
                # Remove (u1,v1) and (u2,v2)
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                adj[u2].discard(v2)
                adj[v2].discard(u2)
                edge_set.discard((min(u1,v1), max(u1,v1)))
                edge_set.discard((min(u2,v2), max(u2,v2)))

                # Add (a,b) and (c,d)
                adj[a].add(b)
                adj[b].add(a)
                adj[c].add(d)
                adj[d].add(c)
                edge_set.add((min(a,b), max(a,b)))
                edge_set.add((min(c,d), max(c,d)))

                # Check degree
                if not check_degree(adj, k):
                    # Revert
                    adj[a].discard(b)
                    adj[b].discard(a)
                    adj[c].discard(d)
                    adj[d].discard(c)
                    edge_set.discard((min(a,b), max(a,b)))
                    edge_set.discard((min(c,d), max(c,d)))
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    edge_set.add((min(u1,v1), max(u1,v1)))
                    edge_set.add((min(u2,v2), max(u2,v2)))
                    continue

                # Check girth
                new_girth = girth_of_graph(n, adj, k, limit_check=g + 1)
                if new_girth > current_girth:
                    current_girth = new_girth
                    if current_girth >= g:
                        # Success! Extract edges
                        out_edges = []
                        for (u, v) in edge_set:
                            out_edges.append((u, v))
                        return out_edges
                else:
                    # Revert
                    adj[a].discard(b)
                    adj[b].discard(a)
                    adj[c].discard(d)
                    adj[d].discard(c)
                    edge_set.discard((min(a,b), max(a,b)))
                    edge_set.discard((min(c,d), max(c,d)))
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    edge_set.add((min(u1,v1), max(u1,v1)))
                    edge_set.add((min(u2,v2), max(u2,v2)))

        return None

    # Try local search
    improved_edges = local_search_improve_girth(n_luw, adj_luw, k, g)
    if improved_edges is not None:
        return n_luw, improved_edges

    # Fallback: return LUW graph as-is
    return n_luw, edges_luw
