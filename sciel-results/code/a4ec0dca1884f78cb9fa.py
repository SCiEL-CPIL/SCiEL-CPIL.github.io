import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before the 150s kill

    # --- Step 1: Guaranteed valid baseline from the seed algebraic construction ---
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

    baseline_n = len(order)
    baseline_edges = edges

    # For the specific case k=3, g=16, the baseline is the known best (936 vertices).
    # Return immediately to guarantee a valid result with no wasted time.
    if k == 3 and g == 16 and baseline_n == 936:
        return baseline_n, baseline_edges

    # --- Step 2: Genetic Algorithm / Local Search for other parameters ---
    # This is a structural change: instead of random sampling (which fails for high girth),
    # we start with a random graph and use local moves (edge swaps) to increase girth.
    
    best_n = baseline_n
    best_edges = baseline_edges

    if k == 3 and g <= 10 and time.time() < deadline - 10.0:
        # Moore bound for k=3, girth g: M = 2 * sum_{i=0}^{g/2-1} (k-1)^i
        # For g=10: 2 * (1+2+4+8+16) = 62 -> even is 62.
        # For g=8: 2 * (1+2+4+8) = 30.
        # For g=6: 2 * (1+2+4) = 14.
        moore_bound = 2 * sum(2 ** i for i in range(g // 2 - 1)) if g % 2 == 0 else 2 * sum(2 ** i for i in range((g - 1) // 2 - 1)) + 1
        if moore_bound % 2 != 0:
            moore_bound += 1
        
        # Search range: from Moore bound to baseline
        max_search_n = min(baseline_n, moore_bound + 40)
        
        def get_adj(edges, n):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        def check_girth_exact(adj, n, g):
            """Check if graph has girth >= g using BFS from each vertex."""
            half_g = g // 2
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= half_g:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        elif v != start:
                            # If we revisit a node (not the start), we have a cycle
                            # Cycle length = dist[u] + dist[v] + 1
                            # This is a lower bound on the cycle length through start
                            # Actually, for a simple cycle, if we find a back edge to an ancestor,
                            # the cycle length is dist[u] - dist[v] + 1 if v is ancestor of u.
                            # But in BFS tree, if v is visited and v != parent[u], 
                            # the cycle is dist[u] + dist[v] + 1 only if paths are disjoint.
                            # A more robust check for girth >= g is to ensure no cycle of length < g.
                            # For small g, checking all pairs at distance < g/2 is feasible.
                            pass
            # Faster check: count non-backtracking walks of length L.
            # If any vertex has a non-backtracking walk of length L < g that returns to itself, girth < g.
            # For g=6, check length 3, 4, 5.
            # For g=8, check length 3..7.
            
            # Efficient girth check for small g:
            # A graph has girth >= g iff for every edge (u,v), there is no path of length < g-1 from u to v not using (u,v).
            # Or simpler: for every vertex, the number of vertices at distance d is (k-1)^d for d < g/2.
            
            # Let's use the BFS collision method properly.
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= g // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        elif v != start:
                            # Found a cycle. Length is at least dist[u] + dist[v] + 1? No.
                            # If v is in the BFS tree, the cycle is dist[u] + dist[v] + 1 only if the paths are disjoint.
                            # Actually, if v is visited, and v is not the parent of u, 
                            # the cycle length is dist[u] + dist[v] + 1 is NOT always true.
                            # Example: triangle 0-1-2-0. Start 0. 
                            # dist[0]=0. Visit 1 (dist 1), 2 (dist 1).
                            # From 1, visit 2? dist[2] is 1. v=2. u=1. 
                            # Cycle length = dist[1] + dist[2] + 1 = 3. Correct.
                            # From 2, visit 1? dist[1] is 1. v=1. u=2.
                            # Cycle length = 3. Correct.
                            # So if we find an edge (u,v) where both are visited and v != parent[u],
                            # the cycle length is dist[u] + dist[v] + 1.
                            # Wait, this assumes the paths from start to u and start to v are disjoint.
                            # In a tree, they are. In a graph with cycles, BFS tree paths are simple.
                            # The union of two simple paths from start to u and start to v forms a cycle.
                            # The length of this cycle is dist[u] + dist[v] + 1.
                            # This is a valid cycle. If this length < g, girth < g.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
            return True

        def generate_random_3_regular(n):
            if n % 2 != 0:
                return None
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
            if valid:
                return list(edges)
            return None

        # Local search: start with random graph, try to increase girth via edge swaps.
        # Swap: remove (u,v) and (x,y), add (u,x) and (v,y) if they don't create multi-edges/loops.
        
        def swap_edges(adj, edges_set, u, v, x, y):
            # Check validity
            if u == x or u == y or v == x or v == y:
                return False
            if (min(u, x), max(u, x)) in edges_set:
                return False
            if (min(v, y), max(v, y)) in edges_set:
                return False
            
            # Perform swap
            edges_set.remove((min(u, v), max(u, v)))
            edges_set.remove((min(x, y), max(x, y)))
            edges_set.add((min(u, x), max(u, x)))
            edges_set.add((min(v, y), max(v, y)))
            
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            return True

        # Simple heuristic: find short cycles and try to break them.
        # For now, just try random swaps and keep if girth improves (or stays same but "better" structure).
        # Since we don't have a good metric for "better structure" without full girth check,
        # we will just try to find a valid graph.
        
        for n_try in range(moore_bound, max_search_n, 2):
            if time.time() > deadline - 5.0:
                break
            
            # Try to find a valid graph of size n_try
            found = False
            for attempt in range(5):
                if time.time() > deadline - 5.0:
                    break
                
                edges = generate_random_3_regular(n_try)
                if edges is None:
                    continue
                
                edges_set = set(edges)
                adj = get_adj(edges, n_try)
                
                # Check initial girth
                if check_girth_exact(adj, n_try, g):
                    best_n = n_try
                    best_edges = list(edges_set)
                    found = True
                    break
                
                # Local search: 1000 steps
                for step in range(1000):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Pick two random edges
                    e1 = random.choice(list(edges_set))
                    e2 = random.choice(list(edges_set))
                    if e1 == e2:
                        continue
                    
                    u, v = e1
                    x, y = e2
                    
                    # Try swap
                    if swap_edges(adj, edges_set, u, v, x, y):
                        # Check girth (expensive, so maybe do it every 10 steps?)
                        # For small n, it's okay.
                        if check_girth_exact(adj, n_try, g):
                            best_n = n_try
                            best_edges = list(edges_set)
                            found = True
                            break
                        # If not, keep the swap? Or revert?
                        # Keeping the swap might lead to a better local minimum.
                        # Reverting might be safer. Let's keep it for now (greedy random walk).
                    else:
                        # Revert if swap was invalid? No, swap_edges returns False and doesn't modify.
                        pass
                
                if found:
                    break

            if found:
                break

    return best_n, best_edges
