"""Portfolio SEARCH: try multiple explicit families, keep best (smallest) valid graph."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    deadline = time.time() + 115.0  # safety margin before 150s kill
    rng = random.Random(12345)

    def check_deadline():
        return time.time() > deadline

    # --- Utility: validate graph (simple, k-regular, girth >= g) ---
    def validate(n, edges, k, g):
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Girth check: no cycle of length < g
        # Use BFS from each vertex to find shortest cycle back to start
        # A cycle of length L means there's a path of length L-1 from s to a neighbor of s
        # that doesn't use the edge (s, neighbor) in the forward direction
        # More precisely: girth >= g iff for every vertex s, the shortest non-backtracking
        # closed walk from s has length >= g
        # Equivalently: for each s, do BFS in state space (current, prev) where we track
        # the shortest non-backtracking walk from s. If we can reach s again with length < g, fail.
        # But we need to be careful: reaching s with length L means a cycle of length L.
        # For L >= 2, this is a valid cycle.
        for s in range(n):
            # dist[(v, prev)] = shortest non-backtracking walk length from s to v arriving from prev
            dist = {}
            q = deque()
            for nb in adj[s]:
                dist[(nb, s)] = 1
                q.append((nb, s))
            while q:
                u, prev = q.popleft()
                d = dist[(u, prev)]
                if d >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue  # non-backtracking
                    nd = d + 1
                    if v == s:
                        # closed non-backtracking walk of length nd
                        # This represents a cycle of length nd (if nd >= 2)
                        if nd < g:
                            return False
                        continue
                    key = (v, u)
                    if key not in dist or dist[key] > nd:
                        dist[key] = nd
                        q.append((v, u))
        return True

    def try_graph(n, edges, k, g):
        if not validate(n, edges, k, g):
            return None
        return n, edges

    def random_regular_graph(n, k, rng, max_iters=2000):
        """Generate a random k-regular graph using configuration model + 2-switches."""
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
        for attempt in range(max_iters):
            # Configuration model
            half_edges = []
            for v in range(n):
                half_edges.extend([v] * k)
            rng.shuffle(half_edges)
            # Pair them up
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if not valid:
                continue
            # Check for multi-edges
            has_multi = False
            for u in range(n):
                if len(adj[u]) != len(set(adj[u])):
                    has_multi = True
                    break
            if has_multi:
                continue
            # Apply 2-switches to remove any remaining issues and randomize
            for _ in range(100):
                # Pick two random edges and switch
                e1 = (rng.randint(0, n-1), rng.randint(0, n-1))
                e2 = (rng.randint(0, n-1), rng.randint(0, n-1))
                a, b = e1
                c, d = e2
                if a == c or a == d or b == c or b == d:
                    continue
                if b not in adj[a] or d not in adj[c]:
                    continue
                # Try switching (a,b) and (c,d) -> (a,d) and (c,b)
                if d in adj[a] or b in adj[c]:
                    continue
                # Remove old edges
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].remove(d)
                adj[d].remove(c)
                # Add new edges
                adj[a].append(d)
                adj[d].append(a)
                adj[c].append(b)
                adj[b].append(c)
            # Build edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        return None

    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if n < best_n:
            result = try_graph(n, edges, k, g)
            if result is not None:
                best = result
                best_n = n
        if check_deadline():
            raise TimeoutError

    # --- Family 1: Lazebnik-Ustimenko-Woldar (from seed) ---
    def luw_family(q=None, d=None):
        if q is None:
            q = k
        if d is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # --- Family 2: Circulant / folded circulant constructions ---
    def circulant_family():
        # Try circulant graphs: vertices 0..n-1, edges (i, i+s) mod n for a set of shifts S.
        # For k even: k = 2*m, use m shifts.
        # For k odd: harder, try folded circulants or skip.
        results = []
        # Moore bound lower estimate
        if g % 2 == 1:
            r = (g-1)//2
            moore = 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g//2
            moore = k * sum((k-1)**i for i in range(r))
        
        max_n = moore + 500
        if k % 2 == 0:
            m = k // 2
            for n in range(max(moore, 2*m+1), max_n+1):
                if check_deadline():
                    break
                # Try several shift sets
                shift_sets = []
                # Simple consecutive
                if m <= n//2:
                    shift_sets.append(list(range(1, m+1)))
                # Fibonacci-like or other patterns
                if m == 1 and n > 2:
                    shift_sets.append([1])
                if m == 2 and n > 4:
                    shift_sets.append([1, 2])
                    shift_sets.append([1, 3])
                    shift_sets.append([2, 3])
                if m == 3 and n > 6:
                    shift_sets.append([1, 2, 3])
                    shift_sets.append([1, 2, 4])
                    shift_sets.append([1, 3, 4])
                    shift_sets.append([2, 3, 4])
                
                for shifts in shift_sets:
                    # Check no shift is self-inverse (s = n-s mod n, i.e., 2s = 0 mod n)
                    valid_shifts = True
                    for s in shifts:
                        if 2 * s % n == 0:
                            valid_shifts = False
                            break
                    if not valid_shifts:
                        continue
                    # Also check shifts are distinct mod n and none is 0
                    if len(set(s % n for s in shifts)) != len(shifts):
                        continue
                    if 0 in shifts:
                        continue
                    
                    edges = []
                    for i in range(n):
                        for s in shifts:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    results.append((n, edges))
        else:
            # k odd: try folded circulant or just skip
            pass
        return results

    # --- Family 3: Random regular graph search ---
    def random_regular_search():
        """Try random k-regular graphs near the Moore bound."""
        if g % 2 == 1:
            r = (g-1)//2
            moore = 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g//2
            moore = k * sum((k-1)**i for i in range(r))
        
        results = []
        # Try a few n values near the Moore bound
        for n in range(max(moore, k+1), moore + 200):
            if check_deadline():
                break
            if n * k % 2 != 0:
                continue
            if k >= n:
                continue
            result = random_regular_graph(n, k, rng, max_iters=500)
            if result is not None:
                results.append(result)
        return results

    # --- Run all families ---
    # Run LUW with default parameters
    try:
        n, e = luw_family()
        consider(n, e)
    except Exception as ex:
        pass

    # Try LUW with different d values
    for d_try in range(max(3, g-6), g+2):
        if check_deadline():
            break
        try:
            n, e = luw_family(d=d_try)
            consider(n, e)
        except Exception as ex:
            pass

    # Try circulant family
    try:
        circ_results = circulant_family()
        for n, e in circ_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    # Try random regular graph search
    try:
        rand_results = random_regular_search()
        for n, e in rand_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    if check_deadline():
        if best is not None:
            return best
        # Fallback: return LUW
        n, e = luw_family()
        return n, e

    if best is not None:
        return best

    # Fallback: return LUW
    n, e = luw_family()
    return n, e

# EVOLVE-BLOCK-END
