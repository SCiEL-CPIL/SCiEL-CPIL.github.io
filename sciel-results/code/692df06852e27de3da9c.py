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
        
        # Optimized Girth Check
        # We need to ensure no cycle of length < g.
        # This is equivalent to: for every vertex s, there is no non-backtracking walk of length < g from s to s.
        
        for s in range(n):
            # State: (current_node, previous_node)
            # We track the shortest distance to each state.
            # Since k is small, the number of states is bounded by n * k.
            # We can use a list of dicts or a flat array if we map (u, prev) to an index.
            # Mapping: index = u * k + prev_index_in_adj[u]? No, prev is a neighbor.
            # Simpler: dict {(u, prev): dist}
            
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
                        continue
                    nd = d + 1
                    if v == s:
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

    # --- Family: Incidence graph of a projective plane (for k = q+1, prime power q) ---
    def is_prime_power(m):
        if m < 2:
            return False
        # Check if m is a power of a prime
        for p in range(2, int(math.isqrt(m)) + 1):
            if m % p == 0:
                # Check if m is a power of p
                x = m
                while x % p == 0:
                    x //= p
                if x == 1:
                    return True
                break
        # m might be prime
        if m > 1:
            for p in range(2, int(math.isqrt(m)) + 1):
                if m % p == 0:
                    return False
            return True
        return False

    def incidence_family():
        """
        For k = q+1 where q is a prime power, the incidence graph of the projective plane PG(2,q)
        is a (q+1)-regular bipartite graph with 2*(q^2+q+1) vertices and girth 8.
        For g=8, this gives n = 2*(q^2+q+1) with k=q+1.
        """
        results = []
        if k < 3:
            return results
        q = k - 1
        if not is_prime_power(q):
            return results
        # Build projective plane PG(2,q)
        # Points and lines in homogeneous coordinates over GF(q)
        # Number of points = number of lines = q^2 + q + 1
        n_pts = q * q + q + 1
        
        # Generate all points (1D subspaces) of GF(q)^3
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        
        # Generate all lines (2D subspaces, i.e., planes through origin)
        # A line is determined by a normal vector (a,b,c) up to scalar
        lines = []
        for a in range(q):
            for b in range(q):
                lines.append((a, b, 1))
            lines.append((a, 1, 0))
        lines.append((1, 0, 0))
        
        # Incidence: point (x,y,z) lies on line (a,b,c) iff a*x + b*y + c*z = 0
        # Build bipartite graph: left side = points (0..n_pts-1), right side = lines (n_pts..2*n_pts-1)
        n_total = 2 * n_pts
        edges = []
        for i, pt in enumerate(points):
            x, y, z = pt
            for j, ln in enumerate(lines):
                a, b, c = ln
                if (a * x + b * y + c * z) % q == 0:
                    edges.append((i, n_pts + j))
        
        results.append((n_total, edges))
        return results

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
        # Try n values from Moore bound up to a limit
        # For k=5, g=9, Moore bound is 428.
        # We can try up to 1000 or so, depending on time.
        max_n_to_try = min(moore + 500, 1000)
        for n in range(max(moore, k+1), max_n_to_try + 1):
            if check_deadline():
                break
            if n * k % 2 != 0:
                continue
            if k >= n:
                continue
            # Reduce max_iters as n increases to keep total time bounded
            iters = max(10, 2000 // (n // 100 + 1))
            result = random_regular_graph(n, k, rng, max_iters=iters)
            if result is not None:
                results.append(result)
        return results

    # --- Run all families ---
    # Prioritize Random Search as it can find smaller graphs if they exist,
    # but also keep explicit families.
    
    # 1. Try Incidence (fast to generate, small n for small q)
    try:
        inc_results = incidence_family()
        for n, e in inc_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    # 2. Try Circulant (fast to generate)
    try:
        circ_results = circulant_family()
        for n, e in circ_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    # 3. Try LUW (might be invalid for g=9, but worth a check)
    try:
        n, e = luw_family()
        consider(n, e)
    except Exception as ex:
        pass

    # 4. Try Random Regular Graph Search (most likely to find valid graphs if n is large enough)
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
        # Fallback: return LUW (might be invalid, but better than crash)
        n, e = luw_family()
        return n, e

    if best is not None:
        return best

    # Fallback: return LUW
    n, e = luw_family()
    return n, e

# EVOLVE-BLOCK-END
