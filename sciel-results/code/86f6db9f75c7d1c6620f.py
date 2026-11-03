"""Portfolio SEARCH: try multiple explicit families, keep best (smallest) valid graph."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys
from collections import deque
import numpy as np
import itertools

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
        for s in range(n):
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

    def count_short_cycles(adj, n, k, g, max_len=8):
        """Count cycles of length < g using non-backtracking BFS from each vertex."""
        count = 0
        for s in range(n):
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
                            count += 1
                        continue
                    key = (v, u)
                    if key not in dist or dist[key] > nd:
                        dist[key] = nd
                        q.append((v, u))
        return count

    def find_short_cycles(adj, n, k, g):
        """Find specific short cycles (length < g) by collecting them."""
        cycles = []
        for s in range(n):
            # BFS tracking path
            dist = {}
            parent = {}
            q = deque()
            for nb in adj[s]:
                dist[(nb, s)] = 1
                parent[(nb, s)] = None
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
                            # Reconstruct cycle
                            cycle = [s]
                            cur = (u, prev)
                            while cur is not None:
                                cu, cp = cur
                                cycle.append(cu)
                                cur = parent.get(cur)
                            cycle.reverse()
                            cycles.append(cycle)
                        continue
                    key = (v, u)
                    if key not in dist or dist[key] > nd:
                        dist[key] = nd
                        parent[key] = (u, prev)
                        q.append((v, u))
        return cycles

    def try_graph(n, edges, k, g):
        if not validate(n, edges, k, g):
            return None
        return n, edges

    def random_regular_graph(n, k, rng, max_iters=2000, repair_iters=5000):
        """Generate a random k-regular graph using configuration model + 2-switches with cycle repair."""
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
        
        def try_generate():
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
                return None
            # Check for multi-edges
            has_multi = False
            for u in range(n):
                if len(adj[u]) != len(set(adj[u])):
                    has_multi = True
                    break
            if has_multi:
                return None
            return adj
        
        def do_2_switch(adj, n, k, rng, num_switches=1):
            """Perform 2-switches to randomize while preserving regularity."""
            for _ in range(num_switches):
                # Pick two random distinct edges
                attempts = 0
                while attempts < 50:
                    a = rng.randint(0, n-1)
                    c = rng.randint(0, n-1)
                    if a == c:
                        attempts += 1
                        continue
                    b_list = adj[a]
                    d_list = adj[c]
                    if len(b_list) == 0 or len(d_list) == 0:
                        attempts += 1
                        continue
                    b = b_list[rng.randint(0, len(b_list)-1)]
                    d = d_list[rng.randint(0, len(d_list)-1)]
                    # Check validity: a!=c, b!=d, and no existing edges (a,d) or (c,b)
                    if b == d:
                        attempts += 1
                        continue
                    if d in adj[a] or b in adj[c]:
                        attempts += 1
                        continue
                    # Perform the switch
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].remove(d)
                    adj[d].remove(c)
                    adj[a].append(d)
                    adj[d].append(a)
                    adj[c].append(b)
                    adj[b].append(c)
                    break
                    attempts += 1
        
        def repair_cycles(adj, n, k, g, rng, max_iter):
            """Iteratively repair short cycles using 2-switches."""
            for it in range(max_iter):
                # Find a short cycle
                cycles = find_short_cycles(adj, n, k, g)
                if not cycles:
                    return True
                # Pick a random short cycle
                cycle = cycles[rng.randint(0, len(cycles)-1)]
                # Try to break this cycle by doing a 2-switch involving one of its edges
                # For each edge in the cycle, try to find a partner edge to switch with
                found = False
                for i in range(len(cycle)-1):
                    u, v = cycle[i], cycle[i+1]
                    if v not in adj[u]:
                        continue
                    # Find a vertex w not adjacent to u, and a vertex x not adjacent to v,
                    # such that w-x is an edge and we can do a valid 2-switch
                    # Simpler: just do random 2-switches and hope to break cycles
                    # Try a few random 2-switches
                    for _ in range(20):
                        do_2_switch(adj, n, k, rng, 1)
                    # Re-check
                    cycles2 = find_short_cycles(adj, n, k, g)
                    if not cycles2:
                        return True
                    if len(cycles2) < len(cycles):
                        found = True
                        break
                if not found:
                    # Just do more random switches
                    do_2_switch(adj, n, k, rng, 5)
            return len(find_short_cycles(adj, n, k, g)) == 0
        
        for attempt in range(max_iters):
            adj = try_generate()
            if adj is None:
                continue
            # Initial randomization
            do_2_switch(adj, n, k, rng, 50)
            # Try to repair cycles
            if repair_cycles(adj, n, k, g, rng, repair_iters):
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
        """
        Search for k-regular circulant graphs with large girth.
        For k=5, n must be even. Shifts: {s1, -s1, s2, -s2, n/2}.
        """
        results = []
        if g % 2 == 1:
            r = (g-1)//2
            moore = 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g//2
            moore = k * sum((k-1)**i for i in range(r))
        
        # We are interested in small n. Start from Moore bound.
        # For k=5, n must be even.
        start_n = max(moore, k+1)
        if start_n % 2 != 0:
            start_n += 1
        end_n = start_n + 200  # Search a bit above Moore bound
        
        if k == 5:
            # Specific structure for k=5: 2 pairs + 1 self-inverse
            for n in range(start_n, end_n + 1, 2):
                if check_deadline():
                    break
                # Try a few specific shift sets
                # s1, s2 should be small to avoid small cycles?
                # Or spread out?
                candidates = [
                    (1, 2),
                    (1, 3),
                    (1, 4),
                    (2, 3),
                    (1, 5),
                    (2, 5),
                    (3, 5)
                ]
                for s1, s2 in candidates:
                    if s1 >= n//2 or s2 >= n//2 or s1 == s2:
                        continue
                    shifts = [s1, n-s1, s2, n-s2, n//2]
                    # Verify simplicity and regularity (implied by construction)
                    # Check for multi-edges: shifts must be distinct mod n
                    if len(set(shifts)) != 5:
                        continue
                    # Check for loops: 0 not in shifts
                    if 0 in shifts:
                        continue
                    edges = []
                    for i in range(n):
                        for s in shifts:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    results.append((n, edges))
        elif k % 2 == 0:
            # General even k
            m = k // 2
            for n in range(start_n, end_n + 1):
                if check_deadline():
                    break
                # Try consecutive shifts
                if m <= n//2:
                    shifts = list(range(1, m+1))
                    if len(set(shifts)) == m and 0 not in shifts:
                        edges = []
                        for i in range(n):
                            for s in shifts:
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                        results.append((n, edges))
        return results

    # --- Family 3: Random regular graph search with cycle repair ---
    def random_regular_search():
        """Try random k-regular graphs near the Moore bound with cycle repair."""
        if g % 2 == 1:
            r = (g-1)//2
            moore = 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g//2
            moore = k * sum((k-1)**i for i in range(r))
        
        results = []
        # Try a few n values near the Moore bound
        # For k=5, n must have same parity as k (n*k must be even, so n even when k odd)
        start_n = max(moore, k+1)
        if k % 2 == 1 and start_n % 2 != 0:
            start_n += 1
        
        # Try a range of n values, starting from Moore bound
        for n in range(start_n, start_n + 150):
            if check_deadline():
                break
            if n * k % 2 != 0:
                continue
            if k >= n:
                continue
            # Use fewer repair iterations for larger n to stay within time budget
            time_left = deadline - time.time()
            if time_left < 5:
                break
            max_it = max(50, int(1000 * time_left / 115.0))
            result = random_regular_graph(n, k, rng, max_iters=100, repair_iters=max_it)
            if result is not None:
                results.append(result)
        return results

    # --- Run all families ---
    # Priority: Circulants (deterministic, potentially small), LUW, Random
    
    # Try circulant family first as it might find small n quickly
    try:
        circ_results = circulant_family()
        for n, e in circ_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

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

    # Try random regular graph search with cycle repair
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
        try:
            n, e = luw_family()
            return n, e
        except:
            # Last resort: random graph
            n, e = random_regular_graph(500, k, rng, max_iters=10)
            if e is not None:
                return n, e
            raise Exception("Failed to construct any graph")

    if best is not None:
        return best

    # Fallback: return LUW
    try:
        n, e = luw_family()
        return n, e
    except:
        # Last resort: random graph
        n, e = random_regular_graph(500, k, rng, max_iters=10)
        if e is not None:
            return n, e
        raise Exception("Failed to construct any graph")

# EVOLVE-BLOCK-END
