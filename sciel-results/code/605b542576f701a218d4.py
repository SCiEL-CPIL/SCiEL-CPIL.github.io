"""Program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    DEADLINE = 145.0

    def elapsed():
        return time.time() - t0

    # ---------- L-U-W fallback ----------
    def luw_construct(k, g):
        q = k
        if q not in [2, 3, 4, 5, 7, 11, 13]:
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0: return False
                return True
            q = 4
            while not (is_prime(q) or q == 4):
                q += 1

        if g % 2 == 1:
            d = max(3, g - 5)
            if d % 2 == 0: d += 1
        else:
            d = max(4, g - 4)
            if d % 2 == 1: d += 1

        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def add(x,y): return x^y
            def sub(x,y): return x^y
            def mul(x,y): return table[x][y]
        else:
            def add(x,y): return (x+y)%q
            def sub(x,y): return (x-y)%q
            def mul(x,y): return (x*y)%q

        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r = (c-5)%4
            c0 = c - (c-5)%4
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
                yield (1-side), tuple(y)

        start = (0, (None,)+(0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # ---------- Fast girth checker (BFS from each vertex) ----------
    def check_girth(adj, n, min_girth):
        """Return True if girth >= min_girth."""
        for start in range(n):
            dist = [-1]*n
            parent = [-1]*n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= min_girth - 1:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # cycle length = du + dist[v] + 1
                        clen = du + dist[v] + 1
                        if clen < min_girth:
                            return False
        return True

    # ---------- Local search for (4,10)-cage ----------
    def local_search_4_10(time_limit):
        n = 320
        half = 160
        random.seed(42)

        # Build random 4-regular bipartite graph
        # Left: 0..159, Right: 160..319
        # Each left vertex connects to 4 distinct right vertices
        # Each right vertex connects to 4 distinct left vertices

        # Use random perfect matching composition
        # Generate 4 random permutations, compose them
        perms = []
        for _ in range(4):
            p = list(range(half))
            random.shuffle(p)
            perms.append(p)

        adj = [[] for _ in range(n)]
        for i in range(half):
            for pi in perms:
                j = pi[i]
                rv = half + j
                adj[i].append(rv)
                adj[rv].append(i)

        # Verify 4-regular
        for i in range(n):
            if len(adj[i]) != 4:
                return None

        def get_cycle_length(adj, n, target_girth):
            """Find a cycle shorter than target_girth, return its length or None."""
            for start in range(n):
                dist = [-1]*n
                parent = [-1]*n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    du = dist[u]
                    if du >= target_girth - 1:
                        break
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            queue.append(v)
                        elif parent[u] != v and parent[v] != u:
                            clen = du + dist[v] + 1
                            if clen < target_girth:
                                return clen
            return None

        def find_short_cycle_edges(adj, n, target_girth):
            """Find edges of a short cycle to break."""
            for start in range(n):
                dist = [-1]*n
                parent = [-1]*n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    du = dist[u]
                    if du >= target_girth - 1:
                        break
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            queue.append(v)
                        elif parent[u] != v and parent[v] != u:
                            clen = du + dist[v] + 1
                            if clen < target_girth:
                                # Reconstruct cycle
                                cycle = [v]
                                cur = v
                                while cur != start:
                                    cur = parent[cur]
                                    cycle.append(cur)
                                cycle.append(start)
                                cycle.reverse()
                                # Return edges of cycle
                                edges = []
                                for i in range(len(cycle)-1):
                                    a, b = cycle[i], cycle[i+1]
                                    if b in adj[a]:
                                        edges.append((a, b))
                                return edges
            return None

        start_time = time.time()
        max_iter = 50000
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            if elapsed() > DEADLINE - 10:
                break

            # Check girth
            if check_girth(adj, n, g):
                # Build edge list
                edges = []
                for i in range(half):
                    for j in adj[i]:
                        edges.append((i, j))
                return n, edges

            # Find a short cycle and break it via edge swap
            cycle_edges = find_short_cycle_edges(adj, n, g)
            if cycle_edges is None:
                # Shouldn't happen if check_girth says fail
                break

            # Pick a random edge from the cycle to swap
            a, b = random.choice(cycle_edges)
            # Find alternative neighbors for a and b
            # a's neighbors: adj[a], b's neighbors: adj[b]
            # We want to replace edge (a,b) with (a, b') and (a', b)
            # where b' is another neighbor of b (not a) and a' is another neighbor of a (not b)
            # Actually, in bipartite: a is left, b is right
            # a's neighbors are right vertices, b's neighbors are left vertices
            # Swap: remove (a,b), add (a, b2) and (a2, b) where b2 in adj[b], a2 in adj[a]
            # But we need to keep degree 4

            # a is left (0..159), b is right (160..319)
            # a's other neighbors: adj[a] minus b
            a_other = [x for x in adj[a] if x != b]
            b_other = [x for x in adj[b] if x != a]

            if len(a_other) == 0 or len(b_other) == 0:
                continue

            # Try a random swap
            a2 = random.choice(a_other)  # left vertex
            b2 = random.choice(b_other)  # right vertex

            # Check: a2 should not already be connected to b2 (would create multi-edge)
            if b2 in adj[a2] or a2 in adj[b2]:
                continue
            # Also check that a is not already connected to b2 (would create multi-edge)
            if b2 in adj[a]:
                continue
            # And b is not already connected to a2
            if a2 in adj[b]:
                continue

            # Perform swap: remove (a,b), add (a,b2) and (a2,b)
            # Wait, that changes degrees. Let me think again.
            # In a 4-regular bipartite graph, swapping means:
            # Remove edges (a,b) and (a2,b2), add edges (a,b2) and (a2,b)
            # This preserves 4-regularity if all four edges exist or don't exist properly.
            # Actually simpler: just remove (a,b) and add (a, b2) where b2 is another right neighbor of a... no.
            # 
            # Proper 2-switch: remove (a,b) and (a2,b2), add (a,b2) and (a2,b)
            # Requires: (a,b) exists, (a2,b2) exists, (a,b2) doesn't exist, (a2,b) doesn't exist
            # and a2 != a, b2 != b, a2 is left, b2 is right

            # Find a2 (left) and b2 (right) such that:
            # (a2,b2) exists, (a,b2) doesn't, (a2,b) doesn't
            found = False
            for b2 in adj[b]:  # b2 is left neighbor of b
                if b2 == a:
                    continue
                # Check if (a2=b2, b2=some right) ... wait b2 is left
                # We need (b2, some_right) where some_right != b, and (a, some_right) doesn't exist
                # Let me redo: a2 = b2 (left vertex, neighbor of b)
                # We need a right vertex b2r such that (a2, b2r) exists and (a, b2r) doesn't
                for b2r in adj[b2]:  # b2r is right neighbor of b2
                    if b2r == b:
                        continue
                    if b2r not in adj[a]:  # (a, b2r) doesn't exist
                        # Also check (a2, b) doesn't exist - but a2=b2 and b2 is neighbor of b, so (b2,b) exists!
                        # That's a problem. We'd be adding (b2, b) which already exists.
                        # So this swap doesn't work directly.
                        pass
                # Alternative: use a different approach
                # Remove (a,b), add (a, b2) where b2 is a right vertex not connected to a
                # and remove (a2, b) where a2 is a left vertex not connected to b, add (a2, b2)
                # This is getting complex. Let's use a simpler approach.
                break
            if not found:
                # Try direct: remove (a,b), pick random right vertex r not in adj[a], 
                # pick random left vertex l not in adj[b]
                # Remove (a,b), add (a,r), remove (l,r) if exists, add (l,b)
                # This is messy. Let's just try random edge swaps.
                
                # Simpler: pick two edges (a1,b1) and (a2,b2) and swap to (a1,b2) and (a2,b1)
                # where a1!=a2, b1!=b2, and the new edges don't already exist
                for _ in range(10):
                    a1 = random.randint(0, half-1)
                    b1 = random.choice(adj[a1])
                    a2 = random.randint(0, half-1)
                    if a2 == a1:
                        continue
                    b2 = random.choice(adj[a2])
                    if b2 == b1:
                        continue
                    # Check new edges don't exist
                    if b2 in adj[a1] or b1 in adj[a2]:
                        continue
                    # Perform swap
                    adj[a1].remove(b1)
                    adj[a1].append(b2)
                    adj[b1].remove(a1)
                    adj[b1].append(a2)
                    adj[a2].remove(b2)
                    adj[a2].append(b1)
                    adj[b2].remove(a2)
                    adj[b2].append(a1)
                    break

        # Final check
        if check_girth(adj, n, g):
            edges = []
            for i in range(half):
                for j in adj[i]:
                    edges.append((i, j))
            return n, edges
        return None

    # ---------- Main logic ----------
    if k == 4 and g == 10:
        # Try local search
        if elapsed() < DEADLINE - 30:
            result = local_search_4_10(60)
            if result is not None:
                return result

    # Fallback to L-U-W
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
