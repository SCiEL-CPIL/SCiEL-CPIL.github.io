import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # The task asks for a 3-regular graph with girth >= g.
    # The seed program returns n=936 (score 1.0 relative to some baseline, or just the value).
    # Wait, the prompt says "Score: n, minimized".
    # The seed program returns 936.
    # My strategy: Use LUW (Lifting with Unit Voltages) or similar algebraic construction.
    # A well-known family of 3-regular graphs with large girth is obtained by lifting the Petersen graph.
    # The Petersen graph has girth 5.
    # If we lift it with voltages in Z_m, the resulting graph has 10*m vertices.
    # The girth of the lift is at least 5 * m (if no short cycles are created by voltage cancellations).
    # We need girth >= 16.
    # If we choose m=4, n=40, girth >= 20. This is much smaller than 936.
    # Let's verify if we can find a valid voltage assignment for m=4.
    # If m=4 works, n=40. Score 40.
    # If m=3, n=30, girth >= 15. (15 < 16, so might fail if girth is exactly 15, but maybe 16? No, max girth is 5*m=15).
    # So m must be at least 4 to guarantee girth >= 20 > 16.
    # Actually, the girth of the lift is the length of the shortest cycle in the base graph
    # whose voltage sum is 0 mod m.
    # Petersen graph cycles: length 5, 6, ...
    # We need no cycle of length < 16 with voltage sum 0.
    # Since the shortest cycle in Petersen is 5, the shortest possible cycle in the lift is 5.
    # If m=4, a cycle of length 5 in the lift corresponds to a cycle of length 5 in the base
    # with voltage sum 0 mod 4.
    # If we can avoid any cycle of length 5, 6, ..., 15 having voltage sum 0 mod 4, we are good.
    # Actually, if the girth of the base is 5, the lift has girth at least 5.
    # It is known that random voltage assignments on the Petersen graph over Z_m yield graphs
    # with girth approximately 5 * log(m) or similar? No, the girth is determined by the shortest
    # zero-voltage cycle.
    # For m=4, it is very likely that there is a 5-cycle or 6-cycle with sum 0.
    # Let's try m=5. n=50. Girth >= 25?
    # Actually, the "cage" problem n(3,16) has a lower bound of 512?
    # Wait, the prompt says: "best lower bound is 512".
    # This implies that NO 3-regular graph with girth 16 exists on fewer than 512 vertices?
    # Let me re-read carefully.
    # "The smallest known such graph has 936 vertices ... and the best lower bound is 512."
    # This means n(3,16) >= 512.
    # My LUW idea with m=4 (n=40) is IMPOSSIBLE if the lower bound is 512.
    # Why? Because the Moore bound for (3,16) is 254.
    # The lower bound 512 is much higher.
    # This suggests that random graphs or simple lifts might not reach the required girth for small n.
    # However, the "lower bound" is a theoretical lower bound. It is possible that n=512 is the minimum.
    # The seed program returns 936.
    # If I can find a valid graph with n < 936, I improve the score.
    # The best known is 936.
    # If the lower bound is 512, maybe there exists a graph with n=512, 520, etc.
    # Random 3-regular graphs on n vertices have girth ~ log_2(n).
    # For n=512, log2(512) = 9. Girth ~ 18?
    # Heuristic: Girth of random 3-reg graph on n nodes is approx 2 log_2(n).
    # 2 * 9 = 18.
    # So n=512 should have girth around 18 with high probability?
    # If so, random search on n=512 should find a valid graph quickly.
    # The seed program tries n from 400 to 1000.
    # It uses `check_girth_fast` which is O(n^2).
    # For n=512, n^2 = 262,144. Very fast.
    # Why did the seed program fail to find a better one?
    # It iterates n from 400 to 1000 step 20.
    # It tries 30 attempts per n.
    # If n=512 is valid, it should find it.
    # Maybe the probability is low?
    # Or maybe my heuristic is wrong.
    # Let's check the literature in my head.
    # The (3,16)-cage is not solved. Best known is 936.
    # If random graphs worked, the best known would be much smaller.
    # Actually, the "girth of random regular graph" theory says that for fixed degree k,
    # the girth grows as log_{k-1}(n).
    # For k=3, log_2(n).
    # To get girth 16, we need log_2(n) >= 16/2 = 8? No, girth is diameter-like.
    # The number of walks of length L is approx n * (k-1)^L.
    # Collisions (cycles) occur when n * (k-1)^L ~ n^2 => (k-1)^L ~ n.
    # For k=3, 2^L ~ n.
    # So girth ~ log_2(n).
    # If we want girth 16, we need 2^16 ~ n => n ~ 65536?
    # No, that's for the *typical* girth to be 16.
    # If n=512, 2^L ~ 512 => L=9.
    # So typical girth is around 9 or 10.
    # We need girth 16. That's twice the typical value.
    # The probability of having no cycles of length < 16 is very small for n=512.
    # It's an extreme tail event.
    # So random search is unlikely to succeed for n close to the lower bound.
    
    # However, the seed program returns 936.
    # If I can just return 936, I get score 1.0 (or whatever the baseline is).
    # But the prompt says "Score: n, minimized".
    # And "seed/value".
    # If the seed returns 936, and I return 936, score is 1.0.
    # If I return 900, score is 936/900 > 1.0.
    
    # Is there a deterministic construction that beats 936?
    # The "best known" is 936.
    # Maybe I can try a different base graph for LUW?
    # Or use a different algebraic family?
    
    # Alternative: Use the "Cage" construction by Exoo et al.
    # But I don't have the specific parameters.
    
    # Let's look at the inspirations.
    # Inspiration 1: "defines construct"
    # Inspiration 2: "uses collections, random, sys, time"
    # Inspiration 3: "uses collections, random, sys, time"
    
    # The current program is a random search.
    # If random search is the only way, and it's hard, maybe I should try to optimize the search.
    # But the directive says "Make a STRUCTURAL change".
    
    # What if I use a **Genetic Algorithm** or **Simulated Annealing** on the 2-switch moves?
    # The current program does random 2-switches but doesn't use the cycle information to guide the search.
    # It just breaks the first cycle found? No, it just does a random switch.
    # A structural change: **Guided Local Search**.
    # 1. Generate random 3-reg graph.
    # 2. Find ALL short cycles.
    # 3. Apply 2-switches specifically to break these cycles.
    # 4. Repeat until no short cycles remain.
    
    # This is different from "random 2-switch". It is "targeted 2-switch".
    # This is a different optimization method (Greedy/Heuristic vs Random Walk).
    
    # Let's implement this.
    
    def generate_random_3reg(n, rng):
        if n < 4 or n % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        rng.shuffle(stubs)
        edges = []
        adj = [[] for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
        if not valid:
            return None
        edge_set = set()
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None
            edge_set.add(key)
        for i in range(n):
            if len(adj[i]) != 3 or len(set(adj[i])) != 3:
                return None
        return adj

    def find_short_cycles(adj, n, g):
        """
        Find cycles of length < g.
        Returns a list of (cycle_vertices, edge_to_break).
        We pick an edge in the cycle to break.
        """
        cycles = []
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != start and parent[v] != u and parent[u] != v:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            path1 = []
                            cur = u
                            while cur != -1:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.reverse()
                            
                            path2 = []
                            cur = v
                            while cur != -1:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.reverse()
                            
                            # Cycle is path1 + path2 reversed (excluding start duplicate)
                            # path1: u -> ... -> start
                            # path2: v -> ... -> start
                            # Edge (u,v) closes the loop.
                            # We can break any edge in the cycle.
                            # Let's pick the edge (u,v) or a random edge in the path.
                            # Breaking (u,v) is easy.
                            cycles.append((path1, path2, u, v))
                            if len(cycles) > 10: # Limit for speed
                                return cycles
        return cycles

    def break_cycle(adj, n, u, v, rng):
        """
        Perform a 2-switch to break the edge (u,v) or nearby.
        Actually, a 2-switch on edges (u,v) and (a,b) where a,b are not in the cycle
        might not break the cycle if the new edges don't interfere.
        To definitely break the cycle, we should switch an edge IN the cycle with an edge OUTSIDE.
        Let's pick an edge (x,y) in the cycle and (a,b) outside.
        Replace (x,y) and (a,b) with (x,a) and (y,b) (or cross).
        This removes (x,y) from the cycle, breaking it.
        """
        # Find an edge in the cycle to break. Let's use (u,v) if it's in the cycle?
        # In find_short_cycles, (u,v) is the back edge.
        # The cycle consists of path from u to v via start.
        # Let's just pick a random edge in the graph to switch with (u,v)?
        # No, (u,v) is an edge in the graph.
        # If we switch (u,v) with (a,b), we remove (u,v).
        # This definitely breaks any cycle containing (u,v).
        
        # Collect all edges
        edges = []
        edge_set = set()
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
                    edge_set.add((i, j))
        
        # We want to switch (u,v) with some other edge (a,b).
        # (u,v) must be in the edge list.
        # Let's find its index? Or just pick a random other edge.
        
        for _ in range(100):
            a, b = edges[rng.randint(0, len(edges)-1)]
            if (min(a,b), max(a,b)) == (min(u,v), max(u,v)):
                continue
            if a == u or a == v or b == u or b == v:
                continue
            # Check if (u,a) or (u,b) or (v,a) or (v,b) exists
            if (min(u,a), max(u,a)) in edge_set or (min(u,b), max(u,b)) in edge_set:
                continue
            if (min(v,a), max(v,a)) in edge_set or (min(v,b), max(v,b)) in edge_set:
                continue
            
            # Apply switch: remove (u,v) and (a,b), add (u,a) and (v,b)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[a].remove(b)
            adj[b].remove(a)
            
            adj[u].append(a)
            adj[a].append(u)
            adj[v].append(b)
            adj[b].append(v)
            return True
        return False

    best_n = float('inf')
    best_edges = None
    
    # Try n from 512 (lower bound) upwards.
    # Step 20.
    # Use the guided local search.
    
    for n in range(512, 950, 10):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few random starts
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            rng = random.Random(int(time.time()*1000) + n + attempt)
            adj = generate_random_3reg(n, rng)
            if adj is None:
                continue
            
            # Local search phase
            # Max 5000 steps to avoid getting stuck
            for step in range(5000):
                if time.time() > deadline - 5:
                    break
                cycles = find_short_cycles(adj, n, g)
                if not cycles:
                    # Valid graph found!
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    found = True
                    break
                
                # Pick a random cycle to break
                path1, path2, u, v = cycles[rng.randint(0, len(cycles)-1)]
                if not break_cycle(adj, n, u, v, rng):
                    # If we can't break this one, try a random 2-switch
                    # Or just break out if we fail too many times?
                    # Let's try a random switch
                    pass
            
        if found:
            break # We found a valid graph for this n. Since we iterate n increasing, this is the best n so far.
            # Should we try smaller n? We started at 512.
            # If we found one at 512, we stop.
            # If we didn't find one at 512, we try 522, etc.
            
    # Fallback to seed logic if local search fails
    if best_edges is None:
        # Use the seed's fallback or just return a known valid one if we have time?
        # The seed had a LUW fallback.
        # Let's implement a simple LUW fallback with large d to ensure validity,
        # even if n is large.
        
        def build_luw_fallback(d):
            q = 3
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
            
            rule = {}
            if d >= 2: rule[2] = (1, 1)
            if d >= 3: rule[3] = (2, 1)
            if d >= 4: rule[4] = (1, 2)
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

        best_n, best_edges = build_luw_fallback(11)

    return best_n, best_edges
