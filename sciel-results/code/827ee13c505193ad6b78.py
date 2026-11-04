import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 148.0

    # Moore bound for k=3, g=16
    if k == 3:
        if g % 2 == 0:
            moore_bound = 2 * sum((k-1)**i for i in range(g//2))
        else:
            moore_bound = (k-1) * sum((k-1)**i for i in range((g-1)//2))
    else:
        moore_bound = 1

    def check_girth_fast(adj, n, g):
        """Exact girth check using BFS from each vertex."""
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
                    elif v != start and dist[u] + dist[v] + 1 < g:
                        return False
        return True

    def generate_random_3reg(n, rng):
        """
        Generate a random 3-regular graph on n vertices using configuration model.
        Returns adjacency list or None if not simple/valid.
        """
        if n < 4:
            return None
        if n % 2 != 0:
            return None
            
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        
        rng.shuffle(stubs)
        
        # Pair up stubs
        edges = []
        adj = [[] for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            # Check for duplicate edge (simple graph)
            # Since n is potentially large, checking all pairs is expensive.
            # We'll use a set of frozensets for quick lookup during construction.
            # But for n~4000, 6000 edges, a set is fine.
            # Actually, let's just build and verify at the end to save overhead in loop?
            # No, duplicate edges make it not 3-regular in the simple sense (multigraph).
            # We need simple.
            
            # For efficiency, we'll collect edges and check at end, 
            # but rejecting multigraphs early is better.
            # Let's use a set of (min(u,v), max(u,v))
            if not hasattr(generate_random_3reg, 'edge_set'):
                generate_random_3reg.edge_set = set()
            generate_random_3reg.edge_set.clear()
            
            # This approach clears the set every call, which is bad if called recursively.
            # Let's just build the list and check at the end.
            # Given the low probability of duplicates for large n, this is okay.
            edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
            
        if not valid:
            return None
            
        # Verify simplicity and regularity
        edge_set = set()
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None
            edge_set.add(key)
            
        for i in range(n):
            if len(adj[i]) != 3:
                return None
            if len(set(adj[i])) != 3:
                return None
                
        return adj

    def find_short_cycles(adj, n, g):
        """
        Find cycles of length < g.
        Returns a list of cycles (list of vertices).
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
                        # Found a back edge u-v
                        # Cycle length is dist[u] + dist[v] + 1
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
                            
                            # The cycle is path1 + path2 (reversed)
                            # But we need to be careful about the meeting point (start)
                            # Actually, since we stop at start in dist, parent chain goes to start.
                            # So path1 ends at start, path2 ends at start.
                            # Cycle: path1 (u to start) + path2 (v to start)
                            # Wait, path1 is u->...->start. path2 is v->...->start.
                            # The cycle is u->...->start->...->v->u?
                            # No, the edge is u-v.
                            # So cycle is u-v-...-start-...-u.
                            # path1: u, p(u), ..., start
                            # path2: v, p(v), ..., start
                            # Cycle: u, v, p(v), ..., start, p(u), ..., u
                            # Let's just record the length and the edge (u,v) to break.
                            # For 2-switch, we just need to pick two non-adjacent edges in the cycle.
                            # Simpler: just record the cycle vertices.
                            cycle = path1 + path2[:0:-1]
                            # Check if this cycle is already in list
                            # To avoid duplicates, we can canonicalize the cycle
                            # For now, just append. We will limit the number of cycles found.
                            if len(cycles) < 100:
                                cycles.append(cycle)
        return cycles

    def two_switch(adj, n, rng):
        """
        Perform a random 2-switch.
        Pick two non-adjacent edges (a,b) and (c,d) such that a,b,c,d are distinct
        and no new edges (a,c), (a,d), (b,c), (b,d) exist.
        Replace with (a,c), (b,d) or (a,d), (b,c).
        """
        # Collect all edges
        edges = []
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
                    edge_set.add((u, v))
        
        if len(edges) < 2:
            return False
            
        # Try to find a valid switch
        for _ in range(1000):
            i = rng.randint(0, len(edges)-1)
            j = rng.randint(0, len(edges)-1)
            if i == j:
                continue
            a, b = edges[i]
            c, d = edges[j]
            
            # Check distinct vertices
            if len({a, b, c, d}) < 4:
                continue
                
            # Check if new edges exist
            # Option 1: (a,c) and (b,d)
            if (min(a,c), max(a,c)) in edge_set or (min(b,d), max(b,d)) in edge_set:
                continue
            # Option 2: (a,d) and (b,c)
            if (min(a,d), max(a,d)) in edge_set or (min(b,c), max(b,c)) in edge_set:
                continue
            
            # Choose one option randomly
            if rng.random() < 0.5:
                new_e1 = (a, c)
                new_e2 = (b, d)
            else:
                new_e1 = (a, d)
                new_e2 = (b, c)
                
            # Apply switch
            # Remove old edges
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            
            # Add new edges
            adj[a].append(new_e1[1])
            adj[new_e1[1]].append(a)
            adj[b].append(new_e2[1])
            adj[new_e2[1]].append(b)
            
            return True
            
        return False

    # Main Strategy
    # 1. Start with a size slightly above Moore bound.
    # Moore bound for (3,16) is 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254.
    # But known best is 936. Random graphs need to be larger to have high girth.
    # Let's try sizes: 500, 600, 700, 800, 900, 1000...
    # We want the smallest n such that we can find a 3-reg graph with girth >= 16.
    
    best_n = float('inf')
    best_edges = None
    
    # Try a range of n values
    # Start from a reasonable lower bound. 
    # If n is too small, it's impossible.
    # Let's try n from 300 to 1200 in steps of 20? 
    # Or just try a few specific sizes.
    # The "smallest known" is 936. 
    # Let's try to find one with n < 936 if possible, or just find *a* valid one quickly.
    # The score is n. So we want min n.
    # Let's try n = 512 (lower bound), 600, 700, 800, 900, 1000.
    
    candidate_ns = []
    # Ensure n is even
    for n in range(300, 1200, 10):
        if n % 2 == 0:
            candidate_ns.append(n)
    
    # Sort by n to try smallest first? 
    # But smaller n is harder to find valid.
    # Maybe try a few random n in the range?
    # Let's try to find ANY valid graph first, then try to minimize?
    # No, time limit. 
    # Let's try n values in increasing order, but with a time limit per n.
    
    # Actually, random 3-reg graphs with girth 16 are rare for small n.
    # The probability that a random 3-reg graph has girth >= 2k is approx exp(-n * 2^{k-1} / 2^k ...)?
    # It's known that random 3-reg graphs have girth ~ log_2(n).
    # For girth 16, we need 2^16 ~ 65536 nodes? No.
    # Girth of random 3-reg graph on n nodes is approx 2 log_2(n).
    # So for girth 16, 2 log_2(n) >= 16 => log_2(n) >= 8 => n >= 256.
    # So n=300 should have a decent chance?
    # But "decent chance" might be 1 in 1000.
    # We have 140 seconds.
    # Generating a graph takes ~O(n). Checking takes ~O(n).
    # We can do thousands of attempts.
    
    # Let's try n from 400 upwards.
    # For each n, try up to 50 random graphs.
    # If found, record and try smaller n?
    # No, let's just iterate n from 400 to 1000.
    # If we find one at n=500, we can stop if we want the smallest?
    # But maybe n=450 works?
    
    # Strategy: 
    # 1. Iterate n from 400 to 1000 step 20.
    # 2. For each n, try 20 random graphs.
    # 3. If a valid graph is found, update best_n and best_edges.
    # 4. If time is running out, return best.
    
    # To speed up, we can use a faster girth check?
    # The BFS check is O(n^2) worst case? No, O(n+m) per start, so O(n(n+m)) = O(n^2).
    # For n=1000, n^2 = 1,000,000. Very fast.
    # So we can do many checks.
    
    for n in range(400, 1000, 20):
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
                
            rng = random.Random(int(time.time()*1000) + n + attempt)
            adj = generate_random_3reg(n, rng)
            if adj is None:
                continue
                
            if check_girth_fast(adj, n, g):
                # Convert adj to edges
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                            
                best_n = n
                best_edges = edges
                found = True
                break
                
        # If we found one, should we continue to find smaller?
        # We are iterating n in increasing order.
        # So if we find one at n=400, it's the smallest we will find in this loop.
        # But maybe n=380 works? We started at 400.
        # Let's start at 300.
        
    # If we didn't find any, try larger n?
    if best_edges is None:
        for n in range(1000, 2000, 50):
            if time.time() > deadline - 5:
                break
            for attempt in range(10):
                if time.time() > deadline - 2:
                    break
                rng = random.Random(int(time.time()*1000) + n + attempt)
                adj = generate_random_3reg(n, rng)
                if adj is None:
                    continue
                if check_girth_fast(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    break
            if best_edges:
                break

    # Fallback: If still None, use a known valid construction (LUW with large d)
    if best_edges is None:
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
