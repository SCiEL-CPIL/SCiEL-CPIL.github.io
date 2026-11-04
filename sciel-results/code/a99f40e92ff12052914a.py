import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses fast randomized local search with localized girth checks.
    """
    start_time = time.time()
    time_limit = 115.0
    
    # Moore bound for k=5, g=9 is ~428.
    # We try to find a graph with n close to this, but likely will find ~1200.
    # We start with a reasonable n and try to minimize.
    # However, finding a specific n is hard. 
    # Strategy: Pick a target n, generate random k-regular graph, improve girth.
    # If we can't get girth g, try a larger n? 
    # Actually, for a fixed n, if we can maximize girth, we might reach g.
    # For k=5, n=1000, max girth is around 9-10.
    # So we should try n in [800, 1500].
    
    # We will try a few values of n, starting from a known good range.
    # The known minimum is 1116. Let's try 1100, 1116, 1200, etc.
    # To be safe and fast, we will just pick a single n that is likely to work.
    # n = 1200 is a good bet for k=5, g=9.
    
    # Let's try to minimize n by iterating.
    # But we have only 115s.
    # We will try n = 1116 first (known min), then 1120, 1130...
    # Or just try a random n in a range and see if we can get girth 9.
    
    def random_regular_graph(n, k):
        """
        Generates a random k-regular graph using configuration model.
        Returns adjacency list (list of lists) or None if failed.
        """
        if (n * k) % 2 != 0:
            return None
            
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
            
        random.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        for i in range(n * k // 2):
            u = stubs[2*i]
            v = stubs[2*i+1]
            if u == v:
                # Self loop, retry this pair
                # Simple fix: swap with a later stub? 
                # For simplicity, we'll just retry the whole generation if we get self loops/multiedges
                # But for n=1000, k=5, self loops are rare.
                # Let's just return None and retry.
                return None
            if (u, v) in edge_set or (v, u) in edge_set:
                return None
            edge_set.add((u, v))
            edge_set.add((v, u))
            adj[u].append(v)
            adj[v].append(u)
            
        return adj

    def find_short_cycle(adj, n, g):
        """
        Finds a cycle of length < g.
        Returns list of vertices in the cycle, or None.
        Uses BFS from each node, but optimized.
        """
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= (g - 1) // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        # Reconstruct cycle
                        # Path start -> ... -> u -> v -> ... -> start
                        # We need to trace back from v and u to their LCA
                        # Simpler: just return the edge (u, v) and we can find the cycle later?
                        # For local search, we just need *a* short cycle.
                        # We can return the vertices involved.
                        # Let's just return the edge (u, v) and we will handle it in the swap.
                        # Actually, we need the whole cycle to pick an edge to remove?
                        # No, we can just pick the edge (u, v) and another random edge.
                        # But (u, v) is not necessarily in the cycle if we just found a back edge?
                        # Yes, (u, v) plus the tree paths forms a cycle.
                        return (u, v)
            # If no cycle found from this start, try next
        return None

    def has_short_cycle_fast(adj, n, g):
        """
        Checks if there is any cycle < g.
        """
        return find_short_cycle(adj, n, g) is not None

    def check_swap_validity(adj, n, g, e1, e2, new_e1, new_e2):
        """
        Checks if swapping e1, e2 for new_e1, new_e2 maintains girth >= g.
        e1, e2 are (u1, v1), (u2, v2).
        new_e1 is (u1, u2), new_e2 is (v1, v2) (or similar).
        We only need to check if new_e1 or new_e2 creates a short cycle.
        A short cycle using new_e1 (a, b) must have a path from a to b of length < g-1
        in the graph *without* new_e1 but *with* the rest of the changes?
        Actually, it's safer to apply the swap to a copy and check distances.
        But copying is expensive.
        
        Optimization:
        The only new cycles are those involving the new edges.
        A cycle involving (u1, u2) consists of (u1, u2) + path from u2 to u1.
        The path from u2 to u1 in the new graph must not use (u1, u2).
        It might use (v1, v2) if they are connected? No, (v1, v2) is a different edge.
        So we can check distance between u1 and u2 in the graph *after* removing e1, e2 and *before* adding new edges?
        No, the other new edge (v1, v2) might be part of the path.
        
        Safest fast check:
        1. Remove e1, e2 from adj.
        2. Add new_e1, new_e2 to adj.
        3. Check distance(u1, u2) < g-1? If yes, invalid.
        4. Check distance(v1, v2) < g-1? If yes, invalid.
        5. Also check if new_e1 or new_e2 creates a cycle with *other* edges.
           This is covered by the distance check if we consider the whole graph.
           But doing a full BFS for each swap is O(n*k).
           If we do 1000 swaps, that's 1000 * 1000 * 5 = 5M ops. Feasible.
        
        Let's implement `distance_less_than(adj, src, dst, limit)`.
        """
        # We pass the *new* adjacency list to this function?
        # Or we do the check on the modified list.
        pass

    # We will implement the local search loop.
    
    # Try a range of n values
    # Start with n = 1116 (known min) and go up if it fails.
    # But we only have time for one or two attempts.
    # Let's try n = 1200 as a safe bet.
    
    target_ns = [1116, 1120, 1140, 1200, 1300]
    
    for n in target_ns:
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random graphs for this n
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
                
            adj = random_regular_graph(n, k)
            if adj is None:
                continue
                
            # Local Search
            for step in range(5000):
                if time.time() - start_time > time_limit:
                    break
                    
                # Check girth
                cycle_info = find_short_cycle(adj, n, g)
                if cycle_info is None:
                    # Success!
                    # Build edge list
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
                    
                # If we have a short cycle, try to break it
                u, v = cycle_info
                # Find an edge to swap with
                # Pick a random edge from the whole graph
                # We need a list of all edges for random selection
                # To avoid rebuilding list every time, we can pick random u, v and check if edge exists?
                # Or maintain a list.
                # For n=1200, edges=3000. Rebuilding is fast.
                all_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            all_edges.append((i, j))
                
                # Try to swap (u, v) with a random edge (x, y)
                # We need to ensure (x, y) is not (u, v) or adjacent in a bad way?
                # Just pick random.
                for _ in range(10): # Try 10 random swaps
                    if not all_edges: break
                    x, y = random.choice(all_edges)
                    if (x, y) == (u, v) or (x, y) == (v, u):
                        continue
                    
                    # Try swap: remove (u, v) and (x, y). Add (u, x) and (v, y).
                    # Check validity
                    # 1. No self loops: u != x, v != y, u != y, v != x
                    if u == x or u == y or v == x or v == y:
                        continue
                    # 2. No multi-edges: (u, x) and (v, y) must not exist
                    if x in adj[u] or y in adj[v] or y in adj[u] or x in adj[v]:
                        continue
                        
                    # Perform swap on a copy? Or modify and revert?
                    # Modify and revert is faster.
                    
                    # Save state
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)
                    
                    # Check if this created a short cycle
                    # We only need to check cycles involving the new edges (u, x) and (v, y)
                    # Use fast distance check
                    valid = True
                    # Check (u, x)
                    # Is there a path from u to x of length < g-1?
                    # We can do a BFS from u to x with limit g-2.
                    # But the graph is modified.
                    # Let's just do a full check if step is small?
                    # No, let's do a limited BFS.
                    
                    def bfs_distance_limit(src, dst, limit):
                        if src == dst: return 0
                        dist = {src: 0}
                        queue = deque([src])
                        while queue:
                            curr = queue.popleft()
                            d = dist[curr]
                            if d >= limit:
                                continue
                            for nei in adj[curr]:
                                if nei not in dist:
                                    dist[nei] = d + 1
                                    if nei == dst:
                                        return d + 1
                                    queue.append(nei)
                        return -1 # Not found within limit
                    
                    # Check if dist(u, x) < g-1 (i.e., <= g-2)
                    if bfs_distance_limit(u, x, g - 2) != -1:
                        valid = False
                    if valid and bfs_distance_limit(v, y, g - 2) != -1:
                        valid = False
                            
                    if valid:
                        # Accept swap
                        continue_outer = True
                        break
                    else:
                        # Revert
                        adj[u].remove(x)
                        adj[x].remove(u)
                        adj[v].remove(y)
                        adj[y].remove(v)
                        
                        adj[u].append(v)
                        adj[v].append(u)
                        adj[x].append(y)
                        adj[y].append(x)
                        
            # End local search loop
            if time.time() - start_time > time_limit:
                break

    # Fallback if we didn't find a valid graph in the time limit
    # Use the seed construction (large graph)
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
    return len(order), edges
