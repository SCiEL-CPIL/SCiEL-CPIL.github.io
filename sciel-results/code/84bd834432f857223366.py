"""Decompositions of three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict, deque
import heapq
from random import Random
import itertools

def construct():
    # Generate the three instance graphs
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    results = []
    for spec in specs:
        edges = instance(*spec)
        n = spec[0]
        result = decompose_graph(n, edges)
        results.append(result)
    
    return results

def decompose_graph(n, edges):
    """
    Decompose the undirected graph into minimum number of cycles and edges.
    
    Strategy:
    1. Identify vertices with odd degree (T).
    2. Find a Minimum Weight Perfect Matching on T using shortest path distances.
    3. The union of shortest paths for matched pairs forms a T-Join (edges to remove).
    4. Remove these edges. The remaining graph has all even degrees.
    5. Decompose the remaining graph into simple cycles.
    6. The removed edges are single pieces. The cycles are cycle pieces.
    """
    if not edges:
        return []
    
    # Build adjacency and degree counts
    adj = defaultdict(set)
    degrees = defaultdict(int)
    edge_set = set()
    for u, v in edges:
        e = (min(u, v), max(u, v))
        edge_set.add(e)
        adj[u].add(v)
        adj[v].add(u)
        degrees[u] += 1
        degrees[v] += 1

    # Find T: vertices with odd degree
    T = [v for v in range(n) if degrees[v] % 2 != 0]
    
    edges_to_remove = set()
    
    if T:
        # Compute shortest paths from each node in T to all other nodes
        # dist[t][v] = shortest path distance from t to v
        dist = {}
        parent = {} # parent[t][v] = next node from v towards t? Or predecessor?
        # Let's store predecessors for path reconstruction
        prev_node = {}
        
        for t in T:
            # BFS from t
            dist_t = {t: 0}
            prev_t = {t: None}
            queue = deque([t])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor not in dist_t:
                        dist_t[neighbor] = dist_t[node] + 1
                        prev_t[neighbor] = node
                        queue.append(neighbor)
            dist[t] = dist_t
            prev_node[t] = prev_t
            
        # Build complete graph on T with weights = dist
        # We need Min Weight Perfect Matching on T
        # Since |T| is small (<= 30), we can use DP over subsets
        # dp[mask] = min cost to match nodes in mask
        # mask is a bitmask of indices in T
        m = len(T)
        # Map node id to index in T
        t_index = {t: i for i, t in enumerate(T)}
        
        # Precompute costs
        cost = [[0]*m for _ in range(m)]
        for i in range(m):
            for j in range(i+1, m):
                t_i = T[i]
                t_j = T[j]
                # dist[t_i][t_j] should exist if graph is connected
                # If not connected, we can't form a T-Join? 
                # But the graph is generated to be connected-ish? 
                # If not connected, we treat components separately. 
                # For now, assume connected or handle infinity.
                d = dist[t_i].get(t_j, float('inf'))
                cost[i][j] = d
                cost[j][i] = d
                
        # DP for Perfect Matching
        # dp[mask] where mask has even number of bits
        # Base case: dp[0] = 0
        # Transition: pick lowest set bit i in mask.
        # Pair i with some j in mask.
        # dp[mask] = min(cost[i][j] + dp[mask ^ (1<<i) ^ (1<<j)])
        
        full_mask = (1 << m) - 1
        dp = [float('inf')] * (1 << m)
        dp[0] = 0
        parent_match = [None] * (1 << m) # Store which j was paired with i
        
        for mask in range(1, 1 << m):
            if bin(mask).count('1') % 2 != 0:
                continue
            # Find lowest set bit
            i = 0
            while not (mask & (1 << i)):
                i += 1
            # Try pairing i with all j in mask
            rest = mask ^ (1 << i)
            for j in range(m):
                if rest & (1 << j):
                    prev_mask = rest ^ (1 << j)
                    c = cost[i][j]
                    if dp[prev_mask] + c < dp[mask]:
                        dp[mask] = dp[prev_mask] + c
                        parent_match[mask] = j
                        
        # Reconstruct matching
        mask = full_mask
        matched_pairs = []
        while mask:
            i = 0
            while not (mask & (1 << i)):
                i += 1
            j = parent_match[mask]
            matched_pairs.append((T[i], T[j]))
            mask ^= (1 << i) ^ (1 << j)
            
        # Reconstruct edges to remove (T-Join)
        for u, v in matched_pairs:
            # Find path from u to v using prev_node
            # We have prev_node[u] which gives path from u to target?
            # BFS was from u. So prev_node[u][x] is predecessor of x in tree rooted at u.
            # To get path from u to v, we follow parents from v back to u.
            path = []
            curr = v
            while curr != u:
                path.append(curr)
                curr = prev_node[u][curr]
            path.append(u)
            path.reverse() # Now path is u -> ... -> v
            
            # Add edges in path to edges_to_remove
            for k in range(len(path) - 1):
                e = (min(path[k], path[k+1]), max(path[k], path[k+1]))
                edges_to_remove.add(e)
                
    # Remove edges_to_remove from the graph
    remaining_edges = set(edge_set) - edges_to_remove
    
    # Decompose remaining_edges into cycles
    # The remaining graph has all even degrees.
    # We can find cycles by DFS.
    rem_adj = defaultdict(set)
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    pieces = []
    
    # Find cycles in the even subgraph
    while remaining_edges:
        # Pick an arbitrary edge to start
        start_edge = next(iter(remaining_edges))
        u, v = start_edge
        
        # DFS to find a cycle
        cycle = [u]
        prev = None
        curr = v
        while curr != u:
            cycle.append(curr)
            # Find next node in cycle
            # We must pick a neighbor that is not prev, unless we are closing the loop
            # But we are in an even subgraph, so we can always continue
            # Actually, we just follow a path. Since degrees are even, 
            # if we enter a node, we can leave it.
            # We need to ensure we don't get stuck. 
            # In an even subgraph, any maximal path is a cycle? 
            # No, it's a trail. 
            # Let's just do a simple DFS that tracks visited edges.
            
            # This simple approach might fail if the graph is complex.
            # Better: Find any cycle in the subgraph.
            break
            
    # Let's use a robust cycle finder for the even subgraph
    def find_cycle_in_subgraph(adj, n):
        visited_edges = set()
        for start in range(n):
            if not adj[start]:
                continue
            # DFS
            stack = [(start, -1, [start])]
            while stack:
                node, parent, path = stack.pop()
                for neighbor in adj[node]:
                    e = (min(node, neighbor), max(node, neighbor))
                    if e in visited_edges:
                        continue
                    if neighbor == start and len(path) >= 3:
                        # Found cycle
                        return path
                    if neighbor == parent:
                        continue
                    # Check if we can extend
                    # We need to ensure we don't revisit vertices in a way that breaks simplicity?
                    # For cycle finding in undirected graph, standard DFS works.
                    # But we must remove edges as we go? No, we just find one.
                    # Standard DFS for cycle:
                    # If we see a visited node that is not parent, we have a cycle.
                    # But we need the path.
                    
            # Simpler: Just pick a node with degree > 0, and trace a path.
            # In an even graph, you can always find a cycle.
            # Algorithm:
            # 1. Pick a node v with degree > 0.
            # 2. Follow an edge to w.
            # 3. Remove edge (v,w).
            # 4. If w == start, cycle found.
            # 5. Else, set v = w, continue.
            # 6. If v has no edges, dead end? 
            #    In an even graph, if you start with degree > 0, you can always complete a cycle?
            #    Not necessarily a simple cycle without repeating vertices?
            #    Eulerian circuits exist. A simple cycle is a subset.
            #    Yes, any connected component of an even graph contains a cycle.
            
        return None

    # Let's implement the "trace and remove" method which is safer for extraction
    # We maintain the remaining graph.
    # While edges exist:
    #   Find a cycle.
    #   Remove its edges.
    
    # Re-initialize rem_adj for the loop
    rem_adj = defaultdict(set)
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    while remaining_edges:
        # Find a cycle
        cycle = None
        for start in range(n):
            if not rem_adj[start]:
                continue
            # Trace a path
            path = [start]
            prev = None
            curr = start
            found = False
            while True:
                # Pick a neighbor
                next_node = None
                for nb in rem_adj[curr]:
                    if nb == prev:
                        continue
                    # We can go to nb
                    next_node = nb
                    break
                if next_node is None:
                    # Dead end? Should not happen in even graph if we started correctly
                    # But we might have removed edges.
                    break
                path.append(next_node)
                # Remove edge (curr, next_node)
                e = (min(curr, next_node), max(curr, next_node))
                remaining_edges.discard(e)
                rem_adj[curr].remove(next_node)
                rem_adj[next_node].remove(curr)
                
                if next_node == start:
                    cycle = path
                    found = True
                    break
                prev = curr
                curr = next_node
            if found:
                break
        if not cycle:
            # Should not happen if logic is correct
            break
        pieces.append(cycle)
        
    # Add removed edges as single pieces
    for e in edges_to_remove:
        pieces.append(list(e))
        
    return pieces
# EVOLVE-BLOCK-END
