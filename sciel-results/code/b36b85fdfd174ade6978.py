"""
Decompose three fixed graphs into cycles and edges, minimizing piece count.
Strategy: 2-Factorization Heuristic.
1. Find a maximum-cardinality subgraph H where all degrees are even.
   This is done by finding a maximum matching M in the graph.
   The edges in H are E(G) \ M.
   Since deg_H(v) = deg_G(v) - 1 (if matched) or deg_G(v) (if unmatched),
   and the parity of deg_G(v) determines the structure, 
   actually:
   If we take a matching M, let H = (V, E \ M).
   deg_H(v) = deg_G(v) - 1 if v is matched in M, else deg_G(v).
   This doesn't guarantee even degrees.
   
   Correction: 
   We want a subgraph H such that all degrees in H are even.
   This is equivalent to finding a subset of edges F (the "removed" edges) 
   such that deg_F(v) has the same parity as deg_G(v) for all v.
   The minimal such F is a minimum T-join, where T = {v | deg_G(v) is odd}.
   For a graph, the size of a minimum T-join is at most |T|/2 * (diameter).
   However, finding the exact minimum T-join is complex.
   
   Simpler Heuristic for "Even Subgraph":
   We can greedily remove edges to make degrees even? No.
   
   Let's use the property:
   Any graph G can be decomposed into a collection of cycles and a forest (or just paths).
   Specifically, if we find a matching M, and we remove M from G, do we get even degrees?
   No.
   
   Let's look at the parity argument.
   Let T be the set of odd-degree vertices.
   We need to remove a set of edges F such that the parity of degrees in G-F is 0.
   This means deg_F(v) ≡ deg_G(v) (mod 2).
   So F must be a T-join (a subgraph where the odd-degree vertices are exactly T).
   The smallest T-join minimizes the number of edges removed (which become single edge pieces).
   The remaining edges H = E \ F form a graph with all even degrees.
   A graph with all even degrees decomposes entirely into cycles.
   The number of cycles in H is at most |E(H)|/3 (if all triangles) or roughly |E(H)|/avg_cycle_len.
   The number of edge pieces is |F|.
   Total pieces = (# cycles in H) + |F|.
   
   To minimize pieces, we want |F| small (so H is large) AND the cycles in H to be long.
   The minimum T-join F consists of disjoint paths connecting pairs of vertices in T.
   If we choose F to be a set of shortest paths between pairs of odd vertices, |F| is minimized.
   However, the cycle structure of H depends on how we pair them.
   
   Alternative Strategy:
   1. Identify all odd-degree vertices T.
   2. Find a minimum T-join F. (Approximate with greedy shortest paths).
   3. H = G - F. H has all even degrees.
   4. Decompose H into cycles. To minimize the number of cycles, we want long cycles.
      We can use a "greedy longest cycle" removal on H.
      Since H has even degrees, it is Eulerian.
      We can find an Eulerian tour and split it into cycles? No, that gives 1 cycle if connected.
      Wait, if H is connected and all degrees even, it has an Eulerian circuit.
      But we need a decomposition into *simple* cycles.
      An Eulerian circuit is a closed walk. We can decompose any closed walk into simple cycles.
      The number of simple cycles in the decomposition depends on the structure.
      
      Actually, the most efficient decomposition of an Eulerian graph into cycles 
      in terms of minimizing the number of cycles is not trivial, but 
      extracting cycles greedily (e.g., longest first) is a good heuristic.
      
      Let's refine:
      1. Compute T-join F (minimize |F|).
      2. H = G - F.
      3. Decompose H into simple cycles.
         - While H has edges:
           - Find a cycle in H (preferably long).
           - Remove it.
      4. Pieces = Cycles(H) + Edges(F).
      
   Why is this better than the current approach?
   The current approach picks cycles from G directly. If G has odd degrees, 
   it will inevitably leave behind edges that form paths, which are then treated as single edges.
   The T-join approach explicitly isolates the "odd" part into F, ensuring H is perfectly decomposable into cycles.
   This avoids the "fringe" problem where the greedy cycle finder gets stuck in a part of the graph 
   with odd degrees and leaves many single edges.
"""

import time
import random
from collections import defaultdict, deque
import heapq

_DEADLINE_SECONDS = 140.0 # Safety margin for 150s limit

def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _shortest_paths_bfs(graph, n):
    """
    Compute shortest path distances and next hops from all nodes.
    Returns dist[i][j] and next_hop[i][j].
    """
    dist = [[float('inf')] * n for _ in range(n)]
    next_hop = [[-1] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
        # BFS from i
        q = deque([i])
        while q:
            u = q.popleft()
            for v in graph[u]:
                if dist[i][v] > dist[i][u] + 1:
                    dist[i][v] = dist[i][u] + 1
                    next_hop[i][v] = u
                    q.append(v)
    return dist, next_hop

def _find_t_join(dist, next_hop, T, n):
    """
    Greedily find a T-join by pairing up vertices in T using shortest paths.
    Returns a set of edges forming the T-join.
    """
    if not T:
        return set()
    
    # Sort T vertices to make it deterministic, or just use list
    # A better heuristic: Pair closest pairs first (Greedy Matching on T)
    # For small T (<= 30), we can even do DP for exact minimum weight perfect matching on T
    # But T size is at most n. n=30 is small enough for O(|T|^3) Hungarian or DP.
    # Let's use a simple greedy: sort all pairs (u,v) in T by distance, pick non-conflicting.
    
    T_list = list(T)
    m = len(T_list)
    
    # Map vertex to index in T_list for fast lookup?
    # We need to find min weight perfect matching on T_list with weights dist[u][v].
    
    if m < 2:
        return set()
    
    # DP for Minimum Weight Perfect Matching on small set
    # State: bitmask of used vertices in T_list
    # dp[mask] = min cost to match vertices in mask
    # This is O(2^m * m^2). If m=30, 2^30 is too big.
    # However, in random graphs, the number of odd vertices is usually small?
    # No, in a random graph G(n,p), the number of odd vertices is roughly binomial(n, ...).
    # Expected number of odd vertices is not necessarily small.
    # But wait, the sum of degrees is 2|E|. The number of odd vertices is even.
    # For n=30, m can be up to 30. 2^30 is ~1e9, too slow for Python in 150s if done naively.
    
    # Fallback: Greedy pairing.
    # Generate all pairs, sort by distance, pick greedily.
    pairs = []
    for i in range(m):
        for j in range(i+1, m):
            u = T_list[i]
            v = T_list[j]
            d = dist[u][v]
            if d < float('inf'):
                pairs.append((d, i, j))
    
    pairs.sort()
    
    used = [False] * m
    selected_pairs = []
    
    for d, i, j in pairs:
        if not used[i] and not used[j]:
            used[i] = True
            used[j] = True
            selected_pairs.append((T_list[i], T_list[j]))
            
    # Reconstruct edges
    F_edges = set()
    for u, v in selected_pairs:
        # Trace path from u to v using next_hop
        # next_hop[u][v] gives the first step from u towards v?
        # In our BFS: next_hop[i][v] = u where u is the predecessor of v on shortest path from i?
        # Let's re-verify BFS logic.
        # dist[i][v] = dist[i][u] + 1. u is neighbor of v.
        # So next_hop[i][v] = u.
        # To get path i -> ... -> v:
        # cur = v
        # path = [v]
        # while cur != i:
        #   cur = next_hop[i][cur]
        #   path.append(cur)
        # path is reversed.
        
        path = []
        cur = v
        while cur != u:
            path.append(cur)
            cur = next_hop[u][cur]
            if cur == -1: # Should not happen if dist < inf
                break
        path.append(u)
        path.reverse()
        
        for k in range(len(path)-1):
            a, b = path[k], path[k+1]
            F_edges.add((min(a,b), max(a,b)))
            
    return F_edges

def _decompose_eulerian_to_cycles(adj, n, time_limit):
    """
    Decompose a graph with all even degrees into simple cycles.
    Uses a greedy approach: find a cycle, remove it, repeat.
    To find a cycle, we can use DFS or just follow edges in an Eulerian trail manner
    and extract simple cycles.
    
    Efficient method for Eulerian graph:
    1. While edges exist:
       - Start at any node with deg > 0.
       - Follow edges (Eulerian path logic) until we return to start? 
         No, that gives a closed walk.
       - A closed walk can be decomposed into simple cycles.
       - Alternatively, just do DFS to find ANY simple cycle.
       
       Given the small size (n<=30), DFS to find a cycle is fast.
       We prefer longer cycles to minimize count.
       Heuristic: Try to find the longest cycle? NP-hard.
       Heuristic: Greedy DFS with backtracking to maximize length?
       Let's just find a cycle using Hierholzer's algorithm logic but extracting simple cycles.
       
       Actually, if we just run Hierholzer's algorithm, it produces one Eulerian circuit 
       if the graph is connected. 
       But we need a decomposition into simple cycles.
       Any Eulerian circuit can be decomposed into simple cycles.
       The number of simple cycles in such a decomposition is at most |E|/3?
       
       Better:
       While edges exist:
         1. Find a simple cycle.
            - We can use a randomized DFS or just standard DFS.
            - To bias towards longer cycles, we can try to avoid closing the loop too early?
            - Simple DFS finds *a* cycle.
         2. Remove edges of the cycle.
         
       Let's implement a cycle finder that tries to find a long cycle.
       Since n is small, we can even try to find a Hamiltonian cycle in the remaining subgraph?
       No, too expensive.
       
       Standard DFS cycle finding is O(V+E).
    """
    pieces = []
    start_time = time.time()
    
    # Make a mutable copy of adjacency
    # adj is list of sets
    remaining_edges = sum(len(s) for s in adj) // 2
    
    while remaining_edges > 0:
        if time.time() - start_time > time_limit:
            break
            
        # Find a node with degree > 0
        start_node = -1
        for i in range(n):
            if adj[i]:
                start_node = i
                break
        
        if start_node == -1:
            break
            
        # Find a cycle starting/ending at start_node
        # DFS
        path = [start_node]
        visited = {start_node}
        cycle_found = None
        
        # Iterative DFS
        stack = [(start_node, [start_node])]
        # To find a specific cycle, we need to track the path properly.
        # Standard DFS for cycle detection:
        # If we hit a visited node that is in the current path, we have a cycle.
        
        # Let's use a recursive-like structure with explicit stack
        # Stack items: (node, path_so_far)
        
        found = False
        # To encourage longer cycles, we can shuffle neighbors?
        # Or just pick the first one.
        
        # Optimization: Since we want to minimize number of pieces, we want long cycles.
        # A simple heuristic: In the DFS, prefer neighbors that have not been visited 
        # and have high degree?
        
        # Let's just do a standard DFS to find ANY cycle.
        # The length distribution will be whatever the graph structure allows.
        
        # Reset for this search
        stack = [(start_node, [start_node])]
        visited_in_path = {start_node}
        
        while stack and not found:
            node, path = stack.pop()
            if node == start_node and len(path) >= 3:
                # This logic is flawed for simple DFS cycle finding.
                # We need to detect back edges to ancestors.
                pass
        
        # Let's use a simpler, robust cycle finder:
        # Since the graph is small, we can try all starting edges?
        # No, let's use a proper DFS that tracks the path.
        
        def find_cycle(start):
            stack = [(start, [start])]
            # We need to know which nodes are in the current stack to detect cycles
            # Actually, in an undirected graph, if we visit a neighbor that is already in the path, 
            # we found a cycle.
            
            # To make it efficient, we just need to find one.
            # We'll use a set for O(1) lookup of path membership.
            # But path changes as we backtrack.
            
            # Simpler: 
            # For each edge (u,v), try to find a path from v to u in G - (u,v).
            # If found, (u,v) + path is a cycle.
            
            # This is O(E * (V+E)). With V=30, E~100, this is 100*130 = 13000 ops. Very fast.
            
            for u, v in list(adj[start]): # Just pick one edge from start? 
                 # No, we need to find a cycle in the whole graph.
                 # Let's iterate over all edges in the remaining graph.
                 pass
            
            # Let's restart with the "edge removal" strategy for cycle finding.
            # Pick an arbitrary edge (u,v) from the remaining graph.
            # Remove it.
            # Find path from u to v in the remaining graph.
            # If found, Cycle = Path + (u,v).
            
            # This guarantees a simple cycle.
            # To make it "long", we can pick the edge that maximizes the path length?
            # Or just pick any.
            
            # Let's implement this BFS-based cycle finder.
            
            # 1. Pick an edge (u, v) present in adj.
            edge_u = -1
            edge_v = -1
            for u in range(n):
                if adj[u]:
                    edge_u = u
                    edge_v = next(iter(adj[u]))
                    break
            
            if edge_u == -1:
                return None
            
            # 2. Remove edge (u,v) temporarily
            adj[edge_u].discard(edge_v)
            adj[edge_v].discard(edge_u)
            
            # 3. Find path from edge_u to edge_v in remaining graph
            # BFS
            q = deque([(edge_u, [edge_u])])
            visited = {edge_u}
            path_found = None
            
            while q:
                node, path = q.popleft()
                if node == edge_v:
                    path_found = path
                    break
                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append((nb, path + [nb]))
                        
            # 4. Restore edge
            adj[edge_u].add(edge_v)
            adj[edge_v].add(edge_u)
            
            if path_found:
                # Cycle is path_found + [edge_v] (which connects to edge_u)
                # path_found is [edge_u, ..., edge_v]
                # Cycle vertices: path_found
                # Edges: (path[i], path[i+1]) for i in 0..len-2, and (path[-1], path[0])
                return path_found
            else:
                # No path? Then (u,v) was a bridge. 
                # But in an even-degree graph, there are no bridges?
                # Wait, a graph with all even degrees can have bridges?
                # No. If (u,v) is a bridge, removing it disconnects the graph.
                # Let the component containing u have sum of degrees = 2*E_comp.
                # deg(u) in G is even. deg(u) in G-(u,v) is odd.
                # Sum of degrees in component = sum(deg(v) - 1 if v==u else deg(v))
                # = Sum(deg(v)) - 1.
                # Sum(deg(v)) is even. So Sum(deg(v))-1 is odd.
                # But sum of degrees must be even. Contradiction.
                # So no bridges in an even-degree graph.
                # So path_found should always exist.
                return None

        cycle = find_cycle(start_node)
        
        if cycle:
            pieces.append(cycle)
            # Remove edges of cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                adj[u].discard(v)
                adj[v].discard(u)
            remaining_edges -= len(cycle)
        else:
            # Should not happen if logic is correct, but safety break
            break
            
    return pieces

def _decompose_graph(edges, n, time_limit):
    """
    Main decomposition using T-join strategy.
    """
    if not edges:
        return []
        
    # 1. Build Graph
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        
    # 2. Find Odd Vertices
    T = [i for i in range(n) if deg[i] % 2 != 0]
    
    # 3. Compute Shortest Paths (for T-join)
    # Only needed if T is not empty
    if T:
        # We need distances between all pairs in T.
        # BFS from each node in T is sufficient? 
        # No, the T-join paths might go through nodes not in T.
        # So we need full all-pairs shortest paths or at least from T nodes to all nodes.
        # BFS from each node in T is O(|T| * (V+E)).
        # |T| <= 30. V=30, E~100. 30 * 130 = 3900 ops. Very fast.
        
        dist = {}
        next_hop = {}
        
        for t in T:
            # BFS from t
            d = [float('inf')] * n
            nh = [-1] * n
            d[t] = 0
            q = deque([t])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if d[v] > d[u] + 1:
                        d[v] = d[u] + 1
                        nh[v] = u
                        q.append(v)
            dist[t] = d
            next_hop[t] = nh
            
        # 4. Find T-join F
        # Use the greedy pairing based on distances
        # We need dist[u][v] for u,v in T.
        # dist is indexed by source. dist[u][v] gives distance from u to v.
        
        # Construct the T-join edge set
        F_edges = _find_t_join(dist, next_hop, T, n)
        
        # 5. Build H = G - F
        # H has all even degrees.
        adj_h = defaultdict(set)
        for u in range(n):
            for v in adj[u]:
                if (min(u,v), max(u,v)) not in F_edges:
                    adj_h[u].add(v)
                    
        # 6. Decompose H into cycles
        # Convert adj_h to list of sets for the decomposer
        adj_h_list = [set(adj_h[i]) for i in range(n)]
        
        cycles = _decompose_eulerian_to_cycles(adj_h_list, n, time_limit * 0.5)
        
        # 7. Convert F_edges to pieces
        edge_pieces = [[u, v] for u, v in F_edges]
        
        return cycles + edge_pieces
    else:
        # All degrees even. H = G.
        adj_g_list = [set(adj[i]) for i in range(n)]
        cycles = _decompose_eulerian_to_cycles(adj_g_list, n, time_limit)
        return cycles

def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 5:
            # Fallback: just return all edges
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        
        # Allocate time for this graph
        # We have 3 graphs. Let's give each 1/3 of remaining time, with buffer.
        graph_time = remaining * 0.3 # Generous buffer
        
        pieces = _decompose_graph(edges, n, graph_time)
        results.append(pieces)
        
    return results

# Verification helper (not used in construct, but good for sanity check if run locally)
def _verify(pieces, edges, n):
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    used_edges = set()
    for piece in pieces:
        if len(piece) == 2:
            u, v = piece
            e = (min(u,v), max(u,v))
            if e in used_edges:
                return False
            if e not in edge_set:
                return False
            used_edges.add(e)
        else:
            if len(piece) < 3:
                return False
            # Check distinct vertices
            if len(set(piece)) != len(piece):
                return False
            for i in range(len(piece)):
                u = piece[i]
                v = piece[(i+1)%len(piece)]
                e = (min(u,v), max(u,v))
                if e in used_edges:
                    return False
                if e not in edge_set:
                    return False
                used_edges.add(e)
    if used_edges != edge_set:
        return False
    return True

if __name__ == "__main__":
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose_graph(edges, n, 10.0)
        valid = _verify(pieces, edges, n)
        print(f"n={n}, p={p}, seed={seed}: {len(pieces)} pieces, valid={valid}")
        # Print piece breakdown
        cycles = [p for p in pieces if len(p) > 2]
        edges_p = [p for p in pieces if len(p) == 2]
        print(f"  Cycles: {len(cycles)}, Edges: {len(edges_p)}")
        if cycles:
            print(f"  Cycle lengths: {[len(c) for c in cycles]}")

