import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    # Total possible edges
    all_edges = list(combinations(range(n), 2))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Generate all subgraphs isomorphic to Q3 (3-cube)
    # Q3 has 8 vertices and 12 edges.
    # Vertices of Q3 can be represented as binary strings of length 3.
    # Two vertices are adjacent if they differ in exactly 1 bit.
    q3_vertices = list(range(8))
    q3_edges = []
    for u in range(8):
        for v in range(u + 1, 8):
            # Check if Hamming distance is 1
            diff = bin(u ^ v).count('1')
            if diff == 1:
                q3_edges.append((u, v))
    
    # q3_edges has 12 edges.
    # We need to find all injective mappings from Q3 vertices to 16 vertices
    # such that the edges of Q3 map to edges in the candidate graph.
    # Since the graph is simple and undirected, the order of vertices in the mapping
    # doesn't matter for the set of edges, but the structure does.
    # Actually, we just need the set of 12 edges in the 16-vertex graph that forms a Q3.
    
    # To speed up, we can pre-compute all sets of 12 edges that form a Q3.
    # Number of embeddings:
    # Choose 8 vertices from 16: C(16, 8) = 12870
    # For each set of 8 vertices, how many ways to label them as Q3?
    # Q3 has 8! / 8 (automorphisms) = 40320 / 8 = 5040 labelings?
    # Wait, the automorphism group of Q3 has size 48 (hyperoctahedral group B3).
    # So 8! / 48 = 40320 / 48 = 840 distinct subgraphs per set of 8 vertices.
    # Total potential Q3s: 12870 * 840 = ~10.8 million.
    # This is too many to store explicitly if we check all of them for every move.
    # However, we can use a different strategy.
    
    # Strategy:
    # 1. Start with a dense graph (e.g., K16).
    # 2. Identify Q3s present in the current graph.
    # 3. Remove edges that are part of many Q3s.
    # 4. Repeat until no Q3s remain.
    # 5. Then try to add edges back if possible (greedy).
    
    # Efficient Q3 detection:
    # Instead of pre-computing all 10M Q3s, we can check for Q3s on the fly.
    # But checking all 12870 sets of 8 vertices for each move is expensive.
    
    # Alternative:
    # Use a heuristic based on degrees and local structure.
    # Or, pre-compute a smaller set of "critical" Q3s?
    
    # Let's try the greedy removal approach with on-the-fly Q3 detection.
    # To make it fast, we can sample or use a heuristic to find Q3s.
    # Actually, let's try a different structural approach first to get a good seed.
    
    # Seed: K7, K7, K2 (from current program) -> 21 + 21 + 1 = 43 edges.
    # Is this Q3-free?
    # Q3 is bipartite. K7 is not bipartite.
    # Any Q3 subgraph must be bipartite.
    # In K7, any 4 vertices form a K4, which contains odd cycles.
    # A Q3 is bipartite, so it cannot contain any odd cycle.
    # Therefore, a Q3 cannot have all its vertices in a clique of size >= 4?
    # Wait. A Q3 is a bipartite graph. If we have a clique K4, it contains triangles.
    # A bipartite graph cannot contain triangles.
    # So, any subgraph of K4 is not a Q3 (since Q3 has no triangles).
    # However, a Q3 can have vertices from multiple cliques.
    
    # Let's verify if K7 U K7 U K2 is Q3-free.
    # Vertices 0-6 in K7, 7-13 in K7, 14-15 in K2.
    # No edges between these groups.
    # A Q3 has 8 vertices.
    # By Pigeonhole Principle, if we pick 8 vertices from 3 groups of sizes 7, 7, 2:
    # Case 1: All 8 in one group? Impossible (max 7).
    # Case 2: 7 in one, 1 in another.
       # The 7 vertices in K7 form a K7. The 1 vertex in the other group is isolated from them.
       # The induced subgraph on these 8 vertices is K7 + isolated vertex.
       # This graph is not connected. Q3 is connected. So no Q3.
    # Case 3: 6 in one, 2 in another.
       # Induced subgraph: K6 + K2 (or K1+K1). Disconnected. No Q3.
    # Case 4: 6 in one, 1 in second, 1 in third.
       # Disconnected.
    # Case 5: 5 in one, 3 in another... all disconnected.
    
    # Since there are no edges between the groups, any subgraph with vertices from multiple groups is disconnected (unless we add edges).
    # Q3 is connected.
    # Therefore, K7 U K7 U K2 is Q3-free.
    # Edges: 21 + 21 + 1 = 43.
    
    # Can we do better?
    # What if we connect the groups slightly?
    # If we add an edge between K7 and K7, we might create a Q3.
    
    # Let's try to improve 43 using local search.
    
    # Initialize with K7, K7, K2
    adj = [[0]*n for _ in range(n)]
    edges = set()
    
    def add_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        if (u, v) not in edges:
            edges.add((u, v))
            adj[u][v] = 1
            adj[v][u] = 1
    
    def remove_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        if (u, v) in edges:
            edges.remove((u, v))
            adj[u][v] = 0
            adj[v][u] = 0
    
    # Build initial graph
    for i in range(7):
        for j in range(i+1, 7):
            add_edge(i, j)
    for i in range(7, 14):
        for j in range(i+1, 14):
            add_edge(i, j)
    add_edge(14, 15)
    
    best_edges = len(edges)
    best_adj = [row[:] for row in adj]
    
    # Function to check if a set of 8 vertices contains a Q3
    # This is expensive. We need a faster way.
    # Heuristic: A Q3 is bipartite with 4+4 partition.
    # Check if the induced subgraph on 8 vertices is bipartite and has exactly 12 edges?
    # No, Q3 has exactly 12 edges.
    # So, check if induced subgraph on 8 vertices is isomorphic to Q3.
    
    # To speed up, we can pre-compute the signature of Q3.
    # Degree sequence of Q3: all 8 vertices have degree 3.
    # So, any induced subgraph that is 3-regular on 8 vertices and bipartite is a Q3?
    # Yes, the only 3-regular bipartite graph on 8 vertices is Q3.
    
    def is_q3(vertices):
        # vertices is a tuple of 8 distinct integers
        # Check if induced subgraph is 3-regular and bipartite
        # 1. Check degrees within the subgraph
        for v in vertices:
            deg = 0
            for u in vertices:
                if u != v and adj[v][u]:
                    deg += 1
            if deg != 3:
                return False
        
        # 2. Check if bipartite
        # BFS/DFS
        visited = {}
        for v in vertices:
            visited[v] = 0
        for start in vertices:
            if visited[start] == 0:
                queue = [start]
                visited[start] = 1
                while queue:
                    u = queue.pop(0)
                    for v in vertices:
                        if u != v and adj[u][v]:
                            if visited[v] == 0:
                                visited[v] = 1 - visited[u]
                                queue.append(v)
                            elif visited[v] == visited[u]:
                                return False
        return True
    
    # This is still slow if called many times.
    # Let's try to find Q3s by iterating over combinations of 8 vertices.
    # There are 12870 combinations.
    # Checking each takes ~8*8 operations.
    # 12870 * 64 ~ 800k ops.
    # This is fast enough for a few iterations.
    
    def find_q3s():
        q3s = []
        for combo in combinations(range(n), 8):
            if is_q3(combo):
                q3s.append(combo)
        return q3s
    
    # Greedy removal
    # While there are Q3s, remove an edge that is in the most Q3s.
    
    while time.time() < deadline:
        q3s = find_q3s()
        if not q3s:
            break
        
        # Count edge occurrences in Q3s
        edge_count = {}
        for q3 in q3s:
            for u, v in combinations(q3, 2):
                if adj[u][v]:
                    e = (min(u, v), max(u, v))
                    edge_count[e] = edge_count.get(e, 0) + 1
        
        # Find edge with max count
        if not edge_count:
            break
            
        max_count = max(edge_count.values())
        candidates = [e for e, c in edge_count.items() if c == max_count]
        
        # Pick one randomly or deterministically
        e = candidates[0]
        remove_edge(e[0], e[1])
        
        if len(edges) < best_edges:
            best_edges = len(edges)
            best_adj = [row[:] for row in adj]
    
    # Now try to add edges back
    # Iterate through missing edges, try to add if it doesn't create a Q3.
    missing_edges = [e for e in all_edges if e not in edges]
    random.shuffle(missing_edges)
    
    for e in missing_edges:
        if time.time() > deadline:
            break
        u, v = e
        add_edge(u, v)
        # Check if this created a Q3
        # Only need to check Q3s containing this edge?
        # Or just run find_q3s again?
        # find_q3s is ~0.1s?
        # Let's check.
        
        # Optimization: Only check combinations of 8 vertices that include u and v.
        # Number of such combos: C(14, 6) = 3003.
        # Much faster.
        
        created_q3 = False
        for combo in combinations([x for x in range(n) if x not in (u, v)], 6):
            verts = tuple(list(combo) + [u, v])
            if is_q3(verts):
                created_q3 = True
                break
        
        if created_q3:
            remove_edge(u, v)
        else:
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_adj = [row[:] for row in adj]
    
    # Restore best graph
    adj = best_adj
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.add((i, j))
    
    # Convert to list of lists
    result = [[u, v] for u, v in edges]
    return result
