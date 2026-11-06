"""Construct a dense Q3-free graph on 16 vertices via heuristic edge removal."""
import time
import random
import itertools

DEADLINE = 115.0  # seconds, safety margin under 120s / 150s kill

def _has_q3(adj):
    """
    Check if the graph (given as adjacency bitmasks, list of 16 ints)
    contains a subgraph isomorphic to Q3 (the 3-cube).
    
    Q3 has 8 vertices and 12 edges. It is a bipartite graph with two parts of 4,
    each vertex having degree 3.
    
    We look for 8 vertices that form Q3.
    Strategy: enumerate subsets of 8 vertices, check if induced subgraph contains Q3.
    But 16 choose 8 = 12870, and checking each is expensive.
    
    Better: Q3 is characterized by:
    - 8 vertices, 12 edges
    - bipartite with parts of size 4
    - each vertex degree 3 in the subgraph
    - it's the unique 3-regular bipartite graph on 8 vertices (up to isomorphism)
    
    Actually, Q3 is the only 3-regular bipartite graph on 8 vertices? No, there are others.
    Let me think: Q3 has girth 4. The other 3-regular bipartite graphs on 8 vertices...
    Actually, there's K_{4,4} which is 4-regular. For 3-regular bipartite on 8 vertices,
    Q3 is one. Is there another? 
    
    Let me just do a direct check: for each set of 8 vertices, check if the induced
    subgraph is isomorphic to Q3.
    
    Q3 can be represented as: vertices 0-7, edges:
    (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    
    Actually, let's use a smarter approach. Q3 is the Cartesian product of three K2s.
    Its vertices can be labeled by 3-bit strings, and two vertices are adjacent if they
    differ in exactly one bit.
    
    For a set of 8 vertices, we need to check if there's a bijection to {0,1}^3 such that
    adjacency in the subgraph matches Hamming distance 1.
    
    This is equivalent to: the induced subgraph on 8 vertices must be 3-regular, bipartite
    (with parts of 4), and have exactly 12 edges, and be connected.
    
    Wait, is that sufficient? A 3-regular bipartite connected graph on 8 vertices with parts
    of 4: is it necessarily Q3? 
    The number of 3-regular bipartite graphs on 4+4 vertices: Q3 is one. Is there another?
    Let me enumerate: K_{4,4} minus a perfect matching gives a 3-regular bipartite graph.
    There are 3! = 6 ways to remove a perfect matching from K_{4,4}. Are they all isomorphic?
    Yes, all such graphs are isomorphic to Q3. So the characterization is:
    - 8 vertices
    - 3-regular
    - bipartite with parts of size 4
    - connected
    
    Actually, 3-regular bipartite with parts of 4 automatically has 12 edges. And it's
    connected (since it's 3-regular, it can't be disconnected unless each component has
    at least 4 vertices on each side, but we only have 4 on each side, so it must be connected).
    
    So the check is: induced subgraph on 8 vertices is 3-regular and bipartite with parts of 4.
    
    For efficiency, we can precompute. But 12870 subsets * checking 3-regularity and
    bipartiteness might be okay.
    
    Let me implement this.
    """
    n = 16
    # Enumerate all 8-subsets
    for subset in itertools.combinations(range(n), 8):
        # Check if induced subgraph is 3-regular
        ok = True
        for v in subset:
            deg = 0
            for u in subset:
                if u != v and (adj[v] >> u) & 1:
                    deg += 1
            if deg != 3:
                ok = False
                break
        if not ok:
            continue
        # Check bipartite with parts of size 4
        # Try to 2-color the subgraph
        # Build local adjacency
        local_adj = {}
        for v in subset:
            local_adj[v] = []
        for v in subset:
            for u in subset:
                if u != v and (adj[v] >> u) & 1:
                    local_adj[v].append(u)
        
        # BFS to check bipartiteness and part sizes
        color = {}
        parts = [0, 0]
        for start in subset:
            if start in color:
                continue
            color[start] = 0
            queue = [start]
            bipartite = True
            while queue:
                v = queue.pop(0)
                for u in local_adj[v]:
                    if u not in color:
                        color[u] = 1 - color[v]
                        parts[color[u]] += 1
                        queue.append(u)
                    elif color[u] == color[v]:
                        bipartite = False
                        break
                if not bipartite:
                    break
            if not bipartite:
                break
        if not bipartite:
            continue
        if parts[0] == 4 and parts[1] == 4:
            return True
    return False

def _count_q3s(adj):
    """Count the number of Q3 subgraphs in the graph."""
    n = 16
    count = 0
    for subset in itertools.combinations(range(n), 8):
        ok = True
        for v in subset:
            deg = 0
            for u in subset:
                if u != v and (adj[v] >> u) & 1:
                    deg += 1
            if deg != 3:
                ok = False
                break
        if not ok:
            continue
        local_adj = {}
        for v in subset:
            local_adj[v] = []
        for v in subset:
            for u in subset:
                if u != v and (adj[v] >> u) & 1:
                    local_adj[v].append(u)
        color = {}
        parts = [0, 0]
        bipartite = True
        for start in subset:
            if start in color:
                continue
            color[start] = 0
            parts[0] += 1
            queue = [start]
            while queue:
                v = queue.pop(0)
                for u in local_adj[v]:
                    if u not in color:
                        color[u] = 1 - color[v]
                        parts[color[u]] += 1
                        queue.append(u)
                    elif color[u] == color[v]:
                        bipartite = False
                        break
                if not bipartite:
                    break
            if not bipartite:
                break
        if bipartite and parts[0] == 4 and parts[1] == 4:
            count += 1
    return count

def construct():
    start_time = time.time()
    
    # Start with complete graph K16
    n = 16
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    best_edges = get_edges(adj)
    best_count = len(best_edges)
    
    # Check initial state
    if not _has_q3(adj):
        return best_edges
    
    # Heuristic removal: repeatedly remove the edge that is in the most Q3s
    # or use a greedy approach.
    
    # First, let's try a simple greedy: while Q3 exists, find a Q3 and remove an edge from it.
    # To be smarter, remove the edge that appears in the most Q3s.
    
    # For efficiency, let's track which edges are in Q3s.
    # This is expensive, so let's use a simpler heuristic.
    
    # Strategy: 
    # 1. Find all Q3s
    # 2. For each edge, count how many Q3s contain it
    # 3. Remove the edge with the highest count
    # 4. Repeat until no Q3s
    
    # But finding all Q3s each iteration is expensive. Let's limit iterations.
    
    # Alternative: use a randomized approach with multiple restarts.
    
    # Let's try a greedy removal with edge scoring.
    
    for iteration in range(200):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Check if Q3-free
        if not _has_q3(adj):
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
            break
        
        # Find Q3s and score edges
        # For simplicity, find one Q3 and remove a random edge from it
        # But let's be smarter: find all Q3s and score edges
        
        # To save time, let's just find a few Q3s
        q3s = []
        for subset in itertools.combinations(range(n), 8):
            if time.time() - start_time > DEADLINE - 5:
                break
            ok = True
            for v in subset:
                deg = 0
                for u in subset:
                    if u != v and (adj[v] >> u) & 1:
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if not ok:
                continue
            local_adj = {}
            for v in subset:
                local_adj[v] = []
            for v in subset:
                for u in subset:
                    if u != v and (adj[v] >> u) & 1:
                        local_adj[v].append(u)
            color = {}
            parts = [0, 0]
            bipartite = True
            for start in subset:
                if start in color:
                    continue
                color[start] = 0
                parts[0] += 1
                queue = [start]
                while queue:
                    v = queue.pop(0)
                    for u in local_adj[v]:
                        if u not in color:
                            color[u] = 1 - color[v]
                            parts[color[u]] += 1
                            queue.append(u)
                        elif color[u] == color[v]:
                            bipartite = False
                            break
                    if not bipartite:
                        break
                if not bipartite:
                    break
            if bipartite and parts[0] == 4 and parts[1] == 4:
                # This is a Q3
                q3_edges = []
                for v in subset:
                    for u in local_adj[v]:
                        if v < u:
                            q3_edges.append((v, u))
                q3s.append(q3_edges)
        
        if not q3s:
            # No Q3s found (shouldn't happen if _has_q3 said there is one)
            break
        
        # Score edges by frequency in found Q3s
        edge_score = {}
        for q3_edges in q3s:
            for e in q3_edges:
                edge_score[e] = edge_score.get(e, 0) + 1
        
        # Remove the edge with highest score
        if edge_score:
            max_edge = max(edge_score, key=edge_score.get)
            u, v = max_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    # Final check
    if not _has_q3(adj):
        edges = get_edges(adj)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    return best_edges
