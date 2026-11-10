import time
import random
import sys
from itertools import combinations

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 115  # safety margin for 120s evaluation

def build_adj(edges, n=16):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def is_triangle_free(edges, n=16):
    adj = build_adj(edges, n)
    for u in range(n):
        for v in list(adj[u]):
            if v > u:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
    return True

def is_not_3_colorable(adj, n=16):
    """Backtracking check: is the graph NOT 3-colorable?"""
    # Use DSATUR-like ordering for efficiency
    def dfs(colors, uncolored, adj):
        if not uncolored:
            return False  # successfully 3-colored
        # Pick vertex with most colored neighbors (DSATUR heuristic)
        best_v = -1
        best_score = -1
        for v in uncolored:
            colored_neighbors = [u for u in adj[v] if colors[u] != -1]
            score = len(colored_neighbors)
            if score > best_score:
                best_score = score
                best_v = v
        v = best_v
        for c in range(3):
            ok = True
            for nb in adj[v]:
                if colors[nb] == c:
                    ok = False
                    break
            if ok:
                colors[v] = c
                uncolored.remove(v)
                if not dfs(colors, uncolored, adj):
                    return False
                uncolored.add(v)
                colors[v] = -1
        return True
    
    colors = [-1] * n
    uncolored = set(range(n))
    return dfs(colors, uncolored, adj)

def greedy_3_colorable(adj, n=16, max_trials=100):
    """Fast greedy check with random orderings."""
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = set(colors[nb] for nb in adj[node] if colors[nb] != -1)
            for c in range(3):
                if c not in used:
                    colors[node] = c
                    break
            else:
                break
        else:
            return True  # found a 3-coloring
    return False  # likely not 3-colorable

def is_not_3_colorable_fast(edges, n=16):
    """Combined fast check."""
    adj = build_adj(edges, n)
    if greedy_3_colorable(adj, n, max_trials=50):
        return False  # definitely 3-colorable
    return is_not_3_colorable(adj, n)

def find_min_deletions_k88(n=16):
    """
    Start with K8,8 (vertices 0-7 vs 8-15), delete edges to make it non-3-colorable.
    Use iterative deletion with backtracking on the 3-colorability check.
    """
    left = list(range(8))
    right = list(range(8, 16))
    
    # Start with full K8,8
    full_edges = [[u, v] for u in left for v in right]
    edge_set = set(tuple(sorted(e)) for e in full_edges)
    
    # We need to delete edges to make it non-3-colorable.
    # Strategy: iteratively try deleting edges in a smart order.
    # The key insight: to make a bipartite graph non-3-colorable, we need to
    # create a structure that forces more than 3 colors.
    
    # Actually, any subgraph of a bipartite graph is bipartite, hence 2-colorable,
    # hence 3-colorable. So we CANNOT make a subgraph of K8,8 non-3-colorable!
    # This is a critical realization.
    
    # So we need a different approach. We need a non-bipartite triangle-free graph.
    # The graph must contain an odd cycle but no triangles.
    
    # Strategy: Start with a known non-3-colorable triangle-free graph (Grötzsch)
    # and try to add as many edges as possible while maintaining triangle-free and
    # non-3-colorable properties.
    return None

def build_groetzsch_edges():
    """Build the Grötzsch graph (Mycielski of C5) on 11 vertices."""
    # C5: 0-1-2-3-4-0
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    # Mycielski: add vertices 5-9 (copies of 0-4), edge from v' to each neighbor of v
    for u, v in [[0,1],[1,2],[2,3],[3,4],[4,0]]:
        edges.append([5+u, v])
        edges.append([5+v, u])
    # Add vertex 10 connected to all 5-9
    for u in range(5):
        edges.append([10, 5+u])
    return edges

def mycielski_graph(n_iter=1):
    """
    Build Mycielski graph iteratively.
    Mycielski of a graph G with vertices V:
    - Add a copy of each vertex: v' for v in V
    - Add a new vertex w
    - Edges: (u', v) for each edge (u,v) in G
    - Edges: (w, v') for each v in V
    """
    # Start with C5
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    n = 5
    
    for _ in range(n_iter):
        # Add copies n..2n-1
        old_n = n
        new_edges = []
        # For each edge (u,v) in G, add (u', v) and (v', u) where u'=u+old_n
        for u, v in edges:
            new_edges.append([u + old_n, v])
            new_edges.append([v + old_n, u])
        # Add new vertex 2n-1 connected to all copies
        for u in range(old_n):
            new_edges.append([2*old_n - 1, u + old_n])
        edges = edges + new_edges
        n = 2 * old_n
    
    return edges, n

def local_search_add_edges(start_edges, n=16, max_time=60):
    """
    Start with a non-3-colorable triangle-free graph, try to add edges
    while maintaining both properties.
    """
    current_edges = list(start_edges)
    adj = build_adj(current_edges, n)
    best_edges = list(current_edges)
    best_count = len(current_edges)
    
    # Get current 3-coloring status
    is_nc = is_not_3_colorable_fast(current_edges, n)
    if not is_nc:
        return best_edges
    
    deadline = START_TIME + max_time
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Build candidate edges (non-edges that don't create triangles)
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    # Check if adding (u,v) creates a triangle
                    common = adj[u] & adj[v]
                    if len(common) == 0:
                        candidates.append((u, v))
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            if v in adj[u]:
                continue
            
            # Try adding edge
            new_edges = current_edges + [[u, v]]
            
            # Quick triangle-free check (should pass by construction)
            if not is_triangle_free(new_edges, n):
                continue
            
            # Check non-3-colorable
            if is_not_3_colorable_fast(new_edges, n):
                current_edges = new_edges
                adj[u].add(v)
                adj[v].add(u)
                improved = True
                cnt = len(current_edges)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = list(current_edges)
                break  # restart after improvement
    
    return best_edges

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 115
    
    try:
        # Strategy 1: Start with Grötzsch graph (11 vertices, 11 edges)
        # and add edges to reach 16 vertices with max edges
        groetzsch = build_groetzsch_edges()
        
        # Pad to 16 vertices (vertices 11-15 are isolated initially)
        base_edges = list(groetzsch)
        
        # Try multiple random initializations for the extra vertices
        best_overall = base_edges
        best_count = len(base_edges)
        
        for trial in range(50):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Create random connections for vertices 11-15
            edges_try = list(groetzsch)
            adj = build_adj(edges_try, 16)
            
            extra = [11, 12, 13, 14, 15]
            
            for vertex in extra:
                candidates = list(range(11))
                random.shuffle(candidates)
                
                for c in candidates:
                    if time.time() - START_TIME > TIME_LIMIT:
                        break
                    common = adj[vertex] & adj[c]
                    if len(common) == 0:
                        edges_try.append([vertex, c])
                        adj[vertex].add(c)
                        adj[c].add(vertex)
            
            # Try to add edges between all pairs
            for u in range(16):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                for v in range(u+1, 16):
                    if time.time() - START_TIME > TIME_LIMIT:
                        break
                    if v in adj[u]:
                        continue
                    common = adj[u] & adj[v]
                    if len(common) == 0:
                        edges_try.append([u, v])
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Verify
            if not is_triangle_free(edges_try, 16):
                continue
            
            if not is_not_3_colorable_fast(edges_try, 16):
                continue
            
            cnt = len(edges_try)
            if cnt > best_count:
                best_count = cnt
                best_overall = list(edges_try)
            
            # Local search improvement
            if time.time() - START_TIME < TIME_LIMIT - 10:
                improved = local_search_add_edges(edges_try, 16, max_time=min(30, TIME_LIMIT - (time.time() - START_TIME) - 5))
                if len(improved) > best_count:
                    best_count = len(improved)
                    best_overall = improved
        
        if best_count > len(groetzsch):
            return best_overall
        
        return groetzsch
    
    except Exception as e:
        pass
    
    # Fallback: Grötzsch graph
    return build_groetzsch_edges()
