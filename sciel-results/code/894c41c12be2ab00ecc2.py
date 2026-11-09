import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n = 16
    # All possible edges
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)  # 120
    
    # Build adjacency for triangle-freeness check
    # For triangle-free: if (u,v) is an edge, then N(u) ∩ N(v) must be empty
    # We can precompute for each pair (u,v) which other edges would create a triangle
    
    # Precompute conflict graph: edge e1 conflicts with e2 if adding both creates a triangle
    # e1=(a,b), e2=(c,d) conflict if {a,b} ∩ {c,d} != empty and they share a vertex
    # Actually, two edges conflict (can't both be in a triangle-free graph) if:
    # - They share a vertex AND the other endpoints are adjacent... no wait.
    # In a triangle-free graph, if (a,b) and (a,c) are both edges, then (b,c) cannot be an edge.
    # So the conflict is: edge (a,b) conflicts with edge (b,c) for any c != a, AND
    # edge (a,b) conflicts with edge (a,c) for any c != b.
    # More precisely: in a triangle-free graph, for any vertex v, N(v) is an independent set.
    # So if (u,v) is in the graph, then no two neighbors of u can be adjacent to each other,
    # and no two neighbors of v can be adjacent to each other.
    
    # Let's think of it differently. We want a set of edges S such that:
    # 1. S is triangle-free (no three edges form a triangle)
    # 2. The graph (V, S) is not 3-colorable
    # 3. |S| is maximized
    
    # Triangle-free constraint: for any three vertices a,b,c, not all three of
    # (a,b), (b,c), (c,a) can be in S.
    
    # This is a maximum weight independent set problem on a conflict graph where:
    # - Nodes = all 120 possible edges
    # - Two edge-nodes conflict if they form a triangle with some third edge
    #   i.e., edges (a,b) and (b,c) conflict if (a,c) could also be present
    #   Actually, (a,b) and (b,c) conflict because if both are present, (a,c) cannot be.
    #   But the real constraint is: we cannot have all three of (a,b),(b,c),(c,a).
    #   So (a,b) conflicts with (b,c) only in the sense that they jointly forbid (a,c).
    #   Actually no - the constraint is simply: no triangle. So (a,b) and (b,c) can both
    #   be present as long as (a,c) is not. The conflict is between triples.
    
    # Better formulation: This is a 3-uniform hypergraph independent set problem.
    # But that's hard. Let me think of a different approach.
    
    # Alternative: Use the fact that in a triangle-free graph, each vertex's neighborhood
    # is an independent set. So for each vertex v, the edges incident to v form a "star"
    # and the edges between neighbors of v must all be absent.
    
    # Let me use a different strategy: start with a known dense triangle-free 4-chromatic
    # graph and try to add edges.
    
    # The Clebsch graph has 16 vertices and 40 edges, is triangle-free and 4-chromatic.
    # Let's start from there and try to add more edges.
    
    # Actually, let me think about what the maximum could be.
    # Mantel's theorem: triangle-free graph on n vertices has at most n^2/4 edges.
    # For n=16: at most 64 edges.
    # But we also need 4-chromatic. The Clebsch graph has 40 edges.
    
    # Let me try a branch-and-bound / backtracking approach with good pruning.
    
    # Strategy: 
    # 1. Build Clebsch graph as starting point (40 edges, triangle-free, 4-chromatic)
    # 2. Try to add edges one by one, checking triangle-freeness and 4-chromaticity
    # 3. Use backtracking if stuck
    
    # First, let's build the Clebsch graph
    def build_clebsch():
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                xor = u ^ v
                dist = bin(xor).count('1')
                if dist == 1 or dist == 4:
                    edges.add((u, v))
        return edges
    
    clebsch_edges = build_clebsch()
    
    # Adjacency structure
    def make_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Check triangle-free
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True
    
    # Fast 3-colorability check using bitmasks
    def is_3_colorable(adj):
        """Returns True if graph is 3-colorable."""
        # Use backtracking with MRV and forward checking
        color = [-1] * n
        neighbors = [list(adj[v]) for v in range(n)]
        
        # Try all 3^16 assignments? No, use backtracking.
        # Better: use the fact that we can fix vertex 0 to color 0 by symmetry.
        
        def backtrack(v):
            if v == n:
                return True
            
            # MRV: find uncolored vertex with fewest available colors
            best_v = -1
            best_count = 4
            best_colors = None
            
            for i in range(v, n):
                if color[i] != -1:
                    continue
                used = set(color[j] for j in neighbors[i] if color[j] != -1)
                available = [c for c in range(3) if c not in used]
                if len(available) < best_count:
                    best_count = len(available)
                    best_v = i
                    best_colors = available
                    if best_count == 0:
                        return False
                    if best_count == 1:
                        break
            
            if best_v == -1:
                return True
            
            for c in best_colors:
                color[best_v] = c
                if backtrack(v + 1):
                    return True
                color[best_v] = -1
            
            return False
        
        return backtrack(0)
    
    # Verify Clebsch
    adj = make_adj(clebsch_edges)
    assert is_triangle_free(adj), "Clebsch should be triangle-free"
    assert not is_3_colorable(adj), "Clebsch should not be 3-colorable"
    
    best_edges = set(clebsch_edges)
    best_adj = [set(s) for s in adj]
    best_count = len(best_edges)
    
    # Now try to add edges greedily and with backtracking
    # An edge (u,v) can be added if:
    # 1. It doesn't create a triangle: N(u) ∩ N(v) = empty (excluding each other)
    # 2. The resulting graph is still not 3-colorable
    
    def can_add_edge_no_triangle(adj, u, v):
        """Check if adding (u,v) keeps graph triangle-free."""
        # Need N(u) ∩ N(v) = empty
        common = adj[u] & adj[v]
        return len(common) == 0
    
    def try_add_edge(adj, edge_set, u, v):
        """Try to add edge (u,v). Returns (new_adj, new_edge_set) or None."""
        if not can_add_edge_no_triangle(adj, u, v):
            return None
        new_adj = [set(s) for s in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        new_edge_set = set(edge_set)
        new_edge_set.add((u, v))
        return new_adj, new_edge_set
    
    # Greedy: try to add edges one by one
    def greedy_add(adj, edge_set):
        """Greedily add edges that maintain triangle-freeness and 4-chromaticity."""
        adj = [set(s) for s in adj]
        edge_set = set(edge_set)
        count = len(edge_set)
        
        # Repeatedly try to add an edge
        changed = True
        while changed:
            changed = False
            # Try all missing edges
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) in edge_set:
                        continue
                    if not can_add_edge_no_triangle(adj, u, v):
                        continue
                    new_adj, new_es = try_add_edge(adj, edge_set, u, v)
                    if new_adj is not None and not is_3_colorable(new_adj):
                        adj = new_adj
                        edge_set = new_es
                        count += 1
                        changed = True
                        break
                if changed:
                    break
        return adj, edge_set, count
    
    # Apply greedy
    best_adj, best_edges, best_count = greedy_add(best_adj, best_edges)
    
    # Now use local search / simulated annealing to try to improve further
    def get_missing_edges(edge_set):
        return [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in edge_set]
    
    # Simulated annealing with edge swaps
    current_adj = [set(s) for s in best_adj]
    current_edges = set(best_edges)
    current_count = best_count
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add an edge
            missing = get_missing_edges(current_edges)
            if not missing:
                continue
            u, v = random.choice(missing)
            if not can_add_edge_no_triangle(current_adj, u, v):
                continue
            new_adj, new_es = try_add_edge(current_adj, current_edges, u, v)
            if new_adj is None:
                continue
            
            if not is_3_colorable(new_adj):
                # Valid: accept
                current_adj = new_adj
                current_edges = new_es
                current_count = len(current_edges)
                if current_count > best_count:
                    best_adj = [set(s) for s in current_adj]
                    best_edges = set(current_edges)
                    best_count = current_count
            else:
                # 3-colorable: reject
                pass
        
        elif move_type < 0.8:
            # Swap: remove one edge, add another
            edge_list = list(current_edges)
            missing = get_missing_edges(current_edges)
            if not edge_list or not missing:
                continue
            
            u, v = random.choice(edge_list)
            x, y = random.choice(missing)
            
            # Remove (u,v)
            new_adj = [set(s) for s in current_adj]
            new_adj[u].discard(v)
            new_adj[v].discard(u)
            
            # Add (x,y) if triangle-free
            if not can_add_edge_no_triangle(new_adj, x, y):
                continue
            new_adj[x].add(y)
            new_adj[y].add(x)
            
            new_es = set(current_edges)
            new_es.discard((u, v))
            new_es.add((x, y))
            
            if not is_3_colorable(new_adj):
                # Valid
                delta = len(new_es) - current_count
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_adj = new_adj
                    current_edges = new_es
                    current_count = len(current_edges)
                    if current_count > best_count:
                        best_adj = [set(s) for s in current_adj]
                        best_edges = set(current_edges)
                        best_count = current_count
            # else: reject
        
        else:
            # Remove an edge (to escape local optima)
            edge_list = list(current_edges)
            if not edge_list:
                continue
            u, v = random.choice(edge_list)
            
            new_adj = [set(s) for s in current_adj]
            new_adj[u].discard(v)
            new_adj[v].discard(u)
            new_es = set(current_edges)
            new_es.discard((u, v))
            
            if not is_3_colorable(new_adj):
                # Still 4-chromatic after removal
                delta = -1
                if random.random() < math.exp(delta / T):
                    current_adj = new_adj
                    current_edges = new_es
                    current_count = len(current_edges)
        
        T = max(T * cooling, T_min)
        
        # Periodic restart if stuck
        if steps % 5000 == 0 and current_count < best_count:
            current_adj = [set(s) for s in best_adj]
            current_edges = set(best_edges)
            current_count = best_count
    
    # Final greedy pass on best solution
    best_adj, best_edges, best_count = greedy_add(best_adj, best_edges)
    
    # Convert to edge list
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([u, v])
    
    return result
