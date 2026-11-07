"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from functools import lru_cache

def construct():
    n = 15
    vertices = list(range(n))
    
    # Generate all possible edges
    all_edges = list(itertools.combinations(vertices, 2))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Precompute for each edge (u,v), the set of 3-edge paths s-u-v-t where s!=t
    # A path s-u-v-t exists if edge (s,u) and edge (v,t) exist, s!=t, and (s,u)!=(u,v), (v,t)!=(u,v)
    # Actually for path s-u-v-t: edges are (s,u), (u,v), (v,t)
    # s and t are endpoints, u and v are internal
    
    # For each potential edge (u,v), find all pairs (s,t) such that if edges (s,u) and (v,t) 
    # are also present, we get a forbidden path s-u-v-t with deg(s)==deg(t)
    
    # Strategy: Use branch and bound. We'll try to add edges one by one.
    # But the constraint depends on the FULL graph degrees, which is tricky.
    
    # Alternative approach: 
    # 1. First decide on the degree sequence
    # 2. Then find a graph with that degree sequence that avoids the constraint
    
    # Actually, let me think differently. The constraint is:
    # For every path s-u-v-t (3 edges, s≠t), deg(s) ≠ deg(t)
    
    # This means: if deg(s) == deg(t), then there must be NO path of length 3 between s and t.
    # Equivalently: for any two vertices s,t with the same degree, the distance between them 
    # (if finite) must not be exactly 3, OR more precisely, there is no 3-edge path.
    
    # Wait, the constraint is about ANY 3-edge path, not necessarily shortest. So if deg(s)==deg(t),
    # then s and t must not be connected by any walk of exactly 3 edges that is a simple path.
    
    # Simple path of length 3: s-u-v-t where all vertices distinct and edges (s,u),(u,v),(v,t) exist.
    
    # Let me use a different approach: try to construct a graph with a specific degree structure.
    
    # Idea: Make the graph such that vertices have very different degrees, or use a structure
    # where same-degree vertices are far apart.
    
    # Let me try: make vertex 0 have degree 14 (connected to all), and the rest form a structure.
    # If vertex 0 has degree 14, then any other vertex v has degree at most 14 (connected to 0 plus others).
    
    # Actually, let me try a more systematic approach with backtracking.
    
    # For each pair of vertices, track if they can have a 3-path between them given current edges.
    
    # Let me use a greedy + local search approach:
    # 1. Start with a random or structured graph
    # 2. Check validity
    # 3. Try to add edges while maintaining validity
    
    def is_valid(edges_set, n):
        """Check if the graph defined by edges_set is valid."""
        # Build adjacency and degrees
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for (u, v) in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Check all 3-edge paths s-u-v-t
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        # Path s-u-v-t
                        if deg[s] == deg[t]:
                            return False
        return True
    
    # Start with empty graph and try to add edges greedily
    # But greedy might not be optimal. Let me try multiple random orders.
    
    import random
    random.seed(42)
    
    best_edges = set()
    best_count = 0
    
    # Try multiple random edge orderings
    for trial in range(50):
        random.shuffle(all_edges)
        current_edges = set()
        for edge in all_edges:
            test_edges = current_edges | {edge}
            if is_valid(test_edges, n):
                current_edges = test_edges
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
    
    # Try to improve by removing some edges and adding others
    # Local search: try swapping edges
    
    improved = True
    while improved:
        improved = False
        current = set(best_edges)
        # Try removing each edge and see if we can add more
        for edge in list(current):
            test = current - {edge}
            # Try to add any edge not in test
            added = False
            for new_edge in all_edges:
                if new_edge in test:
                    continue
                if is_valid(test | {new_edge}, n):
                    test = test | {new_edge}
                    added = True
            if len(test) > len(current):
                current = test
                best_edges = test
                best_count = len(test)
                improved = True
                break
    
    # Convert to list of [u, v]
    result = [[u, v] for (u, v) in best_edges]
    
    # Ensure types are correct
    final_result = []
    for edge in result:
        u = int(edge[0])
        v = int(edge[1])
        final_result.append([u, v])
    
    return final_result

# EVOLVE-BLOCK-END
