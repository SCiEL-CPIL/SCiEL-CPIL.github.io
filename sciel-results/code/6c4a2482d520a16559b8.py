"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools

def construct():
    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = generate_edges(n, p, seed)
        
        # Build initial adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        partition = []
        remaining_edges = set(edges)
        
        # To pick edges efficiently, we can just iterate through a copy of remaining_edges
        # But we need to remove edges as we form cliques.
        # A better way: Keep a list of edges, and when we form a clique, 
        # we filter out edges that are part of the clique.
        # However, filtering the whole list every time is O(E * #cliques).
        # Given E is small (< 2000), this is fine.
        
        # Heuristic: Pick edge with max common neighbors in remaining graph.
        # Then grow clique.
        
        while remaining_edges:
            # Find best edge
            best_edge = None
            max_common = -1
            # Sample a few edges or iterate all? 
            # Iterating all is O(E * N) which is ~ 1000 * 64 = 64k ops per clique.
            # #cliques is at most E/2 ~ 500. Total ~ 30M ops. Should be fast enough in Python?
            # Maybe we can sample. But deterministic is safer.
            
            # Let's iterate through a list of remaining edges to avoid set iteration overhead in selection?
            # Actually, set iteration is fine.
            for e in remaining_edges:
                u, v = e
                # Common neighbors in current remaining graph
                # adj[u] and adj[v] reflect the remaining graph?
                # We need to update adj when we remove edges.
                # Let's maintain adj dynamically.
                
                common = adj[u].intersection(adj[v])
                if len(common) > max_common:
                    max_common = len(common)
                    best_edge = e
                    if max_common > n: break # unlikely
            
            if best_edge is None:
                break
                
            u, v = best_edge
            clique = [u, v]
            
            # Candidates for next vertex: common neighbors of all in clique
            candidates = adj[u].intersection(adj[v])
            
            # Greedy growth
            while candidates:
                # Pick candidate w that maximizes connections to other candidates
                # or just pick one that keeps the clique large?
                # Standard greedy: pick w that has max degree in the induced subgraph of candidates
                # Or simply pick w that is connected to the most other candidates.
                
                best_w = None
                max_deg = -1
                for w in candidates:
                    # Count neighbors of w in candidates
                    deg = len(adj[w].intersection(candidates))
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                
                if best_w is None:
                    break
                    
                clique.append(best_w)
                
                # Update candidates: must be connected to best_w
                candidates = candidates.intersection(adj[best_w])
                
            # Remove all edges in this clique from remaining_edges and adj
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u2, v2 = clique[i], clique[j]
                    e = (min(u2, v2), max(u2, v2))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                    # Update adj
                    if v2 in adj[u2]:
                        adj[u2].remove(v2)
                    if u2 in adj[v2]:
                        adj[v2].remove(u2)
                    
            partition.append(clique)
            
        return partition

    p1 = solve_instance(32, 0.6, 601)
    p2 = solve_instance(64, 0.55, 602)
    
    return [p1, p2]

# EVOLVE-BLOCK-END
