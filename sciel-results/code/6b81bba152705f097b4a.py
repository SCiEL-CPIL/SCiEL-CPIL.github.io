"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def greedy_vertex_cover(edges, n, rng_seed):
        """
        Greedy vertex cover with random tie-breaking.
        Returns a set of vertices.
        """
        rng = random.Random(rng_seed)
        # Map vertex to list of incident edges (indices)
        # Or simpler: maintain set of uncovered edges
        uncovered = set(edges)
        cover = set()
        
        # Pre-compute adjacency for faster degree counting if needed, 
        # but for small n, counting on the fly is okay.
        # Let's use a dictionary of neighbors for the current uncovered subgraph?
        # Actually, just iterating through uncovered edges to find max degree vertex is O(E) per step.
        # Max steps is V. So O(V*E). For n=40, E~200, V=40. 8000 ops. Very fast.
        
        while uncovered:
            # Count degrees in the uncovered subgraph
            degrees = {v: 0 for v in range(n)}
            for u, v in uncovered:
                degrees[u] += 1
                degrees[v] += 1
            
            # Find max degree vertex
            max_deg = -1
            candidates = []
            for v in range(n):
                if degrees[v] > max_deg:
                    max_deg = degrees[v]
                    candidates = [v]
                elif degrees[v] == max_deg:
                    candidates.append(v)
            
            if not candidates:
                break # Should not happen if uncovered is not empty
            
            v = rng.choice(candidates)
            cover.add(v)
            
            # Remove all edges incident to v
            new_uncovered = set()
            for e in uncovered:
                if v not in e:
                    new_uncovered.add(e)
            uncovered = new_uncovered
            
        return cover

    def local_search_remove(cover, edges, n):
        """
        Try to remove vertices from the cover if valid.
        A vertex v can be removed if all its neighbors are in the cover.
        """
        # Convert cover to set for fast lookup
        cover_set = set(cover)
        
        # Build adjacency list from original edges
        adj = {v: set() for v in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        changed = True
        while changed:
            changed = False
            for v in list(cover_set):
                # Check if all neighbors are in cover
                if all(neighbor in cover_set for neighbor in adj[v]):
                    cover_set.remove(v)
                    changed = True
                    
        return cover_set

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5.0:
            # Emergency fallback: return all non-isolated vertices
            edges = generate_graph(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        edges = generate_graph(n, p, seed)
        
        best_cover = None
        best_size = float('inf')
        
        # Try multiple random restarts for the greedy phase
        for restart in range(50):
            if time.time() > deadline - 5.0:
                break
                
            gc = greedy_vertex_cover(edges, n, seed + restart)
            # Local search improvement
            lc = local_search_remove(gc, edges, n)
            
            if len(lc) < best_size:
                best_size = len(lc)
                best_cover = lc
                
        if best_cover is None:
            # Fallback if loop didn't run
            touched = {x for e in edges for x in e}
            best_cover = touched
            
        results.append(sorted(best_cover))
        
    return results
# EVOLVE-BLOCK-END
