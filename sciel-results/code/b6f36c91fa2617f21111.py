"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Deadline guard: must finish within 150s wall-clock.
# We use a safety margin to ensure we return in time.
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds to reserve

def _is_triangle_free(edges, n):
    """Check if graph is triangle-free using adjacency sets."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and w in adj[u]:
                    return False
    return True

def _max_independent_set_size(edges, n):
    """Exact maximum independent set size via branch and bound."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best = 0
    
    def branch(candidates, size, current_set):
        nonlocal best
        if size + len(candidates) <= best:
            return
        if not candidates:
            best = max(best, size)
            return
        
        # Choose vertex with max degree in candidates for pruning
        v = max(candidates, key=lambda x: len(adj[x] & set(candidates)))
        
        # Include v
        new_candidates = candidates - {v} - adj[v]
        branch(new_candidates, size + 1, current_set | {v})
        
        # Exclude v
        branch(candidates - {v}, size, current_set)
    
    branch(set(range(n)), 0, frozenset())
    return best

def _construct_andrasfai(n_mod, k):
    """
    Construct Andrasfai-type graph.
    Vertices 0..n-1 where n = n_mod * k or similar.
    i ~ j iff (j - i) mod n is in S, where S is chosen to avoid triangles and large independent sets.
    """
    n = n_mod
    # For Andrasfai graph And(r): Z_{3r} with i~j iff (j-i) mod 3r is 1 mod 3
    # This gives triangle-free graph on 3r vertices with independence number r.
    # For r=9: n=27, alpha=9. We need alpha < 10.
    # For r=10: n=30, alpha=10. This is NOT valid (alpha=10).
    # So And(9) is the largest Andrasfai graph with alpha < 10.
    # But we can try other constructions.
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff % 3 == 1:
                edges.append([i, j])
    return n, edges

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Start with Andrasfai graph And(9): n=27, alpha=9
    # Wait, the original code uses n=26. Let me check.
    # Original: n=26, i~j iff (j-i) mod 26 is 1 mod 3.
    # That's a different graph. Let me use the known best: Andrasfai And(9) has 27 vertices.
    # But the original uses 26. Let me check what the original does.
    
    # Actually, let me try to extend the graph or find better constructions.
    
    # Known result: R(3,10) <= 41, and there exist triangle-free graphs on 39 vertices with no independent set of size 10.
    # The largest known such graph has 39 vertices.
    
    # Strategy: Try to construct a graph with as many vertices as possible.
    # Use a greedy/local search approach to build a triangle-free graph with alpha < 10.
    
    best_n = 0
    best_edges = []
    
    # Try Andrasfai-type constructions
    for r in range(1, 15):
        if time.time() > deadline:
            break
        n = 3 * r
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff % 3 == 1:
                    edges.append([i, j])
        
        # Check validity
        if _is_triangle_free(edges, n):
            alpha = _max_independent_set_size(edges, n)
            if alpha < 10:
                if n > best_n:
                    best_n = n
                    best_edges = edges
    
    # Try random search for larger graphs
    # Use a simple greedy approach: start with empty graph, add edges that don't create triangles
    # while keeping alpha < 10
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Try different target sizes
        for target_n in range(best_n + 1, 41):
            if time.time() > deadline:
                break
            
            # Randomized greedy construction
            n = target_n
            adj = [set() for _ in range(n)]
            edges = []
            
            # Random ordering of vertex pairs
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            success = True
            for u, v in pairs:
                # Check if adding (u,v) creates a triangle
                common = adj[u] & adj[v]
                if common:
                    continue  # Would create triangle, skip
                
                # Check if adding (u,v) keeps alpha < 10
                # This is expensive, so we only check periodically
                if len(edges) % 5 == 0 or len(pairs) - pairs.index((u,v)) < 20:
                    test_edges = edges + [[u, v]]
                    if not _is_triangle_free(test_edges, n):
                        continue
                    alpha = _max_independent_set_size(test_edges, n)
                    if alpha >= 10:
                        continue
                
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
            
            # Final check
            if _is_triangle_free(edges, n):
                alpha = _max_independent_set_size(edges, n)
                if alpha < 10:
                    if n > best_n:
                        best_n = n
                        best_edges = edges
    
    if best_n <= 0:
        # Fallback to Andrasfai And(9)
        n, edges = _construct_andrasfai(27, 9)
        return {'n': n, 'edges': edges}
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
