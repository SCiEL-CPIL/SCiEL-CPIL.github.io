import random
import time
from itertools import combinations

def construct():
    n = 31
    target_edges = 91
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def count_c4_from_edges(edges):
        """Count exact number of 4-cycles in a graph given its edge list."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = (adj[i] & adj[j]).bit_count()
                c4_count += common * (common - 1) // 2
        return c4_count

    def build_polarity_graph():
        """Build a 90-edge C4-free graph using finite field construction for q=5."""
        # For q odd, the polarity graph of PG(2,q) has q^2+q+1 vertices
        # and each vertex has degree q+1, giving (q^2+q+1)(q+1)/2 edges
        # For q=5: 31 vertices, degree 6, edges = 31*6/2 = 93... 
        # Wait, that's 93 not 90. Let me reconsider.
        #
        # Actually, the standard polarity graph construction:
        # Vertices: points of PG(2,q) = q^2+q+1 = 31 for q=5
        # Edge between two points if they are orthogonal under a polarity
        # Each point is adjacent to q+1 = 6 other points
        # Total edges = 31 * 6 / 2 = 93
        # But we need only 90 edges for a C4-free subgraph.
        #
        # Let me use a different approach: construct a C4-free graph with 90 edges
        # using a greedy method with the finite field structure as a guide.
        
        # Use the incidence structure: label vertices 0..30
        # Create a bipartite-like structure that avoids C4
        
        edges = []
        adj = [0] * n
        
        # Greedy construction guided by modular arithmetic
        # Try to create a graph where each pair of vertices shares at most 1 common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                if i == j:
                    continue
                common = (adj[i] & adj[j]).bit_count()
                if common <= 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append((i, j))
                    if len(edges) == 90:
                        return edges, adj
        return edges, adj

    def local_search_swap(edges, adj, max_time=10):
        """Try to improve the graph by swapping edges to reduce C4 count."""
        start = time.time()
        current_c4 = count_c4_from_edges(edges)
        best_edges = edges[:]
        best_adj = adj[:]
        best_c4 = current_c4
        
        edge_set = set(edges)
        
        while time.time() - start < max_time:
            improved = False
            # Try random swaps
            for _ in range(100):
                if time.time() - start >= max_time:
                    break
                
                # Pick a random edge to remove
                e1 = random.choice(edges)
                if e1 not in edge_set:
                    continue
                u1, v1 = e1
                
                # Find a non-edge to add
                # Sample some non-edges
                for _ in range(20):
                    u2, v2 = random.sample(range(n), 2)
                    if u2 > v2:
                        u2, v2 = v2, u2
                    if (u2, v2) in edge_set:
                        continue
                    
                    # Check if swap is valid (maintains 91 edges)
                    # Temporarily remove e1 and add (u2,v2)
                    common_before = (best_adj[u1] & best_adj[v1]).bit_count()
                    # After removing e1, u1 and v1 lose each other as neighbors
                    # After adding (u2,v2), u2 and v2 gain each other as neighbors
                    
                    # Estimate C4 change
                    # This is complex; just try it
                    new_adj = best_adj[:]
                    # Remove e1
                    new_adj[u1] &= ~(1 << v1)
                    new_adj[v1] &= ~(1 << u1)
                    # Add (u2,v2)
                    new_adj[u2] |= (1 << v2)
                    new_adj[v2] |= (1 << u2)
                    
                    new_c4 = count_c4_from_edges([e for e in best_edges if e != e1] + [(u2, v2)])
                    
                    if new_c4 < best_c4:
                        best_edges = [e for e in best_edges if e != e1] + [(u2, v2)]
                        best_adj = new_adj
                        best_c4 = new_c4
                        edge_set.discard(e1)
                        edge_set.add((u2, v2))
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try a few more random restarts of the swap process
                if random.random() < 0.3:
                    break
        
        return best_edges, best_adj, best_c4

    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    # Phase 1: Try deterministic construction
    det_edges, det_adj = build_polarity_graph()
    
    if len(det_edges) >= 90:
        # Find the best 91st edge to add
        edge_set = set(det_edges)
        min_cost = float('inf')
        best_add_edge = None
        
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
            common = (det_adj[u] & det_adj[v]).bit_count()
            cost = common * (common - 1) // 2
            if cost < min_cost:
                min_cost = cost
                best_add_edge = (u, v)
        
        if best_add_edge:
            candidate_edges = det_edges[:90] + [best_add_edge]
            candidate_c4 = count_c4_from_edges(candidate_edges)
            print(f"Deterministic construction: C4 count = {candidate_c4}")
            
            # Phase 2: Local search to improve
            improved_edges, improved_adj, improved_c4 = local_search_swap(candidate_edges, det_adj, max_time=30)
            if improved_c4 < candidate_c4:
                print(f"After local search: C4 count = {improved_c4}")
                return [[u, v] for u, v in improved_edges]
            else:
                return [[u, v] for u, v in candidate_edges]
    
    # Phase 3: Random search if deterministic failed
    best_score = float('inf')
    best_edges = None
    
    attempts = 0
    max_attempts = 5000
    
    while time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        random.shuffle(all_pairs)
        
        neighbors = [0] * n
        edge_count = 0
        current_edges = []
        edge_set = set()
        
        for u, v in all_pairs:
            common = (neighbors[u] & neighbors[v]).bit_count()
            if common <= 1:
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                edge_count += 1
                current_edges.append((u, v))
                edge_set.add((u, v))
                
                if edge_count == 90:
                    break
        
        if edge_count < 90:
            continue
            
        # Evaluate the cost of adding the 91st edge
        min_cost = float('inf')
        best_add_edge = None
        
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
            common = (neighbors[u] & neighbors[v]).bit_count()
            cost = common * (common - 1) // 2
            if cost < min_cost:
                min_cost = cost
                best_add_edge = (u, v)
                
        if min_cost < best_score:
            best_score = min_cost
            best_edges = current_edges[:] + [best_add_edge]
            
            if best_score == 1:
                break
    
    if best_edges is None:
        # Fallback: just return first 91 pairs (bad score, but valid)
        return [[i, j] for i, j in all_pairs[:91]]
        
    return [[u, v] for u, v in best_edges]

# Clear static state
if hasattr(construct, '_best_edges'):
    del construct._best_edges
