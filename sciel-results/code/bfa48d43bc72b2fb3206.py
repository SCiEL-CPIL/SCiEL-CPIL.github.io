"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Leave margin before 150s kill
    
    # Start with K20,20 plus one internal edge
    # This gives score 1 (the edge [0,1] is in 0 triangles since 0 and 1 are in same part)
    # Wait, let's analyze: K20,20 has parts A={0..19}, B={20..39}
    # Edge [0,1] is within A. Triangles containing [0,1] need a vertex v connected to both 0 and 1.
    # v must be in B (since no other edges in A). All vertices in B are connected to both 0 and 1.
    # So there are 20 triangles containing [0,1]. Score = 20.
    
    # We need to minimize the maximum book size.
    # Theorem says score >= 40/6 ≈ 6.67, so minimum integer is 7.
    
    # Let's try a different approach: start with a near-complete bipartite graph
    # and add edges strategically, or use a random search with local optimization.
    
    # Actually, let's think about this more carefully.
    # n=40, m=401 edges. Complete bipartite K20,20 has 400 edges.
    # Adding 1 edge within a part creates triangles with all 20 vertices in the other part.
    # So score = 20 for that edge.
    
    # To reduce this, we could:
    # 1. Use a different base graph
    # 2. Add multiple internal edges but remove some cross edges to balance
    
    # Strategy: Start with K20,20, remove k cross edges, add k+1 internal edges
    # such that each internal edge has fewer common neighbors.
    
    # If we add an internal edge (u,v) in part A, and remove edges from u and v to some vertices in B,
    # the number of triangles containing (u,v) = number of vertices in B connected to both u and v.
    
    # Let's try: remove edges to make internal edges have at most 7 common neighbors.
    
    # For an internal edge in A, if we ensure it shares at most 7 common neighbors in B, score <= 7.
    
    # Let's try a construction:
    # Part A = {0..19}, Part B = {20..39}
    # Start with all cross edges (400 edges).
    # Remove some cross edges and add internal edges.
    
    # Target: 401 edges total.
    # If we add t internal edges in A and t-1 internal edges in B, total = 400 + 2t - 1 = 401, so t=1.
    # That's the original: 1 internal edge, score 20.
    
    # Alternatively: add t internal edges in A, s internal edges in B, remove r cross edges.
    # 400 + t + s - r = 401, so t + s - r = 1.
    
    # If t=2, s=0, r=1: add 2 internal edges in A, remove 1 cross edge.
    # Each internal edge in A has at most 20 common neighbors (minus any removed).
    # If the removed cross edge is incident to one of the internal edge endpoints, that edge has 19 common neighbors.
    # Still high.
    
    # If t=8, s=0, r=7: add 8 internal edges in A, remove 7 cross edges.
    # We need each internal edge to have <= 7 common neighbors in B.
    # This requires removing enough cross edges.
    
    # Let's try a more systematic approach: use simulated annealing or random search.
    
    # For now, let's try a specific construction that might work:
    # Use a circulant-like structure or random graph with controlled triangle density.
    
    # Actually, let's try: complete bipartite K19,21 has 19*21 = 399 edges.
    # Add 2 more edges. If both are internal to the size-19 part, each has 21 common neighbors. Score 21.
    # If one is internal to size-19 part and one to size-21 part:
    # Internal to size-19: 21 common neighbors
    # Internal to size-21: 19 common neighbors
    # Score 21.
    
    # Let's try K18,22: 18*22 = 396 edges. Need 5 more.
    # Add 5 internal edges. Worst case score = 22 or 18 depending on which part.
    
    # It seems hard to get below 7 with simple constructions.
    
    # Let's try a random approach with local search.
    
    def count_books(edges_set):
        """Count the maximum book size for a given set of edges."""
        # Build adjacency
        adj = [set() for _ in range(40)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        max_book = 0
        for u, v in edges_set:
            # Count common neighbors
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book
    
    # Start with K20,20 + 1 internal edge
    initial_edges = set()
    for a in range(20):
        for b in range(20, 40):
            initial_edges.add((a, b) if a < b else (b, a))
    initial_edges.add((0, 1))
    
    current_edges = initial_edges.copy()
    current_score = count_books(current_edges)
    best_edges = current_edges.copy()
    best_score = current_score
    
    # Local search: try swapping edges
    # Move: remove one edge, add another (keeping 401 edges)
    
    all_possible_edges = [(i, j) for i in range(40) for j in range(i+1, 40)]
    
    random.seed(42)
    
    # Time-limited local search
    while time.time() < deadline - 5:
        # Pick a random edge to remove
        if not current_edges:
            break
        remove_edge = random.choice(list(current_edges))
        
        # Try to add a random edge not already present
        # Sample a few candidates
        found_improvement = False
        for _ in range(50):
            add_edge = random.choice(all_possible_edges)
            if add_edge in current_edges:
                continue
            
            # Try the swap
            new_edges = current_edges.copy()
            new_edges.discard(remove_edge)
            new_edges.add(add_edge)
            
            new_score = count_books(new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                found_improvement = True
                if current_score < best_score:
                    best_score = current_score
                    best_edges = current_edges.copy()
                break
            elif new_score == current_score and random.random() < 0.1:
                # Accept neutral moves with low probability
                current_edges = new_edges
                found_improvement = True
                break
        
        if not found_improvement:
            # Try a different remove edge
            continue
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
