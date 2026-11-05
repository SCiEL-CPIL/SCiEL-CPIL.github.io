"""Optimized construction to minimize 4-cycles in a graph with 31 vertices and 91 edges."""
import time
import random
import numpy as np

def construct():
    """
    Construct a graph on 31 vertices with exactly 91 edges minimizing the number of 4-cycles.
    Uses a time-bounded local search starting from a structured initial graph.
    """
    start_time = time.time()
    deadline = 145.0  # seconds, with safety margin before 150s kill
    n = 31
    target_edges = 91
    
    def count_c4(adj_matrix):
        """
        Count 4-cycles using adjacency matrix.
        A 4-cycle is a set of 4 distinct vertices a,b,c,d with edges ab, bc, cd, da.
        Each 4-cycle is counted 8 times (4 choices for start, 2 directions).
        We return the count divided by 8.
        """
        # For each pair of vertices (u, v), count common neighbors w
        # A 4-cycle u-w-v-x-u corresponds to pairs (u,v) and (w,x) where w,x are common neighbors
        # More efficiently: for each pair (i,j), let c_ij = number of common neighbors
        # Then the number of 4-cycles is sum over all pairs (i,j) of C(c_ij, 2) / 2
        # Wait, let me think more carefully.
        
        # A 4-cycle is a sequence of 4 distinct vertices v1, v2, v3, v4 with edges v1-v2, v2-v3, v3-v4, v4-v1.
        # Each 4-cycle has 8 representations (4 starting points * 2 directions).
        # 
        # Alternative: count the number of "2-paths" of length 2. For each pair of vertices (u,v),
        # let d(u,v) be the number of common neighbors. Then the number of 4-cycles is:
        # sum_{u<v} C(d(u,v), 2) / 2
        # Because each 4-cycle u-w-v-x-u is counted once for the pair (u,v) and once for (w,x),
        # and within each pair, the two common neighbors are chosen.
        
        # Actually: for a fixed pair (u,v), if they have k common neighbors, then there are C(k,2) 
        # ways to choose two common neighbors w,x, which gives a 4-cycle u-w-v-x-u (or u-x-v-w-u).
        # Each 4-cycle is counted exactly twice this way (once for each pair of opposite vertices).
        # So total 4-cycles = (1/2) * sum_{u<v} C(d(u,v), 2)
        
        # Compute d(u,v) for all pairs using matrix multiplication
        # d(u,v) = sum_w A[u,w] * A[w,v] = (A^2)[u,v]
        A = adj_matrix
        A2 = A @ A  # numpy matrix multiplication
        # d(u,v) = A2[u,v] for u != v, and A2[u,u] = degree(u)
        
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = A2[u, v]
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2
    
    def edges_to_adj(edges):
        """Convert edge list to adjacency matrix."""
        adj = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    def adj_to_edges(adj):
        """Convert adjacency matrix to edge list."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges
    
    def generate_random_graph(num_edges, seed=None):
        """Generate a random graph with exactly num_edges edges."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)
        return all_pairs[:num_edges]
    
    def local_search(edges, max_iter=5000):
        """
        Local search: try swapping one edge for another to reduce 4-cycles.
        """
        adj = edges_to_adj(edges)
        current_c4 = count_c4(adj)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge to remove
            u, v = edges[random.randint(0, len(edges) - 1)]
            
            # Find a non-edge to add
            # We need to find (a, b) not in adj, a != b, a,b not both equal to u or v necessarily
            # Actually we just need any non-edge
            # For efficiency, let's try a few random non-edges
            improved = False
            for attempt in range(10):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a == b or adj[a, b]:
                    continue
                
                # Check if adding (a,b) and removing (u,v) keeps exactly target_edges
                # It does: we remove one, add one
                
                # Quick check: does this swap reduce 4-cycles?
                # We can estimate the change in 4-cycles without full recount
                # 
                # Removing edge (u,v): reduces 4-cycles that use this edge
                # Adding edge (a,b): increases 4-cycles that use this edge
                #
                # Number of 4-cycles using edge (u,v): 
                # For each common neighbor w of u and v, and for each... 
                # Actually, a 4-cycle using edge (u,v) has the form u-v-x-y-u or u-v-y-x-u
                # This requires x adjacent to v and y, and y adjacent to u and x
                # So it's a 2-path from u to v of length 3: u-y-x-v
                # Number of such 2-paths = number of pairs (y,x) where y~u, x~v, y~x, y!=v, x!=u
                # = sum over y in N(u)\{v} of |N(y) ∩ N(v)\{u}|
                
                # This is getting complex. Let's just do the full recount for simplicity,
                # but limit the number of swaps we try per iteration.
                
                # Apply swap
                adj[u, v] = 0
                adj[v, u] = 0
                adj[a, b] = 1
                adj[b, a] = 1
                
                new_c4 = count_c4(adj)
                
                if new_c4 < current_c4:
                    # Accept the swap
                    edges.remove([u, v])
                    edges.append([a, b])
                    current_c4 = new_c4
                    improved = True
                    break
                else:
                    # Revert
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj[a, b] = 0
                    adj[b, a] = 0
            
            if not improved:
                # Try a few more random restarts of the local search
                continue
        
        return edges, current_c4
    
    # Start with a random graph
    best_edges = None
    best_c4 = float('inf')
    
    # Try several random starting points
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        
        edges = generate_random_graph(target_edges, seed=seed)
        edges, c4 = local_search(edges, max_iter=2000)
        
        if c4 < best_c4:
            best_c4 = c4
            best_edges = edges[:]
    
    # If we haven't found anything good, do a final check
    if best_edges is None or len(best_edges) != target_edges:
        # Fallback: use a structured approach
        # Start with a random graph and do more local search
        edges = generate_random_graph(target_edges, seed=12345)
        best_edges, best_c4 = local_search(edges, max_iter=10000)
    
    # Final verification
    assert len(best_edges) == target_edges, f"Expected {target_edges} edges, got {len(best_edges)}"
    assert len(set(tuple(sorted(e)) for e in best_edges)) == target_edges, "Duplicate edges found"
    for e in best_edges:
        assert 0 <= e[0] < n and 0 <= e[1] < n, f"Vertex out of range: {e}"
        assert e[0] != e[1], f"Self-loop: {e}"
    
    return best_edges
