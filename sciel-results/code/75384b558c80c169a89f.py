"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 110.0
    
    # 1. Initial Random Graph
    # To ensure exactly 401 edges, we pick a random set.
    # Total possible edges = 40*39/2 = 780.
    # 401 is about half.
    
    # Helper to create edge list from indices
    def get_edges_from_mask(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges

    # Initialize adjacency matrix
    adj = np.zeros((n, n), dtype=np.uint8)
    
    # Create a random graph with 401 edges
    # We can do this by picking random pairs until we have enough, 
    # but we must avoid duplicates.
    # A faster way: shuffle all possible edges and take first 401.
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    random.shuffle(all_edges)
    initial_edges = all_edges[:target_edges]
    
    for u, v in initial_edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    # Compute initial triangle counts
    # tri[u, v] = number of common neighbors of u and v
    # This is equivalent to (adj @ adj)[u, v] for u != v
    # But we only care about edges that exist.
    tri = np.zeros((n, n), dtype=np.int32)
    
    # Efficient triangle count calculation
    # tri[u, v] = sum_w adj[u, w] * adj[v, w]
    # This is the (u, v) entry of adj @ adj
    # However, adj @ adj includes self-loops in the diagonal if we aren't careful, 
    # but for u != v, it's exactly the number of common neighbors.
    tri_matrix = adj @ adj
    # We only care about tri[u, v] where adj[u, v] == 1
    # For non-edges, tri value doesn't matter for the "book" definition, 
    # but it's needed for updates.
    tri = tri_matrix.astype(np.int32)
    
    # Initialize max_book
    # We need max tri[u, v] for all edges (u, v)
    # Mask out non-edges
    masked_tri = np.where(adj == 1, tri, 0)
    current_max_book = int(np.max(masked_tri))
    
    best_adj = adj.copy()
    best_tri = tri.copy()
    best_max_book = current_max_book
    
    # Local Search
    # Precompute list of edges for quick removal
    # We will maintain a set of edges for O(1) removal check? 
    # Or just use the matrix.
    
    # To pick a random edge to remove:
    # We can keep a list of edges and update it, or just sample from matrix.
    # Sampling from matrix: pick random u, v, if adj[u,v], use it.
    # Given density ~0.5, this is fast.
    
    # To pick a random non-edge to add:
    # Pick random u, v, if not adj[u,v], use it.
    
    while time.time() < deadline:
        # Pick random edge to remove
        # Retry a few times if we pick a non-edge
        while True:
            u1 = random.randint(0, n-1)
            v1 = random.randint(0, n-1)
            if u1 == v1: continue
            if u1 > v1: u1, v1 = v1, u1
            if adj[u1, v1]:
                break
        
        # Pick random non-edge to add
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2: continue
            if u2 > v2: u2, v2 = v2, u2
            if not adj[u2, v2]:
                break
        
        # Calculate impact of removing (u1, v1)
        # Common neighbors of u1, v1
        # cn1 = np.where(adj[u1] & adj[v1])[0]
        # This is fast with numpy.
        cn1 = np.where(adj[u1] & adj[v1])[0]
        
        # Calculate impact of adding (u2, v2)
        # Note: u2, v2 is currently a non-edge.
        # cn2 = common neighbors of u2, v2
        cn2 = np.where(adj[u2] & adj[v2])[0]
        
        # We need to simulate the swap on a copy of tri?
        # Or calculate delta?
        # Given the small size, copying tri (1600 ints) is cheap.
        new_tri = tri.copy()
        
        # 1. Remove (u1, v1)
        # tri[u1, v1] becomes 0
        new_tri[u1, v1] = 0
        new_tri[v1, u1] = 0
        
        # For each w in cn1:
        # tri[u1, w] -= 1
        # tri[v1, w] -= 1
        if len(cn1) > 0:
            new_tri[u1, cn1] -= 1
            new_tri[v1, cn1] -= 1
            # Symmetry
            new_tri[cn1, u1] -= 1
            new_tri[cn1, v1] -= 1
            
        # 2. Add (u2, v2)
        # tri[u2, v2] becomes len(cn2)
        new_tri[u2, v2] = len(cn2)
        new_tri[v2, u2] = len(cn2)
        
        # For each w in cn2:
        # tri[u2, w] += 1
        # tri[v2, w] += 1
        if len(cn2) > 0:
            new_tri[u2, cn2] += 1
            new_tri[v2, cn2] += 1
            new_tri[cn2, u2] += 1
            new_tri[cn2, v2] += 1
            
        # Update adjacency matrix for this trial
        # We don't actually need to update adj for the trial, 
        # just to compute the new max book on the *affected* edges?
        # No, the max book is over ALL edges.
        # The only edges whose triangle count changed are:
        # - (u1, v1) [removed]
        # - (u1, w) for w in cn1
        # - (v1, w) for w in cn1
        # - (u2, v2) [added]
        # - (u2, w) for w in cn2
        # - (v2, w) for w in cn2
        
        # To find the new max book, we need to check the max of:
        # - The old max book, EXCEPT if the edge that achieved it was modified.
        # - The new values of the modified edges.
        
        # This is tricky because we don't know which edge was the old max.
        # However, n=40 is small. 
        # We can just compute max over the modified edges and compare with 
        # a "cached" max that excludes modified edges?
        # Or just recompute max over all edges using the new_tri matrix?
        # np.max is very fast.
        
        # Let's just recompute max on new_tri for edges that exist in the NEW graph.
        # New graph edges = Old edges - {(u1, v1)} + {(u2, v2)}
        
        # We can create a mask for the new edges.
        # new_adj = adj.copy() is expensive? 40x40 is small.
        # Let's just update adj in place, check, and revert if needed.
        
        # Apply changes to adj
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        # Compute new max book
        # Mask out non-edges in new_tri
        # We can use np.where(adj == 1, new_tri, 0)
        new_max_book = int(np.max(np.where(adj == 1, new_tri, 0)))
        
        # Decide to accept
        accepted = False
        if new_max_book < current_max_book:
            accepted = True
        elif new_max_book == current_max_book:
            # Accept if it reduces variance or just random acceptance to escape plateaus
            # Let's check if it reduces the sum of squares of triangle counts
            # This encourages more uniform distribution
            old_sum_sq = np.sum(np.where(adj == 1, tri, 0) ** 2) # Wait, adj is already modified!
            # We need to be careful.
            
            # Let's revert first to calculate old stats safely?
            # Or just accept with small probability?
            # Given time limit, simple hill climbing might get stuck.
            # Let's try: accept if new_max_book <= current_max_book.
            # If equal, accept with 50% chance?
            # Better: accept if it reduces the count of edges with max book size.
            
            # Let's just accept if <= for now to see if it improves.
            # Actually, if we accept equal, we might wander.
            # Let's stick to strict improvement first. If stuck, we can add noise.
            # But random walk with "equal" acceptance is often good for escaping local minima.
            # Let's accept if new_max_book <= current_max_book.
            accepted = True
            
        if accepted:
            tri = new_tri
            current_max_book = new_max_book
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_adj = adj.copy()
                best_tri = tri.copy()
        else:
            # Revert adj
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            # tri remains unchanged
            
    # Return best edges found
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                best_edges.append([i, j])
                
    return best_edges
# EVOLVE-BLOCK-END
