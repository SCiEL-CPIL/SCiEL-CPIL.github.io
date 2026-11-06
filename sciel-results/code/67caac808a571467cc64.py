"""Graph on 16 vertices, 65 edges: minimize max triangle degree sum."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    # Start with K_{6,10} + 5 internal edges (baseline score 24)
    # Partition: A = {0..5} (6 vertices), B = {6..15} (10 vertices)
    # Cross edges: all 6*10 = 60
    # Need 5 more edges within A or B
    # Internal edges within A: C(6,2)=15 possible, within B: C(10,2)=45 possible
    
    def build_edges(partition_a, internal_edges_a, internal_edges_b):
        """Build edge list from partition and internal edges."""
        edges = []
        # Cross edges
        for u in partition_a:
            for v in range(n):
                if v not in partition_a:
                    edges.append([u, v])
        # Internal edges in A
        for e in internal_edges_a:
            edges.append([e[0], e[1]])
        # Internal edges in B
        for e in internal_edges_b:
            edges.append([e[0], e[1]])
        return edges
    
    def compute_score_fast(adj, n):
        """Compute max triangle degree sum using bitmask adjacency."""
        # adj[i] is a bitmask of neighbors of i
        deg = np.zeros(n, dtype=np.int32)
        for i in range(n):
            deg[i] = bin(adj[i]).count('1')
        
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    # common has bits for vertices connected to both i and j
                    while common:
                        k_bit = common & (-common)
                        k = k_bit.bit_length() - 1
                        common ^= k_bit
                        if k > j:  # ensure i < j < k to avoid duplicates
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def edges_to_adj(edges, n):
        """Convert edge list to bitmask adjacency."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    # Initial construction: K_{6,10} with 5 internal edges in A
    # A = {0,1,2,3,4,5}, B = {6,...,15}
    # To minimize triangle degree sums, we want internal edges to create triangles
    # with vertices of low degree. In K_{6,10}, deg(a) = 10 for a in A, deg(b) = 6 for b in B.
    # An internal edge (a1,a2) in A creates triangles (a1,a2,b) for each b in B.
    # deg(a1)+deg(a2)+deg(b) = (10+1)+(10+1)+6 = 28 if both have 1 internal edge... 
    # Wait, let me reconsider.
    
    # Actually in K_{6,10}: each a in A has degree 10, each b in B has degree 6.
    # Adding internal edge (a1,a2): a1 and a2 each gain +1 degree.
    # Triangle (a1,a2,b): sum = 11 + 11 + 6 = 28. That's bad.
    
    # Better: add internal edges in B. B has 10 vertices, each with degree 6.
    # Internal edge (b1,b2) in B: b1,b2 each gain +1 degree.
    # Triangle (a,b1,b2): sum = 10 + 7 + 7 = 24. 
    # With 5 internal edges in B, if they form a matching or sparse structure,
    # the max degree in B becomes 6 + (number of internal edges incident to that vertex).
    # If we use a 5-matching in B, 5 vertices get degree 7, others stay 6.
    # Max triangle sum: 10 + 7 + 7 = 24.
    
    # Let's try: 5 disjoint edges in B (a matching of size 5)
    # B = {6,7,8,9,10,11,12,13,14,15}
    # Matching: (6,7), (8,9), (10,11), (12,13), (14,15)
    
    initial_internal_b = [[6,7], [8,9], [10,11], [12,13], [14,15]]
    initial_edges = build_edges(list(range(6)), [], initial_internal_b)
    initial_adj = edges_to_adj(initial_edges, n)
    initial_score = compute_score_fast(initial_adj, n)
    
    best_edges = initial_edges
    best_adj = initial_adj
    best_score = initial_score
    
    # Local search: try to improve by swapping edges
    # Move: remove one edge, add another, keeping 65 edges total
    # Or: 2-opt style: remove two edges, add two edges
    
    random.seed(42)
    
    def try_improve(adj, edges, score, max_iters=50000):
        """Local search with edge swaps."""
        current_adj = adj[:]
        current_edges = edges[:]
        current_score = score
        improved = True
        
        while improved:
            improved = False
            for _ in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                
                # Pick a random edge to remove
                idx = random.randint(0, len(current_edges) - 1)
                u, v = current_edges[idx]
                
                # Try adding a random non-edge
                for attempt in range(10):
                    ru = random.randint(0, n-1)
                    rv = random.randint(0, n-1)
                    if ru == rv:
                        continue
                    if ru > rv:
                        ru, rv = rv, ru
                    
                    # Check if (ru,rv) is already an edge
                    if current_adj[ru] & (1 << rv):
                        continue
                    # Check if it's the same edge we're removing
                    if (ru == u and rv == v) or (ru == v and rv == u):
                        continue
                    
                    # Apply swap
                    old_adj = current_adj[:]
                    old_edges = current_edges[:]
                    
                    # Remove (u,v)
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_edges[idx] = [ru, rv]
                    
                    # Add (ru,rv)
                    current_adj[ru] |= (1 << rv)
                    current_adj[rv] |= (1 << ru)
                    
                    new_score = compute_score_fast(current_adj, n)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        current_adj = old_adj
                        current_edges = old_edges
                        break  # Try a different random edge
                    
                    break
            
            if time.time() > deadline - 5.0:
                break
        
        return current_edges, current_adj, current_score
    
    # Run local search
    best_edges, best_adj, best_score = try_improve(best_adj, best_edges, best_score)
    
    if time.time() > deadline - 5.0:
        return {'n': n, 'edges': best_edges}
    
    # Try 2-opt moves: remove 2 edges, add 2 edges
    def try_2opt(adj, edges, score, max_iters=20000):
        """2-opt local search."""
        current_adj = adj[:]
        current_edges = edges[:]
        current_score = score
        improved = True
        
        while improved:
            improved = False
            for _ in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                
                # Pick two random edges to remove
                idx1 = random.randint(0, len(current_edges) - 1)
                idx2 = random.randint(0, len(current_edges) - 1)
                if idx1 == idx2:
                    continue
                u1, v1 = current_edges[idx1]
                u2, v2 = current_edges[idx2]
                
                # Try adding two random non-edges
                for attempt in range(5):
                    ru1 = random.randint(0, n-1)
                    rv1 = random.randint(0, n-1)
                    ru2 = random.randint(0, n-1)
                    rv2 = random.randint(0, n-1)
                    if ru1 == rv1 or ru2 == rv2:
                        continue
                    if ru1 > rv1:
                        ru1, rv1 = rv1, ru1
                    if ru2 > rv2:
                        ru2, rv2 = rv2, ru2
                    
                    # Check non-edges and no duplicates
                    if current_adj[ru1] & (1 << rv1):
                        continue
                    if current_adj[ru2] & (1 << rv2):
                        continue
                    if (ru1 == u1 and rv1 == v1) or (ru1 == v1 and rv1 == u1):
                        continue
                    if (ru2 == u2 and rv2 == v2) or (ru2 == v2 and rv2 == u2):
                        continue
                    if (ru1 == u2 and rv1 == v2) or (ru1 == v2 and rv1 == u2):
                        continue
                    if (ru1 == u1 and rv1 == v2) or (ru1 == v2 and rv1 == u1):
                        continue
                    if ru1 == ru2 and rv1 == rv2:
                        continue
                    
                    # Apply
                    old_adj = current_adj[:]
                    old_edges = current_edges[:]
                    
                    current_adj[u1] &= ~(1 << v1)
                    current_adj[v1] &= ~(1 << u1)
                    current_adj[u2] &= ~(1 << v2)
                    current_adj[v2] &= ~(1 << u2)
                    current_edges[idx1] = [ru1, rv1]
                    current_edges[idx2] = [ru2, rv2]
                    
                    current_adj[ru1] |= (1 << rv1)
                    current_adj[rv1] |= (1 << ru1)
                    current_adj[ru2] |= (1 << rv2)
                    current_adj[rv2] |= (1 << ru2)
                    
                    new_score = compute_score_fast(current_adj, n)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        current_adj = old_adj
                        current_edges = old_edges
                        break
                    
                    break
            
            if time.time() > deadline - 5.0:
                break
        
        return current_edges, current_adj, current_score
    
    best_edges, best_adj, best_score = try_2opt(best_adj, best_edges, best_score)
    
    # Multiple restarts with different initial configurations
    if time.time() < deadline - 10.0:
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Different initial internal edge configurations in B
            b_vertices = list(range(6, 16))
            random.shuffle(b_vertices)
            internal_b = []
            for i in range(5):
                internal_b.append([b_vertices[2*i], b_vertices[2*i+1]])
            
            edges = build_edges(list(range(6)), [], internal_b)
            adj = edges_to_adj(edges, n)
            score = compute_score_fast(adj, n)
            
            if score < best_score:
                best_edges, best_adj, best_score = edges, adj, score
            
            # Local search from this start
            if time.time() < deadline - 10.0:
                e, a, s = try_improve(adj, edges, score, max_iters=20000)
                if s < best_score:
                    best_edges, best_adj, best_score = e, a, s
                
                if time.time() < deadline - 10.0:
                    e, a, s = try_2opt(a, e, s, max_iters=10000)
                    if s < best_score:
                        best_edges, best_adj, best_score = e, a, s
    
    return {'n': n, 'edges': best_edges}
