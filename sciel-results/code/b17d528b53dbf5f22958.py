import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {edge: i for i, edge in enumerate(all_edges)}
    E = len(all_edges)
    
    # Precompute adjacency matrix as a numpy array for faster operations
    # adj_np[i, j] is 1 if edge (i, j) exists, else 0
    adj_np = np.zeros((n, n), dtype=np.int8)
    
    def creates_q3(u, v):
        # Check if adding edge (u, v) creates a Q3
        # We assume (u, v) is NOT currently in adj_np
        
        # Neighbors of u (excluding v, since (u,v) not in adj yet)
        # In numpy, adj_np[u] is a vector of 0s and 1s.
        # We need to find squares u-a-b-v.
        # a in N(u), b in N(v), and (a,b) is an edge.
        
        # Vectorized check:
        # For each a in N(u), b in N(v), check if adj[a, b] == 1.
        
        # Let's construct the set of squares more efficiently.
        # squares is a list of (a, b) pairs.
        
        # Instead of lists, we can use bitmasks which are very fast in Python if done carefully, 
        # but numpy might be faster for the "exists e, f" part.
        
        # Let's stick to the bitmask logic but optimize the inner loop.
        # The current bottleneck is likely the O(S^2) loop over squares and the inner checks.
        
        # Alternative:
        # A Q3 is two squares sharing an edge? No, Q3 is 8 vertices.
        # The previous logic:
        # 1. Find all squares containing edge (u,v).
        # 2. Check if any two squares can be "completed" to a Q3.
        
        # Let's optimize the bitmask iteration.
        neighbors_u = adj_np[u]
        neighbors_v = adj_np[v]
        
        # We need to iterate over a in neighbors_u, b in neighbors_v such that adj[a, b] == 1.
        # This is equivalent to finding non-zero entries in the submatrix adj[neighbors_u, :][:, neighbors_v].
        
        # Let's use numpy to find these pairs.
        # mask_u = neighbors_u (vector of 1s for neighbors)
        # mask_v = neighbors_v
        # submatrix = adj_np[np.ix_(neighbors_u.nonzero()[0], neighbors_v.nonzero()[0])]
        # This creates a copy, which might be slow.
        
        # Bitmask approach was:
        # a_mask = neighbors_u (int)
        # b_mask = neighbors_v (int)
        
        # Let's convert adj_np rows to int bitmasks?
        # This requires conversion every time adj changes.
        # adj_int = [0]*n
        # for i in range(n): adj_int[i] = sum(1 << j for j in range(n) if adj_np[i, j])
        
        # This conversion is O(n^2) = 256 ops.
        # Greedy step: 120 edges * 256 = 30k ops.
        # Local search step: 120 edges * 256 = 30k ops.
        # Total per restart: ~1000 steps * 30k = 30M ops.
        # 115s / 30M = ~4M ops/s. Python can do ~10-50M simple ops/s.
        # So this is feasible.
        
        # Let's maintain adj_int alongside adj_np.
        
        # Wait, the current code uses `adj` which is a list of ints (bitmasks).
        # It does NOT use adj_np. I added adj_np in my draft but didn't use it in creates_q3.
        # Let's stick to the bitmask approach but optimize the logic.
        
        # Optimization 1: Precompute squares list more efficiently?
        # Optimization 2: Vectorize the "exists e, f" check.
        
        # Let's try to use numpy for the "exists e, f" check.
        # For a given pair of squares (a,b) and (c,d):
        # e_candidates = neighbors of a AND neighbors of c, excluding {u,v,a,b,c,d}
        # f_candidates = neighbors of b AND neighbors of d, excluding {u,v,a,b,c,d}
        # Check if any e in e_candidates is connected to any f in f_candidates.
        
        # Using bitmasks:
        # e_mask = adj[a] & adj[c] & ~used_mask
        # f_mask = adj[b] & adj[d] & ~used_mask
        # Check if any e in e_mask has a neighbor in f_mask.
        # This is: OR over e in e_mask of (adj[e] & f_mask) != 0
        
        # This is hard to vectorize in pure Python without loops.
        # But we can precompute a "neighborhood union" or similar?
        
        # Alternative:
        # If we maintain adj as a list of sets, intersection is O(1) average? No, O(min(len)).
        
        # Let's try to speed up the bitmask iteration.
        # The inner loop:
        # e_mask = e_candidates
        # while e_mask:
        #   e = ...
        #   if adj[e] & f_candidates: return True
        
        # We can precompute `adj_union`? No.
        
        # Let's try a different strategy:
        # Instead of checking all pairs of squares, can we check for Q3 existence differently?
        # A Q3 contains 12 squares.
        # If adding (u,v) creates a Q3, then (u,v) is part of that Q3.
        
        # Let's stick to the current logic but optimize the constants.
        
        # One major optimization:
        # If the number of squares is large, we are likely to find a conflict quickly.
        # If the number of squares is small, the loop is fast.
        
        # Let's try to use `any()` and generator expressions for the inner checks?
        # Python generators are slower than explicit loops usually.
        
        # Let's try to cache the result of `creates_q3`?
        # The state of the graph changes, so caching is hard.
        
        # Let's try to improve the local search.
        # The current local search does:
        # 1. Try to add any missing edge.
        # 2. If no edge added, remove one, try to add another.
        
        # This is a 1-opt move.
        # We can try 2-opt moves: remove 2, add 2.
        # Or just random restarts with better initialization.
        
        # The inspiration 1 (score 1.3846) uses "Faster Q3 check via neighbor set intersection".
        # Let's implement that.
        
        # Neighbor set intersection:
        # Maintain adj as a list of sets of ints.
        # e_candidates = adj[a] & adj[c] - used_set
        # f_candidates = adj[b] & adj[d] - used_set
        # Check if any e in e_candidates has a neighbor in f_candidates.
        # This is: any(adj[e] & f_candidates for e in e_candidates)
        
        # Set intersection in Python is implemented in C and is very fast.
        # Let's switch to sets.
        
        return False # Placeholder, will be replaced

    # Let's rewrite creates_q3 using sets
    adj_sets = [set() for _ in range(n)]
    
    def creates_q3_sets(u, v):
        # u, v are not connected
        # Find squares
        squares = []
        # a in adj[u], b in adj[v], (a,b) in adj
        # To iterate efficiently:
        # Iterate a in adj[u]
        for a in adj_sets[u]:
            # b in adj[v] and b in adj[a]
            # intersection = adj_sets[v] & adj_sets[a]
            common = adj_sets[v] & adj_sets[a]
            for b in common:
                if a != b: # a and b are distinct vertices
                    squares.append((a, b))
        
        if len(squares) < 2:
            return False
        
        # Check pairs of squares
        num_squares = len(squares)
        for i in range(num_squares):
            a, b = squares[i]
            for j in range(i + 1, num_squares):
                c, d = squares[j]
                
                # Check distinctness
                if a == c or a == d or b == c or b == d:
                    continue
                if a == b or c == d:
                    continue
                
                used = {u, v, a, b, c, d}
                
                # e candidates
                e_cands = (adj_sets[a] & adj_sets[c]) - used
                if not e_cands:
                    continue
                
                # f candidates
                f_cands = (adj_sets[b] & adj_sets[d]) - used
                if not f_cands:
                    continue
                
                # Check connection
                # any(e in e_cands such that adj[e] & f_cands is not empty)
                # We can iterate over the smaller set
                if len(e_cands) < len(f_cands):
                    for e in e_cands:
                        if adj_sets[e] & f_cands:
                            return True
                else:
                    for f in f_cands:
                        if adj_sets[f] & e_cands:
                            return True
                            
        return False

    best_edges = 0
    best_graph = set()
    
    # Local Search
    for restart in range(2000):
        if time.time() > deadline:
            break
        
        # Initialize graph
        random.shuffle(all_edges)
        current_graph = set()
        
        for edge in all_edges:
            u, v = edge
            if not creates_q3_sets(u, v):
                current_graph.add(edge)
                adj_sets[u].add(v)
                adj_sets[v].add(u)
        
        # Local Improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any missing edge
            for edge in all_edges:
                if edge in current_graph:
                    continue
                u, v = edge
                if not creates_q3_sets(u, v):
                    current_graph.add(edge)
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)
                    improved = True
                    break
            
            # 2. If no edge added, try a 1-remove 1-add move
            if not improved:
                if current_graph:
                    remove_edge = random.choice(list(current_graph))
                    u, v = remove_edge
                    current_graph.remove(remove_edge)
                    adj_sets[u].remove(v)
                    adj_sets[v].remove(u)
                    
                    added = False
                    random.shuffle(all_edges)
                    for edge in all_edges:
                        if edge == remove_edge:
                            continue
                        if edge in current_graph:
                            continue
                        u2, v2 = edge
                        if not creates_q3_sets(u2, v2):
                            current_graph.add(edge)
                            adj_sets[u2].add(v2)
                            adj_sets[v2].add(u2)
                            added = True
                            break
                    
                    if not added:
                        # Revert
                        current_graph.add(remove_edge)
                        adj_sets[u].add(v)
                        adj_sets[v].add(u)
                        break
                    else:
                        improved = True
                        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
            
    return [[u, v] for u, v in best_graph]
