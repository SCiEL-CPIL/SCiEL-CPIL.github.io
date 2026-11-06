"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    N = 40
    M = 401
    
    # Represent graph using adjacency masks (Python ints)
    # adj_mask[i] is an integer where bit j is 1 if edge (i,j) exists
    def random_graph(n, m):
        adj = [0] * n
        edges = []
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if not (adj[u] >> v) & 1:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return adj, edges

    def compute_books(adj, n):
        """Compute book size for each edge. Returns list of (edge, book_size) and max book."""
        books = {}
        max_book = 0
        for u in range(n):
            # Iterate over set bits in adj[u]
            mask_u = adj[u]
            while mask_u:
                # Get lowest set bit
                lsb = mask_u & -mask_u
                v = lsb.bit_length() - 1
                if v > u: # Only process each edge once
                    common = (adj[u] & adj[v]).bit_count()
                    books[(u, v)] = common
                    if common > max_book:
                        max_book = common
                mask_u ^= lsb
        return books, max_book

    def evaluate_swap(adj, books, max_book, remove_edge, add_edge, n):
        """
        Evaluate the effect of removing remove_edge and adding add_edge.
        Returns the new max_book.
        """
        u1, v1 = remove_edge
        u2, v2 = add_edge
        
        # Check if edges are valid (remove exists, add doesn't)
        if not ((adj[u1] >> v1) & 1):
            return max_book # Invalid removal
        if (adj[u2] >> v2) & 1:
            return max_book # Invalid addition
        
        # We need to update books for edges affected by the change.
        # Affected edges are those incident to u1, v1, u2, v2.
        
        # To avoid modifying the original state, we can just calculate the new values
        # for the affected edges and determine the new global max.
        
        # The new max book is the max of:
        # 1. Books of edges NOT affected (which is at most the current max_book, 
        #    unless the current max_book edge is affected and decreases).
        # 2. New books of edges affected.
        
        # Let's identify the set of affected edges.
        affected_edges = set()
        
        # Edges incident to u1
        mask = adj[u1]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if v != v1: # Exclude the edge being removed itself for now? 
                        # Actually, the edge (u1,v1) is being removed.
                        # Its book size becomes 0 (or it's gone).
                        # Other edges (u1, v) might change if v1 was a common neighbor.
                        # v1 is a common neighbor of u1 and v if (v1, v) is an edge.
                        if (adj[v1] >> v) & 1:
                            affected_edges.add((min(u1, v), max(u1, v)))
            mask ^= lsb
            
        # Edges incident to v1
        mask = adj[v1]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if v != u1:
                if (adj[u1] >> v) & 1:
                    affected_edges.add((min(v1, v), max(v1, v)))
            mask ^= lsb
            
        # Edges incident to u2 (addition)
        mask = adj[u2]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if v != v2:
                if (adj[v2] >> v) & 1:
                    affected_edges.add((min(u2, v), max(u2, v)))
            mask ^= lsb
            
        # Edges incident to v2 (addition)
        mask = adj[v2]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if v != u2:
                if (adj[u2] >> v) & 1:
                    affected_edges.add((min(v2, v), max(v2, v)))
            mask ^= lsb
            
        # The new edge (u2, v2) itself
        affected_edges.add((min(u2, v2), max(u2, v2)))
        
        # The removed edge (u1, v1) is no longer in the graph.
        # If it was the max, the new max might drop.
        
        # Calculate new book sizes for affected edges
        # We need to simulate the adjacency change for these specific edges.
        # New adj for u1: remove v1, add v2? No, u1 is only affected by removal of v1.
        # Wait, the swap is: remove (u1,v1), add (u2,v2).
        # This only affects adjacency of u1, v1, u2, v2.
        
        new_adj = adj[:] # Copy is expensive? N=40, so list of 40 ints. Cheap.
        new_adj[u1] ^= (1 << v1)
        new_adj[v1] ^= (1 << u1)
        new_adj[u2] |= (1 << v2)
        new_adj[v2] |= (1 << u2)
        
        max_new_book = 0
        
        # We need to check the max book size of the *entire* graph.
        # Edges not in affected_edges keep their old book size.
        # If the current max_book edge is not in affected_edges, then max_new_book >= current max_book.
        # If it is in affected_edges, it might decrease.
        
        # Strategy:
        # 1. Find the max book size among affected edges in the new graph.
        # 2. Find the max book size among *unaffected* edges in the old graph.
        #    If the global max edge was affected, we need the second max, etc.
        #    Simpler: Just recompute the max for the whole graph?
        #    Recomputing whole graph is O(E * bit_count) ~ 400 * 10 = 4000 ops.
        #    If we do 100 candidates, that's 400,000 ops.
        #    In Python, 400k ops is ~0.05s.
        #    We have 145s. So we can do ~2900 batches of 100.
        #    This is feasible.
        
        # Let's just recompute the max book for the new_adj graph.
        # It's simple and robust.
        
        for u in range(n):
            mask_u = new_adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                v = lsb.bit_length() - 1
                if v > u:
                    common = (new_adj[u] & new_adj[v]).bit_count()
                    if common > max_new_book:
                        max_new_book = common
                mask_u ^= lsb
                
        return max_new_book

    # Initial Graph
    # Start with K_{20,20} + 1 edge? No, that's bad (book=20).
    # Start with a random graph.
    adj, edges = random_graph(N, M)
    books, current_max_book = compute_books(adj, N)
    
    best_max_book = current_max_book
    best_adj = adj[:]
    best_edges = edges[:]
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Generate batch of candidates
        batch_size = 200
        candidates = []
        for _ in range(batch_size):
            # Pick a random edge to remove
            if not edges:
                break
            r_idx = random.randint(0, len(edges) - 1)
            remove_edge = edges[r_idx]
            
            # Pick a random edge to add
            attempts = 0
            while attempts < 20:
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if not ((adj[u] >> v) & 1):
                    add_edge = (u, v)
                    break
                attempts += 1
            else:
                continue
            
            candidates.append((remove_edge, add_edge))
        
        best_candidate = None
        best_candidate_score = current_max_book
        
        for rem, add in candidates:
            score = evaluate_swap(adj, books, current_max_book, rem, add, N)
            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate = (rem, add)
        
        if best_candidate:
            rem, add = best_candidate
            # Apply swap
            u1, v1 = rem
            u2, v2 = add
            
            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            # Update edge list
            edges.remove(rem)
            edges.append(add)
            
            current_max_book = best_candidate_score
            no_improve_count = 0
            
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_adj = adj[:]
                best_edges = edges[:]
        else:
            no_improve_count += 1
            
            # If stuck, try a random restart or perturbation
            if no_improve_count % 100 == 0:
                # Perturb: swap 5 random edges
                for _ in range(5):
                    if not edges: break
                    r_idx = random.randint(0, len(edges) - 1)
                    rem = edges[r_idx]
                    u1, v1 = rem
                    adj[u1] ^= (1 << v1)
                    adj[v1] ^= (1 << u1)
                    edges.remove(rem)
                    
                    attempts = 0
                    while attempts < 20:
                        u = random.randint(0, N - 1)
                        v = random.randint(0, N - 1)
                        if u == v: continue
                        if u > v: u, v = v, u
                        if not ((adj[u] >> v) & 1):
                            add = (u, v)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges.append(add)
                            break
                        attempts += 1
                
                books, current_max_book = compute_books(adj, N)
                
    # Return best_edges
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
