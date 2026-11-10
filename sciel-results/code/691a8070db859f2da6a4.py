import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_bits(x):
        return bin(x).count("1")

    def max_clique(adj, n):
        """
        Finds a maximum clique using a branch and bound approach 
        based on Bron-Kerbosch with pivot and coloring bound.
        Returns the list of vertices in the maximum clique.
        """
        # Use a simple recursive backtracking with upper bound
        # For n=64, this needs to be efficient.
        
        # We'll use a greedy coloring bound for the remaining candidates.
        # This is a standard technique for maximum clique (Tomita's algorithm).
        
        best_clique = []
        best_size = 0
        
        # We need a function to color the subgraph induced by R (candidates)
        # to get an upper bound on the clique size.
        
        def expand(clique, candidates):
            nonlocal best_clique, best_size
            
            if not candidates:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique[:]
                return
            
            # Greedy coloring of the subgraph induced by candidates
            # We want to order candidates by color number to prune effectively
            # Color classes provide an upper bound: max color number + current clique size
            # Actually, the bound is: if current_size + max_color < best_size, prune.
            
            # Implementing full Tomita coloring is complex. 
            # A simpler bound: count of candidates + current size.
            # This is weak. Let's try a slightly better heuristic bound.
            
            # Let's try a simple recursive search with a heuristic order.
            # Order candidates by degree in the remaining graph.
            
            # To speed this up, we can use a heuristic:
            # If we just want A large clique, greedy is often good enough for random graphs.
            # But the task asks for a DIFFERENT strategy.
            # The current program uses greedy.
            # Let's implement a proper branch and bound with a simple bound first.
            
            # Bound: len(clique) + count_bits(candidates)
            if len(clique) + count_bits(candidates) <= best_size:
                return
            
            # Pick a vertex to branch on.
            # Heuristic: pick vertex with max degree in candidates.
            # Or simply pick the first one.
            # Let's pick the one with max degree to find large cliques faster.
            
            best_v = -1
            best_deg = -1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                deg = count_bits(adj[v] & candidates)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                c ^= lsb
            
            if best_v == -1:
                return
            
            # Branch 1: Include best_v
            new_clique = clique + [best_v]
            new_candidates = candidates & adj[best_v]
            expand(new_clique, new_candidates)
            
            # Branch 2: Exclude best_v
            candidates &= ~(1 << best_v)
            expand(clique, candidates)

        # Initial call
        # To avoid deep recursion and stack overflow, and to handle n=64, 
        # we might need to be careful. 
        # However, random graphs with p=0.55 and n=64 have diameter 2 and large cliques.
        # The maximum clique size is likely around 6-8.
        
        expand([], (1 << n) - 1)
        return best_clique

    def clique_partition(adj, n, deadline):
        rem_adj = adj[:]
        cliques = []
        
        while True:
            if time.time() > deadline:
                break
            
            # Find a maximum clique in the remaining graph
            # We need to modify max_clique to work on rem_adj
            # My max_clique function above uses 'adj' from closure. 
            # I need to pass the adjacency matrix.
            
            # Let's refactor max_clique to take adj as argument.
            
            max_c = find_max_clique(rem_adj, n)
            
            if not max_c:
                break
            
            cliques.append(max_c)
            
            # Remove edges in this clique
            for u in max_c:
                for v in max_c:
                    if u < v:
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
            
            # Check if done
            has_edges = False
            for u in range(n):
                if rem_adj[u]:
                    has_edges = True
                    break
            if not has_edges:
                break
        
        return cliques

    def find_max_clique(adj, n):
        best_clique = []
        best_size = 0
        
        def expand(clique, candidates):
            nonlocal best_clique, best_size
            
            if not candidates:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique[:]
                return
            
            # Simple bound
            if len(clique) + count_bits(candidates) <= best_size:
                return
            
            # Heuristic: pick vertex with max degree in candidates
            best_v = -1
            best_deg = -1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                deg = count_bits(adj[v] & candidates)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                c ^= lsb
            
            if best_v == -1:
                return
            
            # Include best_v
            new_clique = clique + [best_v]
            new_candidates = candidates & adj[best_v]
            expand(new_clique, new_candidates)
            
            # Exclude best_v
            candidates &= ~(1 << best_v)
            expand(clique, candidates)

        expand([], (1 << n) - 1)
        return best_clique

    # Main execution
    start_time = time.time()
    # Reserve 10s for final processing
    deadline = start_time + 140
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    # The branch and bound might be slow if not optimized.
    # For n=64, the search space is huge. 
    # However, for dense random graphs, the maximum clique is unique or few, and found quickly.
    # But "removing edges" changes the graph.
    
    # Let's try it. If it's too slow, we fall back to greedy.
    
    try:
        cliques_1 = clique_partition(adj_1, 32, deadline)
        if time.time() > deadline:
             # Fallback for 1 if timed out? No, we should return best found.
             # But if we didn't finish, the partition is invalid (edges missing).
             # So we must ensure we finish or use a faster method.
             pass
    except:
        cliques_1 = []
        
    try:
        cliques_2 = clique_partition(adj_2, 64, deadline)
    except:
        cliques_2 = []
    
    # Validation and Fallback
    # If the B&B didn't cover all edges (due to timeout or bug), we must complete it with greedy or edges.
    
    def complete_partition(adj, n, initial_cliques):
        rem_adj = adj[:]
        for c in initial_cliques:
            for u in c:
                for v in c:
                    if u < v:
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
        
        # Check remaining edges
        # If there are remaining edges, add them as single edges or small cliques
        # To be safe and valid, add single edges for any remaining.
        # This minimizes risk of invalidity.
        # But we want to minimize count. 
        # Let's try to finish with greedy if time permits.
        
        # For now, just add single edges for any remaining to guarantee validity.
        # This might increase count, but ensures score > 0.
        # Actually, let's try a quick greedy finish if time allows.
        
        if time.time() < deadline - 5:
             # Greedy finish
             while True:
                 max_deg = 0
                 start_v = -1
                 for u in range(n):
                     deg = count_bits(rem_adj[u])
                     if deg > max_deg:
                         max_deg = deg
                         start_v = u
                 if max_deg == 0:
                     break
                 
                 clique = [start_v]
                 candidates = rem_adj[start_v]
                 while candidates:
                     best_cand = -1
                     best_deg = -1
                     c = candidates
                     while c:
                         lsb = c & -c
                         v = lsb.bit_length() - 1
                         deg = count_bits(rem_adj[v])
                         if deg > best_deg:
                             best_deg = deg
                             best_cand = v
                         c ^= lsb
                     if best_cand == -1:
                         break
                     clique.append(best_cand)
                     candidates &= rem_adj[best_cand]
                     
                 for u in clique:
                     for v in clique:
                         if u < v:
                             rem_adj[u] &= ~(1 << v)
                             rem_adj[v] &= ~(1 << u)
                 initial_cliques.append(clique)
        
        # Final safety: add any remaining edges as single edges
        for u in range(n):
            for v in range(u+1, n):
                if rem_adj[u] & (1 << v):
                    initial_cliques.append([u, v])
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
        
        return initial_cliques

    cliques_1 = complete_partition(adj_1, 32, cliques_1)
    cliques_2 = complete_partition(adj_2, 64, cliques_2)
    
    return [cliques_1, cliques_2]
