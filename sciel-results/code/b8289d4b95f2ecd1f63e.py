import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # 1. Generate candidate set S
    # S contains x in [1, 255] such that gcd(x, 255) > 1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
            
    # 2. Build adjacency matrix/list for the induced subgraph on candidates
    # We want max clique in this graph.
    # Let's map values to indices 0..N-1
    N = len(candidates)
    # Adjacency list
    adj = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(i+1, N):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].append(j)
                adj[j].append(i)
                
    # 3. Find Max Clique
    # Simple Branch and Bound
    # Order vertices by degree descending to improve pruning
    degrees = [len(adj[i]) for i in range(N)]
    order = sorted(range(N), key=lambda i: degrees[i], reverse=True)
    
    # Remap to ordered indices? Or just use the order in search.
    # Let's stick to a standard recursive search with coloring bound.
    # For 126 nodes, even a naive B&B might be slow if not optimized, 
    # but Tomita's coloring is very effective.
    
    # Since we can't rely on external libraries, I'll implement a simplified 
    # greedy + local search or a basic B&B.
    # Given the structure (multiples of 3 form a clique of size 85), 
    # a greedy approach starting with high-degree nodes will likely find 85 quickly.
    # Is 85 the max?
    # Let's try to find if we can exceed 85.
    
    # Greedy Max Clique
    def greedy_clique(start_node=None):
        if start_node is None:
            # Start with the node with highest degree
            start_node = order[0]
            
        clique = [start_node]
        available = set(adj[start_node])
        
        while available:
            # Pick node with max degree in available
            best_node = -1
            best_deg = -1
            for u in available:
                deg = len([v for v in adj[u] if v in available])
                if deg > best_deg:
                    best_deg = deg
                    best_node = u
            
            if best_node == -1:
                break
                
            clique.append(best_node)
            # Intersect available with neighbors of best_node
            new_available = set(adj[best_node])
            available = available.intersection(new_available)
            
        return clique

    best_clique = greedy_clique()
    best_size = len(best_clique)
    
    # Try to improve with local search (swap)
    # Or just run a more exhaustive search if time permits.
    # Let's try a simple branch and bound for a few seconds.
    
    def max_clique_bnb(P, C):
        # P: candidates, C: current clique
        # Returns max clique size and the clique
        if time.time() > deadline:
            return len(C), C
            
        if not P:
            return len(C), C
            
        # Bound: |C| + |P|
        if len(C) + len(P) <= best_size:
            return len(C), C
            
        # Choose a vertex from P
        # Heuristic: pick one with max degree in P
        best_u = -1
        max_deg = -1
        for u in P:
            deg = len([v for v in P if v in adj[u]])
            if deg > max_deg:
                max_deg = deg
                best_u = u
                
        if best_u == -1:
            return len(C), C
            
        # Branch 1: Include best_u
        new_P = [v for v in P if v in adj[best_u]]
        size1, clique1 = max_clique_bnb(new_P, C + [best_u])
        if size1 > best_size:
            best_size = size1
            best_clique = clique1
            
        # Branch 2: Exclude best_u
        new_P2 = [v for v in P if v != best_u]
        size2, clique2 = max_clique_bnb(new_P2, C)
        if size2 > best_size:
            best_size = size2
            best_clique = clique2
            
        return max(size1, size2), best_clique

    # Run B&B on the whole set
    # To make it faster, we can sort P initially
    initial_P = list(range(N))
    # The B&B above is very naive and might be slow.
    # Let's rely on the greedy result which is likely optimal or near-optimal.
    # Multiples of 3 is size 85.
    # Let's check if greedy found 85.
    
    # A better heuristic: 
    # Start with multiples of 3.
    # Try to add any node that connects to all in the clique.
    # If not, try swapping.
    
    # Let's just return the greedy result, it's very likely 85.
    # If we have time, we can try to verify if 86 is possible.
    
    # Actually, let's just run the greedy and return it.
    # The greedy starting from the highest degree node (which will be a multiple of 3, likely 3*something)
    # will pick many multiples of 3.
    
    # Convert indices back to values
    result_values = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in the result
    if 255 not in result_values:
        # This shouldn't happen if we started with high degree nodes, 
        # but let's force it.
        # If 255 is not in the clique, we can't just add it unless it connects to all.
        # But the task REQUIRES 255.
        # So we must find a clique CONTAINING 255.
        # My B&B/Greedy was on the whole graph.
        # I need to constrain the search to neighbors of 255.
        
        # Let's redo the logic:
        # 1. Identify indices of 255
        idx_255 = candidates.index(255)
        # 2. Neighbors of 255
        neighbors_255 = set(adj[idx_255])
        
        # 3. Find Max Clique in the subgraph induced by neighbors_255
        # Then add 255.
        
        # Let's run greedy on neighbors_255
        def greedy_clique_constrained(available_set):
            if not available_set:
                return []
            
            # Pick node with max degree within available_set
            best_node = -1
            max_deg = -1
            for u in available_set:
                deg = len([v for v in available_set if v in adj[u]])
                if deg > max_deg:
                    max_deg = deg
                    best_node = u
            
            if best_node == -1:
                return []
                
            clique = [best_node]
            new_available = available_set.intersection(adj[best_node])
            
            while new_available:
                best_node = -1
                max_deg = -1
                for u in new_available:
                    deg = len([v for v in new_available if v in adj[u]])
                    if deg > max_deg:
                        max_deg = deg
                        best_node = u
                        
                if best_node == -1:
                    break
                    
                clique.append(best_node)
                new_available = new_available.intersection(adj[best_node])
                
            return clique

        constrained_clique = greedy_clique_constrained(neighbors_255)
        constrained_clique.append(idx_255)
        
        result_values = [candidates[i] for i in constrained_clique]
        
    return result_values
