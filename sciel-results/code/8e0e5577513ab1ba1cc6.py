    def greedy_clique(start_node, nodes_list):
        clique = [start_node]
        available = set(nodes_list)
        available.remove(start_node)
        # Intersect with neighbors of start_node
        # Optimization: adj[start_node] is a set.
        # available = available & adj[start_node]
        available = available.intersection(adj[start_node])
        
        while len(clique) < target_size:
            if not available:
                break
            # Pick node with max degree in available
            # deg(n) = len(adj[n] & available)
            # This is O(|available| * |adj[n]|) which is expensive.
            # Approximation: Pick node with max len(adj[n])? 
            # Or just pick the first one?
            # Let's try max len(adj[n]) first, it's O(N).
            best_node = max(available, key=lambda x: len(adj[x]))
            clique.append(best_node)
            available.remove(best_node)
            available = available.intersection(adj[best_node])
            
        return clique
