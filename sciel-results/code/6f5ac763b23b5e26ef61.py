    def color_and_bound(candidates):
        if not candidates:
            return Fraction(0, 1)
        
        nodes = []
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            nodes.append(idx)
            tmp ^= lsb
            
        # Sort by weight descending
        nodes.sort(key=lambda x: weights[x], reverse=True)
        
        colors = {}
        color_max_weights = []
        
        for node in nodes:
            # Find colors of neighbors that are already colored
            # Neighbors in candidates: adj[node] & candidates
            # But we only care about those already processed (in `colors`)
            # We can iterate through `nodes` and check if `other` is in `colors` and is a neighbor.
            # Better: Iterate through neighbors in `adj[node]` and check if they are in `colors`.
            
            neighbor_colors = set()
            # Get neighbors of node within candidates
            # nbs = adj[node] & candidates
            # Iterate through set bits of nbs
            nbs = adj[node] & candidates
            tmp_nbs = nbs
            while tmp_nbs:
                lsb = tmp_nbs & (-tmp_nbs)
                idx_nb = lsb.bit_length() - 1
                if idx_nb in colors:
                    neighbor_colors.add(colors[idx_nb])
                tmp_nbs ^= lsb
                
            c = 0
            while c in neighbor_colors:
                c += 1
                
            colors[node] = c
            if c < len(color_max_weights):
                if weights[node] > color_max_weights[c]:
                    color_max_weights[c] = weights[node]
            else:
                color_max_weights.append(weights[node])
                
        return sum(color_max_weights)
