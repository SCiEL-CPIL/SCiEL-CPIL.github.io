    def greedy_color(adj, num_vertices):
        # Order vertices by degree descending
        order = sorted(range(num_vertices), key=lambda x: len(adj[x]), reverse=True)
        colors = [-1] * num_vertices
        # We will assign colors in the order defined
        # We need to map back to original indices
        # Let's create a list of (degree, index) and sort
        # Actually, we can just iterate through the sorted order
        
        # To make it efficient, we can keep track of used colors for neighbors
        # But for N=30, simple check is fine.
        
        for idx in order:
            # Find smallest color not used by neighbors
            used_colors = set()
            for u in adj[idx]:
                if colors[u] != -1:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            colors[idx] = c
        
        # Build partition
        max_c = max(colors) if colors else 0
        partition = [[] for _ in range(max_c + 1)]
        for i in range(num_vertices):
            partition[colors[i]].append(i)
        return [p for p in partition if p]
