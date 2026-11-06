        # ... inside construct ...
        # After building valid_paths
        final_paths = []
        for p in valid_paths:
            if len(p) > 1 and p[0] == p[-1]:
                # It's a cycle
                # Break it: remove the last edge?
                # The path is v0-v1-...-vk-v0.
                # Edges: (v0,v1), ..., (vk,v0).
                # We can make a path v0-v1-...-vk.
                # And a path (vk, v0) -> [vk, v0].
                # This covers all edges.
                # Path 1: p[:-1]
                # Path 2: [p[-1], p[0]]
                final_paths.append(p[:-1])
                final_paths.append([p[-1], p[0]])
            else:
                final_paths.append(p)
        
        all_results.append(final_paths)
