<<<<<<< SEARCH
        for k in range(19, 0, -1):
            tree_edges = trees[k-1]
            # avail_edges is all_edges - used_edges
            # For efficiency, we pass the set
            mapping = find_embedding(tree_edges, all_edges - used_edges)
            if mapping is None:
                # Fallback: try random restarts or different root?
                # For now, if it fails, we might return what we have.
                # But it should succeed.
                pass
            else:
                result[k-1] = mapping
                add_edges(tree_edges, mapping, used_edges)
=======
        # Maintain avail_edges for efficiency
        avail_edges = set(all_edges)
        
        for k in range(19, 0, -1):
            tree_edges = trees[k-1]
            mapping = find_embedding(tree_edges, avail_edges)
            if mapping is None:
                pass
            else:
                result[k-1] = mapping
                # Update used_edges and avail_edges
                for u, v in tree_edges:
                    hu = mapping[u]
                    hv = mapping[v]
                    if hu > hv: hu, hv = hv, hu
                    used_edges.add((hu, hv))
                    if (hu, hv) in avail_edges:
                        avail_edges.remove((hu, hv))
>>>>>>> REPLACE
