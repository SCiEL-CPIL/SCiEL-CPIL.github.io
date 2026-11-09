        if is_edge:
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            m -= 1
            
            # Check triples with u
            for t_idx in triple_index[u]:
                if t_idx in viol_set:
                    if not is_violating(t_idx):
                        viol_set.remove(t_idx)
            # Check triples with v
            for t_idx in triple_index[v]:
                if t_idx in viol_set:
                    if not is_violating(t_idx):
                        viol_set.remove(t_idx)
