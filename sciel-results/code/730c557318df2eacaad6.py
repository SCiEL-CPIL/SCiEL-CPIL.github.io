<<<<<<< SEARCH
    def local_search_swap(adj, deadline):
        """Try 1-1 swaps: remove one edge, add another, maintaining constraints."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        current_edges.append((u, v))
            random.shuffle(current_edges)
            
            # Try all possible swaps
            for u, v in current_edges:
                if time.time() >= deadline:
                    return adj, False
                # Remove edge (u,v)
                adj[u].remove(v)
                adj[v].remove(u)
                
                # Try adding a new edge
                found_swap = False
                for a, b in all_pairs:
                    if time.time() >= deadline:
                        break
                    if (a, b) == (u, v) or (a, b) == (v, u):
                        continue
                    if b in adj[a]:
                        continue
                    # Check triangle-free
                    common = adj[a] & adj[b]
                    if common:
                        continue
                    adj[a].add(b)
                    adj[b].add(a)
                    if not is_3_colorable(adj):
                        # Accept the swap
                        improved = True
                        found_swap = True
                        break
                    adj[a].remove(b)
                    adj[b].remove(a)
                
                if found_swap:
                    return adj, True
                
                # Restore edge (u,v)
                adj[u].add(v)
                adj[v].add(u)
        return adj, False
=======
    def local_search_swap(adj, deadline):
        """Try 1-1 swaps: remove one edge, add another, maintaining constraints."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        current_edges.append((u, v))
            random.shuffle(current_edges)
            
            # Try all possible swaps
            for u, v in current_edges:
                if time.time() >= deadline:
                    return adj, False
                # Remove edge (u,v)
                adj[u].remove(v)
                adj[v].remove(u)
                
                # Try adding a new edge
                found_swap = False
                # Pre-compute candidates for this state?
                # Just iterate all_pairs
                for a, b in all_pairs:
                    if time.time() >= deadline:
                        break
                    if (a, b) == (u, v) or (a, b) == (v, u):
                        continue
                    if b in adj[a]:
                        continue
                    # Check triangle-free
                    if adj[a] & adj[b]:
                        continue
                    adj[a].add(b)
                    adj[b].add(a)
                    if not is_3_colorable(adj):
                        # Accept the swap
                        improved = True
                        found_swap = True
                        break
                    adj[a].remove(b)
                    adj[b].remove(a)
                
                if found_swap:
                    return adj, True
                
                # Restore edge (u,v)
                adj[u].add(v)
                adj[v].add(u)
        return adj, False
>>>>>>> REPLACE

