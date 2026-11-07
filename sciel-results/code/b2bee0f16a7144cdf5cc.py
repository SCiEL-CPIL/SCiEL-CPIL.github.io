        def forward_check(v, h, current_mapping):
            # v is assigned h. Check unassigned neighbors nb of v.
            for nb in adj[v]:
                if current_mapping[nb] != -1:
                    continue
                # nb is unassigned. 
                # Used hosts by assigned neighbors of nb:
                used_by_nb = set()
                for x in adj[nb]:
                    if current_mapping[x] != -1:
                        used_by_nb.add(current_mapping[x])
                # Also v is assigned h, and v is a neighbor of nb.
                used_by_nb.add(h)
                
                # Check if there is any h' in [0, n) not in used_by_nb 
                # such that (h, h') not in used_edges
                # Note: (h, h') is an edge in K20. 
                # If h' is in used_by_nb, it's invalid due to injectivity.
                # We need to find at least one valid h'.
                
                found = False
                for h_prime in range(n):
                    if h_prime in used_by_nb:
                        continue
                    edge = (min(h, h_prime), max(h, h_prime))
                    if edge in used_edges:
                        continue
                    # If we reach here, h_prime is a valid candidate for nb
                    found = True
                    break
                if not found:
                    return False
            return True
