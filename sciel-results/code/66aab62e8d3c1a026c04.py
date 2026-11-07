            # Check bicolored cycles
            valid = True
            # Group neighbors by color
            neighbors_by_color = {}
            for u in adj[v]:
                cu = coloring[u]
                if cu == -1 or cu == c:
                    continue
                if cu not in neighbors_by_color:
                    neighbors_by_color[cu] = []
                neighbors_by_color[cu].append(u)
            
            # For each color c' used by neighbors
            for c_prime, neighbors in neighbors_by_color.items():
                pair = (min(c, c_prime), max(c, c_prime))
                
                if pair not in uf_states:
                    parent = list(range(n))
                    rank = [0] * n
                    uf_states[pair] = (parent, rank)
                
                parent, rank = uf_states[pair]
                
                # Check if any two neighbors are in the same component
                seen_roots = set()
                for u in neighbors:
                    root = find(parent, u)
                    if root in seen_roots:
                        valid = False
                        break
                    seen_roots.add(root)
                
                if not valid:
                    break
            
            if not valid:
                continue
