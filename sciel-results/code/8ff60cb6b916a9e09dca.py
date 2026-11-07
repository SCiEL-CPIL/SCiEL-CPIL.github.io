    def solve(k, v_idx, colors, pair_parents):
        if time.time() > deadline:
            return None
        if v_idx == N:
            return colors[:]
        
        v = VERTEX_ORDER[v_idx]
        
        neighbor_colors = set()
        for u in ADJ[v]:
            if colors[u] != -1:
                neighbor_colors.add(colors[u])
                
        for c in range(k):
            if c in neighbor_colors:
                continue
                
            # Check cycle creation
            valid = True
            neighbors_by_color = defaultdict(list)
            for u in ADJ[v]:
                d = colors[u]
                if d != -1 and d != c:
                    neighbors_by_color[d].append(u)
                    
            for d, us in neighbors_by_color.items():
                if len(us) > 1:
                    key = (c, d) if c < d else (d, c)
                    if key in pair_parents:
                        parent = pair_parents[key]
                        # Find root of first neighbor
                        u0 = us[0]
                        # Ensure u0 is in parent
                        # It should be, because it was assigned earlier
                        ru0 = u0
                        while parent[ru0] != ru0:
                            ru0 = parent[ru0]
                        
                        for u in us[1:]:
                            ru = u
                            while parent[ru] != ru:
                                ru = parent[ru]
                            if ru == ru0:
                                valid = False
                                break
                    if not valid:
                        break
                        
            if not valid:
                continue
                
            colors[v] = c
            new_pair_parents = {k_p: p[:] for k_p, p in pair_parents.items()}
            
            for d, us in neighbors_by_color.items():
                key = (c, d) if c < d else (d, c)
                if key not in new_pair_parents:
                    new_pair_parents[key] = [-1] * N
                
                parent = new_pair_parents[key]
                
                # Init v
                if parent[v] == -1:
                    parent[v] = v
                    
                for u in us:
                    if parent[u] == -1:
                        parent[u] = u
                    # Union
                    rv = v
                    while parent[rv] != rv:
                        rv = parent[rv]
                    ru = u
                    while parent[ru] != ru:
                        ru = parent[ru]
                    if rv != ru:
                        parent[rv] = ru
                        
            result = solve(k, v_idx + 1, colors, new_pair_parents)
            if result is not None:
                return result
                
            colors[v] = -1
            
        return None
