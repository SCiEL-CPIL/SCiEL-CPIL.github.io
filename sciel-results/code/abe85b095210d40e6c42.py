    def has_q3_with_edge(adj, u, v):
        nu = [x for x in range(n) if adj[u][x] and x != v]
        nv = [x for x in range(n) if adj[v][x] and x != u]
        
        if len(nu) < 2 or len(nv) < 2:
            return False
            
        for i in range(len(nu)):
            for j in range(i+1, len(nu)):
                u1 = nu[i]
                u2 = nu[j]
                for k in range(len(nv)):
                    for l in range(k+1, len(nv)):
                        v1 = nv[k]
                        v2 = nv[l]
                        
                        if not (adj[u1][v1] and adj[u2][v2]):
                            continue
                            
                        # Find x, y
                        candidates = [x for x in range(n) if x not in (u, v, u1, u2, v1, v2)]
                        if len(candidates) < 2:
                            continue
                            
                        for idx_x in range(len(candidates)):
                            for idx_y in range(idx_x + 1, len(candidates)):
                                x = candidates[idx_x]
                                y = candidates[idx_y]
                                
                                if not adj[x][y]:
                                    continue
                                    
                                # Check Pattern A
                                if (adj[u1][x] and adj[u2][y] and adj[v1][y] and adj[v2][x]):
                                    return True
                                # Check Pattern B
                                if (adj[u1][y] and adj[u2][x] and adj[v1][x] and adj[v2][y]):
                                    return True
        return False
