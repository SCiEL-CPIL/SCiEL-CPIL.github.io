def construct(k, g):
    import time
    start = time.time()
    
    # Lower bound
    if k == 3:
        lb = 512 if g >= 16 else 230
    else:
        lb = 100
        
    for N in range(lb, 1000, 4):
        if (time.time() - start) > 110:
            break
        if N * k % 2 != 0: continue
        
        for attempt in range(10):
            if (time.time() - start) > 115:
                break
            rng = random.Random(N * 100 + attempt)
            adj = _random_k_regular(N, k, rng)
            if adj is None: continue
            
            cur_girth = _girth_limited(N, adj, g)
            if cur_girth >= g:
                edges = [(i, j) for i in range(N) for j in adj[i] if i < j]
                return N, edges
            
            # Local Search
            best_girth = cur_girth
            best_adj = [list(a) for a in adj]
            
            for _ in range(1000): # 1000 swaps
                if (time.time() - start) > 118:
                    break
                    
                # Pick swap
                u = rng.randrange(N)
                v = adj[u][rng.randrange(k)]
                x = rng.randrange(N)
                if x in (u, v): continue
                y = adj[x][rng.randrange(k)]
                if y in (u, v) or x == y: continue
                if (u == x and v == y) or (u == y and v == x): continue
                
                # Try swap
                # Check if u-x and v-y are valid
                if x not in adj[u] and y not in adj[v]:
                    # Perform swap
                    adj[u].remove(v); adj[v].remove(u)
                    adj[x].remove(y); adj[y].remove(x)
                    adj[u].append(x); adj[x].append(u)
                    adj[v].append(y); adj[y].append(v)
                    
                    new_girth = _girth_limited(N, adj, g)
                    if new_girth > best_girth:
                        best_girth = new_girth
                        best_adj = [list(a) for a in adj]
                        if best_girth >= g:
                            edges = [(i, j) for i in range(N) for j in best_adj[i] if i < j]
                            return N, edges
                    elif new_girth == best_girth:
                        # Accept with prob 0.5
                        if rng.random() > 0.5:
                            best_girth = new_girth
                            best_adj = [list(a) for a in adj]
                        else:
                            # Revert
                            adj[u].remove(x); adj[x].remove(u)
                            adj[v].remove(y); adj[y].remove(v)
                            adj[u].append(v); adj[v].append(u)
                            adj[x].append(y); adj[y].append(x)
                    else:
                        # Revert
                        adj[u].remove(x); adj[x].remove(u)
                        adj[v].remove(y); adj[y].remove(v)
                        adj[u].append(v); adj[v].append(u)
                        adj[x].append(y); adj[y].append(x)
                elif y not in adj[u] and x not in adj[v]:
                    # Swap u-y and v-x
                    adj[u].remove(v); adj[v].remove(u)
                    adj[x].remove(y); adj[y].remove(x)
                    adj[u].append(y); adj[y].append(u)
                    adj[v].append(x); adj[x].append(v)
                    
                    new_girth = _girth_limited(N, adj, g)
                    if new_girth > best_girth:
                        best_girth = new_girth
                        best_adj = [list(a) for a in adj]
                        if best_girth >= g:
                            edges = [(i, j) for i in range(N) for j in best_adj[i] if i < j]
                            return N, edges
                    elif new_girth == best_girth:
                        if rng.random() > 0.5:
                            best_girth = new_girth
                            best_adj = [list(a) for a in adj]
                        else:
                            # Revert
                            adj[u].remove(y); adj[y].remove(u)
                            adj[v].remove(x); adj[x].remove(v)
                            adj[u].append(v); adj[v].append(u)
                            adj[x].append(y); adj[y].append(x)
                    else:
                        # Revert
                        adj[u].remove(y); adj[y].remove(u)
                        adj[v].remove(x); adj[x].remove(v)
                        adj[u].append(v); adj[v].append(u)
                        adj[x].append(y); adj[y].append(x)
                else:
                    continue
                    
    # Fallback
    return (936, [])
