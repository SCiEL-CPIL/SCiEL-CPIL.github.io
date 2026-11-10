        def find_longest_cycle(allowed_edges, adj_local):
            if not allowed_edges:
                return None
            
            best_cycle = None
            best_len = 0
            nodes = list(set([u for e in allowed_edges for u in e]))
            random.shuffle(nodes)
            
            # Limit total steps for this search
            steps = 0
            max_steps_for_search = 1000
            
            for start in nodes[:5]:
                visited = {start}
                path = [start]
                
                def dfs(u):
                    nonlocal best_cycle, best_len, steps
                    if time.time() > deadline - 5.0 or steps > max_steps_for_search:
                        return

                    steps += 1
                    if len(path) > 30:
                        return

                    if len(path) >= 3 and get_edge_key(u, start) in allowed_edges:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                    
                    neighbors = [v for v in adj_local.get(u, []) if get_edge_key(u, v) in allowed_edges]
                    neighbors = [v for v in neighbors if v not in visited]
                    random.shuffle(neighbors)
                    
                    for v in neighbors:
                        visited.add(v)
                        path.append(v)
                        dfs(v)
                        path.pop()
                        visited.remove(v)
                
                dfs(start)
                if steps > max_steps_for_search:
                    break

            return best_cycle
