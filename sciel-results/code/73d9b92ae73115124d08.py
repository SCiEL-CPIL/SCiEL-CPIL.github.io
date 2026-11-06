<<<<<<< SEARCH
        def find_best_cycle():
            """
            Find a cycle that maximizes the number of edges removed.
            We prioritize longer cycles.
            Uses DFS with backtracking to find the longest cycle within time limits.
            """
            if not remaining_edges:
                return None
            
            # Identify candidate start vertices: those with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None
            
            # Sort candidates by degree (descending) to find large cycles faster
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            best_cycle = None
            best_len = 0
            
            # Time check inside the search
            search_deadline = time.time() + 0.5 # 0.5s per cycle extraction attempt
            
            for start in candidates[:10]: # Limit starts to avoid explosion
                if time.time() > search_deadline:
                    break
                
                # DFS to find cycles starting from 'start'
                # We want the longest cycle.
                # Path: [start, ..., current]
                # To close, current must connect to start.
                
                visited = {start}
                path = [start]
                
                # Stack for DFS: (current, next_index, path_copy)
                # To save memory, we modify path in place and backtrack
                
                # Standard DFS for longest cycle is NP-hard, but N is small.
                # We will do a limited DFS.
                
                # Optimization: Only try to close the cycle if the path length is >= 3
                # and the closing edge exists.
                
                # Instead of full DFS which might be too slow, we do a "Greedy Longest Path"
                # that backtracks only if it hits a dead end.
                
                def dfs(current, path):
                    nonlocal best_cycle, best_len, search_deadline
                    
                    if time.time() > search_deadline:
                        return
                    
                    # Check if we can close the cycle
                    if len(path) >= 3:
                        if (current, start) in remaining_edges or (start, current) in remaining_edges:
                            # Valid cycle found
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = list(path)
                            # We could continue to find even longer cycles, 
                            # but for performance, we might stop or continue.
                            # Let's continue to try to find longer ones if time permits.
                    
                    # Extend path
                    for neighbor in adj[current]:
                        if neighbor == start:
                            # Only allowed if we are closing the cycle (handled above)
                            # If we haven't reached min length, we can't close.
                            # If we can close, we did.
                            # We should not add start to path again.
                            continue
                        if neighbor not in visited:
                            # Check if edge (current, neighbor) is in remaining_edges
                            # It must be, because adj is built from remaining_edges.
                            visited.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor, path)
                            path.pop()
                            visited.remove(neighbor)
                            if time.time() > search_deadline:
                                return

                # Reset state for this start
                visited = {start}
                path = [start]
                dfs(start, path)
                
            if best_cycle:
                return best_cycle
            
            # Fallback: Find any cycle (e.g., triangle)
            # If no cycle found by DFS (maybe time ran out or graph is a tree), 
            # check for simple triangles.
            for u in list(adj.keys()):
                if len(adj[u]) < 2:
                    continue
                for v in list(adj[u]):
                    if v <= u: continue
                    for w in list(adj[v]):
                        if w == u or w == v: continue
                        if (u, w) in remaining_edges or (w, u) in remaining_edges:
                            # Triangle u-v-w-u
                            cycle = [u, v, w]
                            return cycle
            
            return None
=======
        def find_best_cycle():
            """
            Find a cycle that maximizes the number of edges removed.
            Uses greedy path extension with limited backtracking.
            """
            if not remaining_edges:
                return None
            
            # Identify candidate start vertices: those with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None
            
            # Sort candidates by degree (descending) to find large cycles faster
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            best_cycle = None
            best_len = 0
            
            search_deadline = time.time() + 0.8
            
            for start in candidates[:15]:
                if time.time() > search_deadline:
                    break
                
                # Greedy longest path with backtracking
                visited = {start}
                path = [start]
                
                def greedy_dfs(current, path, depth=0):
                    nonlocal best_cycle, best_len, search_deadline
                    
                    if time.time() > search_deadline:
                        return
                    
                    # Check if we can close the cycle
                    if len(path) >= 3:
                        if (current, start) in remaining_edges or (start, current) in remaining_edges:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = list(path)
                    
                    # If depth is too high, stop exploring
                    if depth > 50:
                        return
                    
                    # Sort neighbors by degree (prefer high-degree nodes for longer paths)
                    neighbors = [n for n in adj[current] if n != start and n not in visited]
                    neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                    
                    for neighbor in neighbors[:8]:  # Limit branching factor
                        visited.add(neighbor)
                        path.append(neighbor)
                        greedy_dfs(neighbor, path, depth + 1)
                        path.pop()
                        visited.remove(neighbor)
                        if time.time() > search_deadline:
                            return
                
                greedy_dfs(start, path, 0)
                
            if best_cycle:
                return best_cycle
            
            # Fallback: Find any cycle (triangle)
            for u in list(adj.keys()):
                if len(adj[u]) < 2:
                    continue
                for v in list(adj[u]):
                    if v <= u: continue
                    for w in list(adj[v]):
                        if w == u or w == v: continue
                        if (u, w) in remaining_edges or (w, u) in remaining_edges:
                            return [u, v, w]
            
            return None
>>>>>>> REPLACE

