        def find_longest_path(start, end):
            best_path = []
            visited = [False] * n
            path = []
            visited_count = 0
            
            def dfs(node):
                nonlocal best_path, visited_count
                visited[node] = True
                visited_count += 1
                path.append(node)
                
                if node == end:
                    if len(path) > len(best_path):
                        best_path = path[:]
                else:
                    # Pruning: if current path length + remaining unvisited nodes <= best path length
                    # Note: not all unvisited nodes are reachable, so this is a safe upper bound
                    if best_path and len(path) + (n - visited_count) <= len(best_path):
                        pass
                    else:
                        for nb in adj[node]:
                            if not visited[nb]:
                                dfs(nb)
                
                path.pop()
                visited[node] = False
                visited_count -= 1

            dfs(start)
            return best_path
