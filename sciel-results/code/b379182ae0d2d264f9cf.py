def has_path_len_4(u, v, adj):
    if u == v:
        return False
    
    # BFS to find shortest path
    visited = {u: 0}
    queue = [u]
    found_shortest = -1
    while queue:
        node = queue.pop(0)
        dist = visited[node]
        if node == v:
            found_shortest = dist
            break
        if dist >= 4:
            continue # Don't expand further if we already reached depth 4
        for nb in adj[node]:
            if nb not in visited:
                visited[nb] = dist + 1
                queue.append(nb)
                
    if found_shortest == -1:
        return False
    if found_shortest == 4:
        return True
    if found_shortest < 4:
        # Need to check for other simple paths of length 4
        # Reset visited for DFS
        visited_dfs = {u}
        def dfs(node, depth):
            if depth == 4:
                return node == v
            for nb in adj[node]:
                if nb not in visited_dfs:
                    visited_dfs.add(nb)
                    if dfs(nb, depth + 1):
                        return True
                    visited_dfs.remove(nb)
            return False
        return dfs(u, 0)
    return False
