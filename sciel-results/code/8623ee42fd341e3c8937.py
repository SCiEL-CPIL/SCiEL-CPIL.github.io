<<<<<<< SEARCH
    # Check for short cycles using BFS from each vertex (limited depth)
    for start in range(n):
        # BFS up to depth 4 (to detect cycles of length <= 8)
        visited = {}
        visited[start] = (-1, -1, 0)  # (parent, parent_of_parent, depth)
        queue = deque([(start, -1, -1, 0)])
        while queue:
            node, parent, gp, depth = queue.popleft()
            if depth >= 5:
                continue
            for nbr in adj[node]:
                if nbr == parent:
                    continue
                if nbr not in visited:
                    visited[nbr] = (node, parent, depth + 1)
                    queue.append((nbr, node, parent, depth + 1))
                else:
                    vparent, vgp, vdepth = visited[nbr]
                    if nbr == gp:
                        # Cycle of length depth + 1
                        if depth + 1 < 10:
                            return _construct_lw(4, 5)
                    elif vparent == node:
                        # Back edge, skip
                        pass
                    else:
                        # Check if this creates a short cycle
                        cycle_len = depth + vdepth + 1
                        if cycle_len < 10:
                            return _construct_lw(4, 5)
    
    return n, edges
=======
    # Check for short cycles using non-backtracking walks from each vertex
    # A cycle of length L < g means there's a non-backtracking walk of length L from v back to v
    for start in range(n):
        # Non-backtracking BFS: state is (node, prev_node)
        # Track minimum depth to reach each state
        min_depth = {}
        queue = deque()
        for nbr in adj[start]:
            min_depth[(nbr, start)] = 1
            queue.append((nbr, start, 1))
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= 5:  # We only need to check up to depth 4 for cycles < 10
                continue
            for nbr in adj[node]:
                if nbr == prev:
                    continue
                if nbr == start and depth + 1 < 10:
                    return _construct_lw(4, 5)
                state = (nbr, node)
                if state not in min_depth or min_depth[state] > depth + 1:
                    min_depth[state] = depth + 1
                    queue.append((nbr, node, depth + 1))
    
    return n, edges
>>>>>>> REPLACE
