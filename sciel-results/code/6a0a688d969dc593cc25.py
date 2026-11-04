    def check_girth(adj_sets, n, g):
        """
        Returns True if girth >= g.
        adj_sets: list of sets of neighbors.
        """
        for start in range(n):
            # BFS from start
            visited = {start: 0}
            queue = [(start, 0)]
            while queue:
                node, depth = queue.pop(0)
                if depth >= g - 1:
                    continue
                for neighbor in adj_sets[node]:
                    if neighbor == start and depth + 1 < g:
                        # Found cycle of length depth + 1
                        return False
                    if neighbor not in visited:
                        visited[neighbor] = depth + 1
                        queue.append((neighbor, depth + 1))
                    else:
                        # If we see a visited node, it might be a cycle.
                        # But BFS finds shortest path.
                        # If neighbor is start, we handled it.
                        # If neighbor is visited at depth d2, cycle length is depth + 1 + d2?
                        # No, standard girth check:
                        # If we encounter a node already visited, the cycle length is depth + 1 + visited[neighbor]?
                        # Not exactly.
                        # Simpler: If we reach a neighbor of start at depth < g-1, fail.
                        # Also, if we reach any node via a path of length L, and it was reached via a path of length L' < L, 
                        # the cycle is L + L' + 1?
                        # Actually, for girth, we just need to ensure no cycle < g.
                        # A cycle of length L exists if there is a path of length L-1 between two adjacent vertices.
                        # BFS from start: if we reach a node `u` at depth `d`, and `u` is a neighbor of `start`, 
                        # then we have a cycle of length `d+1`.
                        # So checking neighbors of start is sufficient?
                        # No. A cycle might not include start.
                        # But if a cycle of length < g exists, it contains some vertex.
                        # If we BFS from *every* vertex, we will find it.
                        # Optimization: Only BFS until depth g-1.
                        # If we find a back-edge to a node at depth < current_depth, 
                        # cycle length = current_depth - visited_depth + 1.
                        
            # The above logic is slightly flawed for general cycles not involving start.
            # Correct BFS girth check:
            # For each edge (u,v), check if there is a path from u to v of length < g-1 not using (u,v).
            # This is what the original code did, but slowly.
            # Alternative: 
            # Run BFS from each vertex up to depth g-1.
            # If we visit a node twice, or visit a neighbor of the start node, we have a cycle.
            # Specifically, if we reach a node `w` that is a neighbor of `start`, 
            # and the path length is `d`, then cycle length is `d+1`.
            # If `d+1 < g`, girth < g.
            # Does this catch all cycles?
            # Yes, because every cycle has a vertex. If we start BFS at that vertex, 
            # we will traverse the cycle and hit the neighbor of the start vertex.
            # So checking "is the current node a neighbor of start?" at each step is sufficient.
            
            pass

    def _fast_check_girth(adj_sets, n, g):
        for start in range(n):
            visited = {start: 0}
            queue = collections.deque([(start, 0)])
            while queue:
                node, depth = queue.popleft()
                if depth >= g - 1:
                    continue
                for neighbor in adj_sets[node]:
                    if neighbor == start:
                        if depth + 1 < g:
                            return False
                    elif neighbor not in visited:
                        visited[neighbor] = depth + 1
                        queue.append((neighbor, depth + 1))
        return True
    