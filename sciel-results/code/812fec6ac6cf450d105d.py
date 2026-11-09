import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_cycle(adj, n, rng):
        """
        Find a cycle in the graph using a randomized DFS.
        Returns a list of vertices in the cycle, or None if no cycle found.
        """
        visited = [False] * n
        parent = [-1] * n
        # Try multiple random starts
        for _ in range(10):
            start = rng.randint(0, n - 1)
            if not adj[start]:
                continue
            # DFS
            stack = [(start, -1)]
            visited[start] = True
            path = [start]
            while stack:
                node, par = stack.pop()
                # Shuffle neighbors for randomness
                neighbors = list(adj[node])
                rng.shuffle(neighbors)
                for nei in neighbors:
                    if not visited[nei]:
                        visited[nei] = True
                        parent[nei] = node
                        path.append(nei)
                        stack.append((nei, node))
                    elif nei != par and parent[nei] != -1 and parent[nei] != node:
                        # Found a back edge to an ancestor
                        # Extract cycle
                        cycle = [nei]
                        cur = node
                        while cur != nei:
                            cur = parent[cur]
                            cycle.append(cur)
                        # The cycle is nei -> ... -> node -> nei
                        # But we need to reverse the part from node back to nei
                        # Actually, the path from nei to node is nei, parent[nei], ..., node
                        # So the cycle is nei, ..., node, nei
                        # We have the path from start to node, and nei is an ancestor.
                        # Let's reconstruct properly.
                        # We have parent pointers.
                        # Cycle: start from nei, go up to node, then back to nei.
                        # But the path in DFS is not necessarily simple in the way we want.
                        # Let's do a proper DFS with a path stack.
                        pass
            # Reset visited for next start
            visited = [False] * n
            parent = [-1] * n
        
        # The above DFS is flawed for cycle extraction. Let's use a simpler method.
        # We'll use a standard DFS that records the path.
        visited = [False] * n
        for start in range(n):
            if visited[start] or not adj[start]:
                continue
            stack = [start]
            visited[start] = True
            path = [start]
            while stack:
                node = stack[-1]
                found = False
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
                        path.append(nei)
                        found = True
                        break
                    elif nei != path[0] and len(path) > 1 and nei in path:
                        # Found a cycle
                        idx = path.index(nei)
                        cycle = path[idx:]
                        # Ensure the cycle is valid (consecutive vertices are adjacent)
                        # And it's a simple cycle
                        if len(cycle) >= 3:
                            return cycle
                if not found:
                    stack.pop()
                    path.pop()
        return None

    def find_long_cycle(adj, n, rng, max_tries=20):
        """
        Try to find a long cycle by multiple random DFS attempts.
        """
        best_cycle = None
        for _ in range(max_tries):
            # Shuffle the order of starting vertices
            starts = list(range(n))
            rng.shuffle(starts)
            for start in starts:
                if not adj[start]:
                    continue
                visited = [False] * n
                stack = [start]
                visited[start] = True
                path = [start]
                while stack:
                    node = stack[-1]
                    found = False
                    # Shuffle neighbors to randomize
                    neighbors = list(adj[node])
                    rng.shuffle(neighbors)
                    for nei in neighbors:
                        if not visited[nei]:
                            visited[nei] = True
                            stack.append(nei)
                            path.append(nei)
                            found = True
                            break
                        elif nei in path and len(path) > 1:
                            idx = path.index(nei)
                            cycle = path[idx:]
                            if len(cycle) >= 3:
                                if best_cycle is None or len(cycle) > len(best_cycle):
                                    best_cycle = cycle[:]
                                # Break out of inner loops
                                stack = []
                                path = []
                                break
                    if not stack:
                        break
                if best_cycle and len(best_cycle) >= 5:  # Early stop if we found a decently long cycle
                    break
        return best_cycle

    def greedy_decompose(edges, n, rng, max_iterations=1000):
        """
        Greedily find cycles and remove them.
        Returns a list of pieces.
        """
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        pieces = []
        adj = build_adjacency(remaining_edges, n)
        
        for _ in range(max_iterations):
            # Find a cycle
            cycle = find_long_cycle(adj, n, rng)
            if not cycle:
                break
            
            # Extract edges from the cycle
            cycle_edges = set()
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in remaining_edges:
                    cycle_edges.add(e)
                else:
                    # This should not happen if adj is correct, but break if it does
                    cycle = None
                    break
            
            if not cycle or not cycle_edges:
                break
            
            # Remove these edges
            remaining_edges -= cycle_edges
            # Update adjacency
            for e in cycle_edges:
                u, v = e
                if v in adj[u]:
                    adj[u].remove(v)
                if u in adj[v]:
                    adj[v].remove(u)
            
            # Add the cycle as a piece
            # The cycle is a list of vertices. We need to output it in a consistent order.
            # The piece is a list of vertices in the cycle.
            pieces.append(cycle)
        
        # Add remaining edges as single-edge pieces
        for e in remaining_edges:
            pieces.append([e[0], e[1]])
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        best_pieces = None
        best_score = float('inf')
        
        # Multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
            rng = random.Random(seed * 1000 + restart)
            pieces = greedy_decompose(edges, n, rng)
            
            # Validate and count
            # Check edge-disjointness and coverage
            edge_count = {}
            valid = True
            for piece in pieces:
                if len(piece) == 2:
                    e = tuple(sorted(piece))
                    edge_count[e] = edge_count.get(e, 0) + 1
                elif len(piece) >= 3:
                    # Cycle
                    for i in range(len(piece)):
                        u = piece[i]
                        v = piece[(i + 1) % len(piece)]
                        e = tuple(sorted((u, v)))
                        edge_count[e] = edge_count.get(e, 0) + 1
                else:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check that all edges are covered exactly once
            if len(edge_count) != len(edges):
                continue
            for e in edges:
                if edge_count.get(e, 0) != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            score = len(pieces)
            if score < best_score:
                best_score = score
                best_pieces = pieces
        
        if best_pieces is None:
            # Fallback: all edges
            best_pieces = [[list(e)] for e in edges]
        
        results.append(best_pieces)
        
        if time.time() > deadline:
            # If we're running out of time, fill the rest with fallback
            for _ in range(len(specs) - len(results)):
                n2, p2, seed2 = specs[len(results)]
                edges2 = instance(n2, p2, seed2)
                results.append([[list(e)] for e in edges2])
            break
    
    return results

# Fix: The above code has a bug in the fallback loop. Let's correct it.
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_long_cycle(adj, n, rng, max_tries=20):
        """
        Try to find a long cycle by multiple random DFS attempts.
        """
        best_cycle = None
        for _ in range(max_tries):
            # Shuffle the order of starting vertices
            starts = list(range(n))
            rng.shuffle(starts)
            for start in starts:
                if not adj[start]:
                    continue
                visited = [False] * n
                stack = [start]
                visited[start] = True
                path = [start]
                while stack:
                    node = stack[-1]
                    found = False
                    # Shuffle neighbors to randomize
                    neighbors = list(adj[node])
                    rng.shuffle(neighbors)
                    for nei in neighbors:
                        if not visited[nei]:
                            visited[nei] = True
                            stack.append(nei)
                            path.append(nei)
                            found = True
                            break
                        elif nei in path and len(path) > 1:
                            idx = path.index(nei)
                            cycle = path[idx:]
                            if len(cycle) >= 3:
                                if best_cycle is None or len(cycle) > len(best_cycle):
                                    best_cycle = cycle[:]
                                # Break out of inner loops
                                stack = []
                                path = []
                                break
                    if not stack:
                        break
                if best_cycle and len(best_cycle) >= 5:  # Early stop if we found a decently long cycle
                    break
        return best_cycle

    def greedy_decompose(edges, n, rng, max_iterations=1000):
        """
        Greedily find cycles and remove them.
        Returns a list of pieces.
        """
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        pieces = []
        adj = build_adjacency(remaining_edges, n)
        
        for _ in range(max_iterations):
            # Find a cycle
            cycle = find_long_cycle(adj, n, rng)
            if not cycle:
                break
            
            # Extract edges from the cycle
            cycle_edges = set()
            valid_cycle = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in remaining_edges:
                    cycle_edges.add(e)
                else:
                    valid_cycle = False
                    break
            
            if not valid_cycle or not cycle_edges:
                break
            
            # Remove these edges
            remaining_edges -= cycle_edges
            # Update adjacency
            for e in cycle_edges:
                u, v = e
                if v in adj[u]:
                    adj[u].remove(v)
                if u in adj[v]:
                    adj[v].remove(u)
            
            # Add the cycle as a piece
            pieces.append(cycle)
        
        # Add remaining edges as single-edge pieces
        for e in remaining_edges:
            pieces.append([e[0], e[1]])
        
        return pieces

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        best_pieces = None
        best_score = float('inf')
        
        # Multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
            rng = random.Random(seed * 1000 + restart)
            pieces = greedy_decompose(edges, n, rng)
            
            # Validate and count
            edge_count = {}
            valid = True
            for piece in pieces:
                if len(piece) == 2:
                    e = tuple(sorted(piece))
                    edge_count[e] = edge_count.get(e, 0) + 1
                elif len(piece) >= 3:
                    # Cycle
                    for i in range(len(piece)):
                        u = piece[i]
                        v = piece[(i + 1) % len(piece)]
                        e = tuple(sorted((u, v)))
                        edge_count[e] = edge_count.get(e, 0) + 1
                else:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check that all edges are covered exactly once
            if len(edge_count) != len(edges):
                continue
            for e in edges:
                if edge_count.get(e, 0) != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            score = len(pieces)
            if score < best_score:
                best_score = score
                best_pieces = pieces
        
        if best_pieces is None:
            # Fallback: all edges
            best_pieces = [[list(e)] for e in edges]
        
        results.append(best_pieces)
        
        if time.time() > deadline:
            # If we're running out of time, fill the rest with fallback
            for remaining_spec in specs[len(results):]:
                n2, p2, seed2 = remaining_spec
                edges2 = instance(n2, p2, seed2)
                results.append([[list(e)] for e in edges2])
            break
    
    return results
