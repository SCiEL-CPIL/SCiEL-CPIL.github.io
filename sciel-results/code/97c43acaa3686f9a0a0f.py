"""Rewritten program using maximum-weight matching approach for longest path."""
import random
from scipy.optimize import linear_sum_assignment

# Reconstruct the pinned graph exactly as in the seed
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build cost matrix for Hungarian algorithm
    # We split each vertex into in-node and out-node
    # The bipartite graph has 2n nodes: in[0..n-1] on left, out[0..n-1] on right
    # Edge from in[u] to out[v] exists if (u,v) is an edge in original graph
    # We want to find a maximum matching that corresponds to a set of vertex-disjoint paths
    
    # For the Hungarian algorithm, we minimize cost, so we use negative weights
    # Cost = 0 if edge exists, large penalty if not
    BIG = 1000000
    cost = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if v in adj[u]:
                cost[u][v] = 0  # Edge exists, no penalty
            else:
                cost[u][v] = BIG  # No edge, large penalty
    
    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # The matching gives us pairs (u, v) where u is "in" and v is "out"
    # This means u -> v is an edge in the path
    # We need to reconstruct paths from these edges
    
    # Build a mapping: for each vertex, what's its successor in the matching
    # in[u] -> out[v] means u is followed by v
    successor = [None] * n  # successor[u] = v means u -> v
    predecessor = [None] * n  # predecessor[v] = u means u -> v
    
    for u, v in zip(row_ind, col_ind):
        if cost[u][v] < BIG:  # Only if edge actually exists
            successor[u] = v
            predecessor[v] = u
    
    # Find all path segments
    # A vertex is a path start if it has no predecessor
    # A vertex is a path end if it has no successor
    
    # Find all connected components that are paths
    visited = [False] * n
    paths = []
    
    for i in range(n):
        if not visited[i] and predecessor[i] is None:
            # This is a start of a path
            path = [i]
            visited[i] = True
            cur = i
            while successor[cur] is not None:
                cur = successor[cur]
                if visited[cur]:
                    break
                path.append(cur)
                visited[cur] = True
            if len(path) > 1:
                paths.append(path)
    
    # Also handle vertices that are isolated (no edges in matching)
    for i in range(n):
        if not visited[i]:
            visited[i] = True
            # Check if this vertex is part of a cycle or isolated
            if successor[i] is not None and predecessor[i] is not None:
                # Could be in a cycle, trace it
                path = [i]
                cur = i
                while successor[cur] is not None and not visited[successor[cur]]:
                    cur = successor[cur]
                    path.append(cur)
                    visited[cur] = True
                if len(path) > 1:
                    paths.append(path)
            else:
                # Isolated vertex
                pass
    
    # Now we have a set of path segments. We want to concatenate them into one long path.
    # We can try to connect paths by checking if the end of one path connects to the start of another.
    # But first, let's just find the longest single path from the segments.
    
    # Actually, let's try a different approach: use the matching to get a set of edges,
    # then find the longest path in the resulting subgraph.
    
    # Build the subgraph from matched edges
    sub_adj = [set() for _ in range(n)]
    for u, v in zip(row_ind, col_ind):
        if cost[u][v] < BIG:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
    
    # Find longest path in this subgraph using DFS with memoization
    # The subgraph is a collection of paths and cycles, so we can handle it efficiently
    
    # First, find all connected components
    comp_visited = [False] * n
    components = []
    
    for i in range(n):
        if not comp_visited[i] and sub_adj[i]:
            comp = []
            stack = [i]
            comp_visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in sub_adj[node]:
                    if not comp_visited[neighbor]:
                        comp_visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
    
    # For each component, find the longest path
    best_path = []
    
    for comp in components:
        if len(comp) <= 1:
            continue
        
        # Build local adjacency for this component
        local_adj = {v: set() for v in comp}
        for v in comp:
            for u in sub_adj[v]:
                if u in local_adj:
                    local_adj[v].add(u)
        
        # Find endpoints (vertices with degree 1 in this component)
        endpoints = [v for v in comp if len(local_adj[v]) == 1]
        
        if len(endpoints) >= 2:
            # It's a path, try both directions
            for start in endpoints:
                path = [start]
                visited_local = {start}
                cur = start
                while True:
                    next_nodes = [u for u in local_adj[cur] if u not in visited_local]
                    if not next_nodes:
                        break
                    cur = next_nodes[0]
                    path.append(cur)
                    visited_local.add(cur)
                if len(path) > len(best_path):
                    best_path = path
        else:
            # It's a cycle, break it at any point
            if comp:
                start = comp[0]
                path = [start]
                visited_local = {start}
                cur = start
                while True:
                    next_nodes = [u for u in local_adj[cur] if u not in visited_local]
                    if not next_nodes:
                        break
                    cur = next_nodes[0]
                    path.append(cur)
                    visited_local.add(cur)
                if len(path) > len(best_path):
                    best_path = path
    
    # If we didn't find a good path from matching, fall back to greedy
    if len(best_path) < 2:
        # Use greedy approach as fallback
        best = []
        for start in range(n):
            path = [start]
            seen = {start}
            while adj[path[-1]] - seen:
                v = min(adj[path[-1]] - seen, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
            if len(path) > len(best):
                best = path
        best_path = best
    
    # Final verification: ensure it's a valid simple path
    if len(best_path) < 2:
        # Return at least a single vertex or a valid edge
        if edges:
            return [edges[0][0], edges[0][1]]
        return [0]
    
    # Verify all consecutive pairs are edges
    valid = True
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
            valid = False
            break
    
    if not valid:
        # Fall back to greedy
        best = []
        for start in range(n):
            path = [start]
            seen = {start}
            while adj[path[-1]] - seen:
                v = min(adj[path[-1]] - seen, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
            if len(path) > len(best):
                best = path
        best_path = best
    
    return best_path
