"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute a good static ordering: vertices with higher degree first,
    # and among same degree, those with more "structurally central" neighbors first.
    # For this graph, all vertices have degree 4, so we use a BFS-based ordering
    # from a high-degree start to create a nice ordering.
    
    # Actually, let's use a simple heuristic: order by (degree descending, then by index)
    # But since all have degree 4, let's try a different approach.
    
    # Let's use a greedy ordering based on the structure.
    # The graph appears to be a 4-regular graph on 50 vertices.
    # Let's just use the natural ordering 0, 1, 2, ..., 49 but with a twist:
    # we'll process vertices in an order that maximizes constraint propagation.
    
    # For now, use the static order sorted by degree descending, then by index.
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    
    # Precompute for each vertex, the list of neighbors
    # This is already done in adj
    
    # Check if a color assignment is valid:
    # 1. Proper coloring: adjacent vertices have different colors
    # 2. No bicolored cycle: for any two colors c1, c2, the subgraph induced by vertices
    #    with color c1 or c2 must be a forest (no cycles)
    
    def is_valid(color, v, c, deadline):
        """Check if assigning color c to vertex v is valid given current color assignment."""
        if time.time() > deadline:
            return False
        
        # Check properness: no neighbor has color c
        for u in adj[v]:
            if color[u] == c:
                return False
        
        # Check bicolored cycle constraint:
        # For each color d that appears in the neighborhood of v (d != c),
        # we need to check that adding v with color c doesn't create a cycle
        # in the subgraph induced by colors {c, d}.
        
        # Get all distinct colors in the neighborhood
        neighbor_colors = set()
        for u in adj[v]:
            if color[u] != -1:
                neighbor_colors.add(color[u])
        
        for d in neighbor_colors:
            if d == c:
                continue
            
            # Find all neighbors of v that have color d
            neighbors_with_d = [u for u in adj[v] if color[u] == d]
            
            # If there are fewer than 2 such neighbors, no cycle can be formed
            # (a cycle would need at least 2 neighbors of v with color d that are
            # connected in the {c, d}-subgraph)
            if len(neighbors_with_d) < 2:
                continue
            
            # Check if any two neighbors in neighbors_with_d are in the same
            # connected component of the subgraph induced by colors {c, d}.
            # We do this by doing a BFS/DFS from each neighbor and checking
            # if we can reach another neighbor.
            
            # To be efficient, we only need to check connectivity among the
            # neighbors_with_d in the {c, d}-subgraph.
            
            # Use union-find or BFS to check connectivity.
            # Since the graph is small, we can do a simple BFS.
            
            # Build the set of vertices with color c or d that are reachable
            # from each neighbor in neighbors_with_d.
            
            visited = set()
            for start in neighbors_with_d:
                if start in visited:
                    continue
                
                # BFS from start in the {c, d}-subgraph
                queue = deque([start])
                visited.add(start)
                
                while queue:
                    curr = queue.popleft()
                    for nxt in adj[curr]:
                        if color[nxt] in (c, d):
                            if nxt in neighbors_with_d and nxt != start:
                                # Found a path between two neighbors of v with color d
                                # This means adding v creates a cycle in the {c, d}-subgraph
                                return False
                            if nxt not in visited:
                                visited.add(nxt)
                                queue.append(nxt)
        
        return True
    
    def backtrack(idx, color, k, deadline):
        """Backtracking search with k colors."""
        if time.time() > deadline:
            return None
        
        if idx == n:
            return color[:]
        
        v = order[idx]
        
        # Try each color from 0 to k-1
        for c in range(k):
            if is_valid(color, v, c, deadline):
                color[v] = c
                res = backtrack(idx + 1, color, k, deadline)
                if res is not None:
                    return res
                color[v] = -1
        
        return None
    
    # Try increasing k from 3 upward
    deadline = time.time() + 80.0
    
    for k in range(3, 15):
        if time.time() > deadline - 2.0:
            break
        
        color = [-1] * n
        res = backtrack(0, color, k, deadline)
        if res is not None:
            # Verify the solution
            # Check properness
            for u, v in edges:
                if res[u] == res[v]:
                    break
            else:
                # Check bicolored cycles
                valid = True
                for c1 in range(k):
                    for c2 in range(c1 + 1, k):
                        # Check if the subgraph induced by colors c1 and c2 is a forest
                        # A graph is a forest iff |E| <= |V| - 1 for each connected component
                        # Or equivalently, no cycles exist
                        
                        # Count vertices and edges in the subgraph
                        vertices_in_subgraph = set()
                        edges_in_subgraph = 0
                        for i in range(n):
                            if res[i] in (c1, c2):
                                vertices_in_subgraph.add(i)
                        for u, v in edges:
                            if res[u] in (c1, c2) and res[v] in (c1, c2):
                                edges_in_subgraph += 1
                        
                        # A forest on v vertices has at most v - 1 edges
                        # But we need to check per connected component
                        # For simplicity, check if edges >= vertices (which would indicate a cycle
                        # if the graph is connected, but for disconnected graphs, we need per-component check)
                        
                        # Let's do a proper cycle check using DFS
                        visited = set()
                        has_cycle = False
                        for start in vertices_in_subgraph:
                            if start in visited:
                                continue
                            # DFS to detect cycle
                            stack = [(start, -1)]
                            visited.add(start)
                            while stack:
                                curr, parent = stack.pop()
                                for nxt in adj[curr]:
                                    if res[nxt] in (c1, c2):
                                        if nxt == parent:
                                            continue
                                        if nxt in visited:
                                            has_cycle = True
                                            break
                                        visited.add(nxt)
                                        stack.append((nxt, curr))
                                if has_cycle:
                                    break
                            if has_cycle:
                                break
                        
                        if has_cycle:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    return [int(x) for x in res]
    
    # Fallback: use all distinct colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
