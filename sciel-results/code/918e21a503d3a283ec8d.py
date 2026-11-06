"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find largest connected component
    visited = [False] * n
    largest_cc = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(comp) > len(largest_cc):
                largest_cc = comp
    
    if not largest_cc:
        return [0]
    
    cc_set = set(largest_cc)
    cc_nodes = list(largest_cc)
    n_cc = len(cc_nodes)
    
    # Map neighbors within CC
    adj_cc = {u: [v for v in adj[u] if v in cc_set] for u in cc_nodes}
    
    best_path = []
    
    # Strategy 1: Find Diameter (longest shortest path)
    # This is a lower bound on the longest simple path.
    def bfs_farthest(start, adj_map, nodes_set):
        dist = {start: 0}
        parent = {start: None}
        queue = deque([start])
        farthest_node = start
        max_dist = 0
        while queue:
            u = queue.popleft()
            if dist[u] > max_dist:
                max_dist = dist[u]
                farthest_node = u
            for v in adj_map[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
        # Reconstruct path
        path = []
        curr = farthest_node
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path, farthest_node

    # Pick a random start, find farthest, then find farthest from that one
    start_node = random.choice(cc_nodes)
    path1, node1 = bfs_farthest(start_node, adj_cc, cc_set)
    path2, node2 = bfs_farthest(node1, adj_cc, cc_set)
    
    # path2 is the diameter path
    if len(path2) > len(best_path):
        best_path = path2
    
    # Strategy 2: Randomized Greedy with Warnsdorff's Rule
    # Try many random starts
    rng = random.Random(748)
    
    def greedy_warnsdorff(start):
        unvisited = set(cc_nodes)
        unvisited.remove(start)
        path = [start]
        curr = start
        
        while True:
            neighbors = [n for n in adj_cc[curr] if n in unvisited]
            if not neighbors:
                break
            
            # Warnsdorff: choose neighbor with fewest unvisited neighbors
            # We need to count unvisited neighbors for each candidate
            min_count = float('inf')
            best_next = None
            # To add randomness, we can pick from the top few or use a random tie-breaker
            candidates = []
            for nb in neighbors:
                # Count unvisited neighbors of nb
                # Note: nb is unvisited, so we look at its neighbors that are in unvisited
                # The count includes the neighbors that are NOT curr? 
                # Standard Warnsdorff: number of exits from the next square.
                # Here, if we go to nb, the "exits" are unvisited neighbors of nb.
                # curr is visited, so it's not an exit.
                exits = len([x for x in adj_cc[nb] if x in unvisited])
                candidates.append((exits, nb))
            
            # Sort by exits
            candidates.sort(key=lambda x: x[0])
            
            # Pick the first one, or randomize among the lowest few to avoid getting stuck in a specific local pattern
            # Let's just pick the minimum, with random tie-breaking
            min_exits = candidates[0][0]
            min_candidates = [c[1] for c in candidates if c[0] == min_exits]
            
            next_node = rng.choice(min_candidates)
            
            path.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
            
        return path

    # Run greedy multiple times
    # We can also try starting from "leaves" of the CC (degree 1 in CC)
    leaves = [u for u in cc_nodes if len(adj_cc[u]) == 1]
    starts = leaves + [rng.choice(cc_nodes) for _ in range(50)]
    
    for s in starts:
        p = greedy_warnsdorff(s)
        if len(p) > len(best_path):
            best_path = p
            
    # Strategy 3: Local Search / 2-opt like extension?
    # If we have a path, can we extend it?
    # The greedy method already extends until stuck.
    # Can we swap segments?
    # For now, the greedy + diameter should be significantly better than IDDFS which times out or finds short paths due to backtracking overhead.
    
    if not best_path:
        best_path = [cc_nodes[0]]
        
    return best_path
# EVOLVE-BLOCK-END
