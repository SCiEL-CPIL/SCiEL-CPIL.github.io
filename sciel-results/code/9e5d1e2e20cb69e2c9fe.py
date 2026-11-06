import time
import sys
from collections import defaultdict
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        """
        Decompose graph into edge-disjoint simple paths covering all edges.
        Strategy:
        1. Iteratively extract the longest simple path in the remaining graph.
        2. Use dynamic programming (Held-Karp) to find the exact longest simple path.
        3. Remove the edges of this path from the graph and repeat.
        4. If the graph becomes large, fall back to greedy heuristics to save time.
        """
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        original_edges = set(tuple(sorted(e)) for e in edges)
        remaining_edges = set(original_edges)
        
        paths = []
        
        def get_longest_path_dp(rem_edges, rem_adj):
            """
            Find the longest simple path in the subgraph defined by rem_edges.
            Uses DP over subsets of vertices.
            Returns the path as a list of vertices, or None if no edges.
            """
            if not rem_edges:
                return None
            
            vertices = sorted(set(v for e in rem_edges for v in e))
            n_v = len(vertices)
            
            # If the graph is too large for DP, return None to signal fallback
            if n_v > 22: 
                return None
            
            # Map vertex ids to indices for DP
            v_to_idx = {v: i for i, v in enumerate(vertices)}
            
            # DP table: dp[mask][i] = length of longest path ending at vertex i, 
            # using exactly the set of vertices in mask.
            # We also need to reconstruct the path, so we store parent pointers.
            # To save memory, we can store only the previous vertex in the path for each state.
            
            # Initialize
            # dp[mask][i] is a tuple (length, prev_vertex_idx)
            # mask is an integer bitmask
            # i is the index of the current vertex
            
            # Since n_v <= 22, 2^22 is about 4 million. 4M * 22 is ~88M entries.
            # This might be too slow/memory heavy in Python for many iterations.
            # Let's optimize: we only care about the longest path, not all paths.
            # But we need to extract the path.
            
            # Alternative: Use BFS/DFS with pruning? No, DP is better for exactness.
            # Let's limit DP to n_v <= 18 to be safe with time/memory in Python.
            if n_v > 18:
                return None
                
            size = 1 << n_v
            # dp[i][mask] ? Or dp[mask][i]?
            # dp[mask][i] = max length of path ending at i using vertices in mask.
            # We can use a list of lists.
            # dp = [[-1] * n_v for _ in range(size)]
            # parent = [[-1] * n_v for _ in range(size)]
            
            # To save memory, we can use arrays of size size*n_v.
            # But Python lists are slow. Let's try a dictionary-based memoization for reachable states only?
            # Or just use standard lists if n_v is small.
            
            dp = [[-1] * n_v for _ in range(size)]
            parent = [[-1] * n_v for _ in range(size)]
            
            # Initialize single vertex paths
            for i in range(n_v):
                mask = 1 << i
                dp[mask][i] = 1
                parent[mask][i] = -1
                
            best_len = 1
            best_mask = 0
            best_end = 0
            
            # Iterate over masks
            for mask in range(size):
                for i in range(n_v):
                    if dp[mask][i] == -1:
                        continue
                    # Try to extend path ending at i to neighbor j
                    v_i = vertices[i]
                    for v_j in rem_adj[v_i]:
                        j = v_to_idx[v_j]
                        if mask & (1 << j):
                            continue # j already in path
                        new_mask = mask | (1 << j)
                        new_len = dp[mask][i] + 1
                        if new_len > dp[new_mask][j]:
                            dp[new_mask][j] = new_len
                            parent[new_mask][j] = i
                            if new_len > best_len:
                                best_len = new_len
                                best_mask = new_mask
                                best_end = j
                                
            # Reconstruct path
            if best_len <= 1:
                return None
                
            path = []
            mask = best_mask
            end = best_end
            while end != -1:
                path.append(vertices[end])
                prev = parent[mask][end]
                if prev == -1:
                    break
                # Remove current vertex from mask to get previous state
                mask = mask ^ (1 << end)
                end = prev
                
            path.reverse()
            return path

        def greedy_longest_path(rem_edges, rem_adj):
            """
            Fallback: Greedy heuristic to find a long simple path.
            """
            if not rem_edges:
                return None
            
            # Start from a random vertex with high degree
            candidates = [v for v in rem_adj if len(rem_adj[v]) > 0]
            if not candidates:
                return None
            
            # Pick vertex with max degree
            start = max(candidates, key=lambda v: len(rem_adj[v]))
            
            path = [start]
            used = {start}
            current = start
            
            while True:
                neighbors = [nb for nb in rem_adj[current] if nb not in used]
                if not neighbors:
                    break
                # Pick neighbor with max degree (Warnsdorff-like)
                next_v = max(neighbors, key=lambda v: len(rem_adj[v]))
                path.append(next_v)
                used.add(next_v)
                current = next_v
                
            return path

        while remaining_edges:
            # Check time
            if time.time() > deadline:
                # Fallback: add remaining edges as individual paths
                for e in remaining_edges:
                    paths.append([e[0], e[1]])
                break
            
            # Get current adjacency for remaining edges
            rem_adj = defaultdict(set)
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                
            # Try DP first
            path = get_longest_path_dp(remaining_edges, rem_adj)
            
            # If DP fails (too large or no path), use greedy
            if path is None:
                path = greedy_longest_path(remaining_edges, rem_adj)
                
            if path is None:
                # Should not happen if remaining_edges is not empty
                break
                
            # Extract edges from path
            path_edges = set()
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                path_edges.add(e)
                
            # Verify path edges are in remaining_edges (safety)
            valid_path = True
            for e in path_edges:
                if e not in remaining_edges:
                    valid_path = False
                    break
                    
            if not valid_path:
                # If something went wrong, just take one edge
                e = next(iter(remaining_edges))
                path = [e[0], e[1]]
                path_edges = {e}
                
            # Add path to results
            paths.append(path)
            
            # Remove path edges from remaining
            remaining_edges -= path_edges
            
            # Update adj (optional, we rebuild it each loop)
            
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            break
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    # If we ran out of time, fill remaining with trivial edge paths
    while len(results) < 3:
        spec = specs[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
