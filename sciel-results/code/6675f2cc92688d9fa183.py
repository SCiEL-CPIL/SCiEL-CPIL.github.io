"""Clique partition of split graphs using structural properties."""
import time
from random import Random
import networkx as nx

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Identify C and I
        C = list(range(k))
        I = list(range(k, n))
        
        # Build cross-edge set
        cross_edges = set()
        for u, v in edges:
            if (u < k and v >= k) or (u >= k and v < k):
                if u >= k:
                    u, v = v, u
                cross_edges.add((u, v))
        
        # Baseline
        baseline_cliques = []
        if k >= 2:
            baseline_cliques.append(list(C))
        for u, v in cross_edges:
            baseline_cliques.append([u, v])
        
        best_cliques = baseline_cliques
        best_count = len(baseline_cliques)
        
        # Try different partition sizes for C
        # s is the target size of groups in C
        for s in range(2, k + 1):
            if time.time() > deadline - 5:
                break
            
            # Partition C into groups of size s
            groups = []
            for i in range(0, k, s):
                groups.append(C[i:i+s])
            
            # Check validity: if any group has size 1, it MUST be matched
            # If a group has size 1 and is not matched, it can't form a clique.
            
            # Build bipartite graph for matching
            # Left: indices of groups
            # Right: vertices in I
            # Edge if v in I is connected to all u in group
            
            # Precompute neighbors in I for each u in C
            # neighbors_in_I[u] = set of v in I such that (u,v) in cross_edges
            neighbors_in_I = {u: set() for u in C}
            for u, v in cross_edges:
                if u < k and v >= k:
                    neighbors_in_I[u].add(v)
            
            # Build adjacency for matching
            # match_left[g_idx] = list of v in I that can be assigned to group g_idx
            match_left = []
            for g in groups:
                # Intersection of neighbors of all u in g
                if not g:
                    continue
                common = neighbors_in_I[g[0]].copy()
                for u in g[1:]:
                    common &= neighbors_in_I[u]
                match_left.append(common)
            
            # Find Maximum Bipartite Matching
            # Using a simple DFS-based algorithm (Kuhn's algorithm)
            # Left size <= k/2 = 15, Right size = n-k = 60.
            # Very fast.
            
            match_right = {v: -1 for v in I}
            match_left_res = [-1] * len(groups)
            
            def dfs(u, visited):
                for v in match_left[u]:
                    if visited[v]:
                        continue
                    visited[v] = True
                    if match_right[v] == -1 or dfs(match_right[v], visited):
                        match_right[v] = u
                        match_left_res[u] = v
                        return True
                return False
            
            for u in range(len(groups)):
                visited = {v: False for v in I}
                dfs(u, visited)
            
            # Calculate score
            matched_count = 0
            matched_size_sum = 0
            valid = True
            for i, g in enumerate(groups):
                if match_left_res[i] != -1:
                    matched_count += 1
                    matched_size_sum += len(g)
                else:
                    if len(g) == 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            total_cliques = len(groups) + len(cross_edges) - matched_size_sum
            
            if total_cliques < best_count:
                best_count = total_cliques
                # Construct the cliques
                new_cliques = []
                
                # 1. Group cliques
                for i, g in enumerate(groups):
                    if match_left_res[i] != -1:
                        v = match_left_res[i]
                        new_cliques.append(g + [v])
                    else:
                        new_cliques.append(list(g))
                
                # 2. Remaining 2-cliques
                # We need to identify which cross edges are covered by the matched groups
                covered_cross_edges = set()
                for i, g in enumerate(groups):
                    if match_left_res[i] != -1:
                        v = match_left_res[i]
                        for u in g:
                            covered_cross_edges.add((u, v))
                
                for u, v in cross_edges:
                    if (u, v) not in covered_cross_edges:
                        new_cliques.append([u, v])
                
                best_cliques = new_cliques
        
        # Ensure types are Python int and list
        cleaned_cliques = []
        for c in best_cliques:
            cleaned_cliques.append([int(x) for x in c])
        
        return cleaned_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
