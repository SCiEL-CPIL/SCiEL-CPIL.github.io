"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def is_adj(u, v):
        return (min(u,v), max(u,v)) in edge_set
    
    def find_max_clique():
        # Bron-Kerbosch with pivot
        best = []
        
        def bronk(R, P_set, X):
            nonlocal best
            if not P_set and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if not P_set:
                return
            # Pivot selection
            pivot_candidates = list(P_set | X)
            if not pivot_candidates:
                pivot = None
            else:
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
            candidates = list(P_set - adj[pivot]) if pivot is not None else list(P_set)
            for v in candidates:
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set = P_set - {v}
                X = X | {v}
        
        bronk(set(), set(range(n)), set())
        return best
    
    max_clique = find_max_clique()
    k = len(max_clique)
    
    # Try to find a clique subdivision with more branches than max clique
    # by using paths with internal vertices
    # Strategy: try to find k+1 branch vertices where pairs are connected by disjoint paths
    
    def find_path(u, v, forbidden):
        """Find a simple path from u to v avoiding forbidden internal vertices."""
        # BFS
        from collections import deque
        visited = {u}
        parent = {u: None}
        queue = deque([u])
        while queue:
            curr = queue.popleft()
            if curr == v:
                # Reconstruct path
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[curr]:
                if nb not in visited and nb not in forbidden:
                    visited.add(nb)
                    parent[nb] = curr
                    queue.append(nb)
        return None
    
    # Try to find larger subdivision
    # For each candidate set of branch vertices, check if we can find disjoint paths
    def try_subdivision(branches):
        """Try to find disjoint paths between all pairs of branches."""
        k = len(branches)
        if k < 2:
            return None
        pairs = list(combinations(branches, 2))
        paths = {}
        used_interiors = set()
        
        # Sort pairs by some heuristic (e.g., shortest path first)
        def path_length(p):
            return len(p) - 1
        
        # Try greedy: assign shortest path first
        pair_info = []
        for u, v in pairs:
            # Find shortest path avoiding used interiors and other branches
            forbidden = used_interiors | (set(branches) - {u, v})
            path = find_path(u, v, forbidden)
            if path is None:
                return None
            pair_info.append((u, v, path))
        
        # Sort by path length (shorter first to minimize conflict)
        pair_info.sort(key=lambda x: len(x[2]))
        
        assigned_paths = {}
        used_int = set()
        for u, v, path in pair_info:
            # Check if path interiors are still available
            interiors = set(path[1:-1])
            if interiors & used_int:
                return None
            assigned_paths[(u,v)] = path
            used_int |= interiors
        
        return assigned_paths
    
    # First, try the max clique with direct edges
    best_result = None
    if k >= 2:
        branches = list(max_clique)
        paths = {}
        for u, v in combinations(branches, 2):
            if is_adj(u, v):
                paths[(u,v)] = [u, v]
            else:
                paths = None
                break
        if paths is not None:
            best_result = (branches, paths)
    
    # Try to find larger subdivisions
    # Try adding one more vertex to the max clique
    if best_result is not None:
        current_k = len(best_result[0])
        for extra in range(n):
            if extra in best_result[0]:
                continue
            candidate_branches = best_result[0] + [extra]
            result = try_subdivision(candidate_branches)
            if result is not None:
                branches = candidate_branches
                paths = result
                if best_result is None or len(branches) > len(best_result[0]):
                    best_result = (branches, paths)
    
    # Try random sets of vertices
    import random
    random.seed(42)
    deadline = time.time() + 85
    
    # Try various sizes
    for size in range(6, 12):
        if time.time() > deadline:
            break
        for attempt in range(500):
            if time.time() > deadline:
                break
            candidates = random.sample(range(n), size)
            result = try_subdivision(candidates)
            if result is not None:
                if best_result is None or size > len(best_result[0]):
                    best_result = (candidates, result)
    
    # Try all subsets of size k+1, k+2, ...
    if best_result is not None:
        base_k = len(best_result[0])
        for size in range(base_k + 1, min(n, base_k + 4)):
            if time.time() > deadline:
                break
            # Try random subsets
            for attempt in range(2000):
                if time.time() > deadline:
                    break
                candidates = random.sample(range(n), size)
                result = try_subdivision(candidates)
                if result is not None:
                    best_result = (candidates, result)
                    break
    
    if best_result is None:
        # Fallback: use first edge
        u, v = edges[0]
        best_result = ([u, v], {(u,v): [u, v]})
    
    branches, paths = best_result
    # Convert to proper format
    branches_out = [int(b) for b in branches]
    paths_out = []
    for (u, v), path in paths.items():
        paths_out.append([int(x) for x in path])
    
    return {'branches': branches_out, 'paths': paths_out}
# EVOLVE-BLOCK-END
