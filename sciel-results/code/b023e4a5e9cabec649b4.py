"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s out of 90s budget
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structures
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    # Precompute all simple paths of length 1 (direct edges) and length 2 (via one intermediate)
    # For each pair (u,v), store possible paths
    # Path of length 1: [u, v] if edge exists
    # Path of length 2: [u, w, v] if u-w and w-v are edges and w != u, w != v
    
    # We'll use a matching-based approach for finding disjoint paths
    
    def time_check():
        return time.time() > deadline
    
    def find_disjoint_paths(branches):
        """
        Given a set of branch vertices, try to find a simple path between each pair
        such that all interior vertices are distinct and don't include any branch.
        Returns dict mapping (u,v) -> path, or None if impossible.
        """
        if time_check():
            return None
        
        k = len(branches)
        if k < 2:
            return None
        
        # Get all pairs
        pairs = list(combinations(branches, 2))
        
        # For each pair, find candidate paths (shortest first for efficiency)
        # A path is valid if its interior vertices don't include any branch vertex
        branch_set = set(branches)
        
        # For each pair, enumerate possible paths up to some length
        # Start with direct edge, then length-2 paths, etc.
        
        pair_candidates = {}
        for u, v in pairs:
            candidates = []
            
            # Direct edge
            if (min(u,v), max(u,v)) in edge_set:
                candidates.append([u, v])
            
            # Length-2 paths: u - w - v
            for w in range(n):
                if w == u or w == v or w in branch_set:
                    continue
                if w in adj[u] and w in adj[v]:
                    candidates.append([u, w, v])
            
            # Length-3 paths: u - a - b - v
            for a in range(n):
                if a == u or a == v or a in branch_set:
                    continue
                if a not in adj[u]:
                    continue
                for b in range(n):
                    if b == u or b == v or b == a or b in branch_set:
                        continue
                    if b not in adj[v]:
                        continue
                    if a in adj[b]:
                        candidates.append([u, a, b, v])
            
            # Length-4 paths: u - a - b - c - v
            for a in range(n):
                if a == u or a == v or a in branch_set:
                    continue
                if a not in adj[u]:
                    continue
                for b in range(n):
                    if b == u or b == v or b == a or b in branch_set:
                        continue
                    if b not in adj[a]:
                        continue
                    for c in range(n):
                        if c == u or c == v or c == a or c == b or c in branch_set:
                            continue
                        if c not in adj[v]:
                            continue
                        if b in adj[c]:
                            candidates.append([u, a, b, c, v])
            
            if not candidates:
                return None
            pair_candidates[(u,v)] = candidates
        
        # Now we need to select one path per pair such that all interior vertices are disjoint
        # This is a constraint satisfaction / matching problem
        # Use backtracking with ordering
        
        # Order pairs by number of candidates (fewest first) for efficiency
        ordered_pairs = sorted(pairs, key=lambda p: len(pair_candidates[p]))
        
        # Interior vertices used so far
        used_interiors = set()
        
        def backtrack(idx, assignment):
            if time_check():
                return None
            
            if idx == len(ordered_pairs):
                return assignment
            
            u, v = ordered_pairs[idx]
            candidates = pair_candidates[(u,v)]
            
            # Try candidates in order
            for path in candidates:
                interiors = set(path[1:-1])
                if interiors & used_interiors:
                    continue
                
                # Check if this path is simple (no repeated vertices)
                if len(path) != len(set(path)):
                    continue
                
                used_interiors.update(interiors)
                assignment[(u,v)] = path
                
                result = backtrack(idx + 1, assignment)
                if result is not None:
                    return result
                
                used_interiors.difference_update(interiors)
                del assignment[(u,v)]
            
            return None
        
        return backtrack(0, {})
    
    # Search for largest clique subdivision
    # Strategy: try all subsets of vertices of size k, from large k down
    # For each subset, check if it forms a clique (all pairs connected by edges)
    # If so, we can use direct edges as paths (length 1, no interiors)
    # But we also need to check non-clique subsets where paths can be longer
    
    best_result = None
    best_k = 0
    
    # First, find the maximum clique size
    # Use Bron-Kerbosch or simple enumeration for n=24
    def find_max_clique():
        """Find all maximal cliques and return the largest."""
        cliques = []
        
        def bron_kerbosch(R, P, X):
            if time_check():
                return
            if not P and not X:
                cliques.append(list(R))
                return
            # Pivot selection
            if P:
                pivot = max(P, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return cliques
    
    # Find all maximal cliques
    all_cliques = find_max_clique()
    
    # Sort by size descending
    all_cliques.sort(key=len, reverse=True)
    
    # Try cliques from largest to smallest
    for clique in all_cliques:
        if time_check():
            break
        
        k = len(clique)
        if k <= best_k:
            continue
        
        branches = sorted(clique)
        
        # For a clique, direct edges work as paths
        # Check that all pairs have direct edges (they do by definition of clique)
        # Paths are just [u, v] for each pair
        paths = {}
        valid = True
        for u, v in combinations(branches, 2):
            if (min(u,v), max(u,v)) not in edge_set:
                valid = False
                break
            paths[(u,v)] = [u, v]
        
        if valid:
            best_k = k
            best_result = {
                'branches': list(branches),
                'paths': [list(paths[p]) for p in combinations(branches, 2)]
            }
    
    # Now try to extend beyond cliques by using longer paths
    # For subsets that are not cliques, try to find disjoint paths
    # Start from large subsets and work down
    
    if best_k < 2:
        # Fallback: at least 2 branches with a direct edge
        u, v = edges[0]
        return {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }
    
    # Try to find larger subdivisions using non-clique branch sets
    # Try subsets of size best_k+1, best_k+2, etc.
    # This is expensive, so limit the search
    
    max_try_k = min(n, best_k + 3)  # Try a few sizes above current best
    
    for try_k in range(max_try_k, best_k, -1):
        if time_check():
            break
        
        # Try random subsets of size try_k
        # Bias towards vertices with high degree
        degrees = [len(adj[i]) for i in range(n)]
        
        # Try several random subsets
        num_trials = 50
        for trial in range(num_trials):
            if time_check():
                break
            
            # Sample vertices with probability proportional to degree
            weights = [d + 1 for d in degrees]
            total_w = sum(weights)
            branches = set()
            while len(branches) < try_k:
                r = random.random() * total_w
                cum = 0
                for i in range(n):
                    cum += weights[i]
                    if r <= cum:
                        branches.add(i)
                        break
            
            branches = sorted(branches)
            branch_set = set(branches)
            
            result = find_disjoint_paths(branches)
            if result is not None:
                best_k = try_k
                best_result = {
                    'branches': list(branches),
                    'paths': [list(result[p]) for p in combinations(branches, 2)]
                }
                break  # Found one for this size, move to larger
    
    # Also try systematic search for sizes just above best_k
    for try_k in range(best_k + 1, min(n, best_k + 2)):
        if time_check():
            break
        
        # Try all combinations of high-degree vertices
        high_deg_vertices = sorted(range(n), key=lambda i: -len(adj[i]))[:12]
        
        for combo in combinations(high_deg_vertices, try_k):
            if time_check():
                break
            
            branches = sorted(combo)
            branch_set = set(branches)
            
            result = find_disjoint_paths(branches)
            if result is not None:
                best_k = try_k
                best_result = {
                    'branches': list(branches),
                    'paths': [list(result[p]) for p in combinations(branches, 2)]
                }
                break
    
    # Ensure we have a valid result
    if best_result is None:
        u, v = edges[0]
        best_result = {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }
    
    # Convert to proper types
    final_branches = [int(b) for b in best_result['branches']]
    final_paths = [[int(v) for v in p] for p in best_result['paths']]
    
    return {
        'branches': final_branches,
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
