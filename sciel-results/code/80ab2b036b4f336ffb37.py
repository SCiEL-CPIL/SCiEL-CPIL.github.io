"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    # Safety margin: evaluator kills at 90s, we aim to finish by 85s
    DEADLINE = 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency lists and sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    def time_up():
        return time.time() - t_start > DEADLINE

    def find_shortest_path(u, v, banned):
        """Find shortest path from u to v avoiding banned vertices (interior)."""
        if u == v:
            return None
        if v in adj[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt == v:
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def try_find_paths(branches, order):
        """
        Try to find disjoint paths for all pairs in the given order.
        Returns list of paths corresponding to combinations(branches, 2) or None.
        """
        bset = set(branches)
        used_interior = set()
        paths_dict = {}
        
        for a, b in order:
            # Banned: all branches (except a,b) and all previously used interiors
            banned = used_interior | (bset - {a, b})
            path = find_shortest_path(a, b, banned)
            if path is None:
                return None
            interior = set(path[1:-1])
            # Check for overlap (should be handled by banned, but double check)
            if interior & used_interior:
                return None
            if interior & bset:
                return None
            used_interior |= interior
            paths_dict[frozenset((a, b))] = path
            
        return paths_dict

    def validate_and_format(branches, paths_dict):
        """Validate the result and format for return."""
        if not branches or len(branches) < 2:
            return None
        bset = set(branches)
        if len(bset) != len(branches):
            return None
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if set(paths_dict.keys()) != expected_pairs:
            return None
        
        # Final validation pass
        used_interior = set()
        for fkey, path in paths_dict.items():
            if len(path) < 2:
                return None
            a, b = path[0], path[-1]
            if a not in bset or b not in bset:
                return None
            if a == b:
                return None
            interior = set(path[1:-1])
            if interior & bset:
                return None
            if interior & used_interior:
                return None
            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    return None
            used_interior |= interior
            
        # Format output
        out_paths = []
        for a, b in combinations(branches, 2):
            out_paths.append(paths_dict[frozenset((a, b))])
            
        return {'branches': branches, 'paths': out_paths}

    # Strategy: Search for largest k
    # Max possible k is n, but practically limited by connectivity and disjointness.
    # We start from a high bound and go down.
    # For n=24, let's try up to 12.
    
    best_result = None
    rng = random.Random(42)
    
    # Generate candidate sets for a given k
    def generate_candidates(k):
        cands = []
        # 1. Top k by degree
        cands.append(sorted_verts[:k])
        
        # 2. Top k-1 by degree + 1 random
        if k > 1:
            for _ in range(3):
                top = sorted_verts[:k-1]
                rest = [v for v in range(n) if v not in top]
                if rest:
                    cands.append(top + [rng.choice(rest)])
        
        # 3. Random samples
        for _ in range(10):
            cands.append(rng.sample(range(n), k))
            
        # 4. Top k by "cliquishness" (sum of neighbor degrees)
        centrality = [sum(adj[v]) for v in range(n)]
        cent_sorted = sorted(range(n), key=lambda x: centrality[x], reverse=True)
        cands.append(cent_sorted[:k])
        
        # 5. Mix
        for _ in range(5):
            half = k // 2
            if half > 0:
                c1 = sorted_verts[:half]
                c2 = rng.sample(sorted_verts[half:], k-half) if len(sorted_verts) > half else []
                cands.append(c1 + c2)
                
        return cands

    # Max clique size estimation for lower bound
    # Simple greedy clique
    def greedy_clique():
        remaining = set(range(n))
        clique = []
        while remaining:
            # Pick vertex with max degree in remaining
            v = max(remaining, key=lambda x: len(adj[x] & remaining))
            clique.append(v)
            remaining = remaining & adj[v]
        return clique

    # Use greedy clique size as a hint, but search higher
    g_clique = greedy_clique()
    start_k = min(12, n - 2) # Cap at 12 for performance, or n-2
    
    for k in range(start_k, 1, -1):
        if time_up():
            break
            
        candidates = generate_candidates(k)
        
        # Deduplicate candidates
        seen = set()
        unique_cands = []
        for c in candidates:
            key = frozenset(c)
            if key not in seen:
                seen.add(key)
                unique_cands.append(c)
                
        # Try each candidate
        for cand in unique_cands:
            if time_up():
                break
            
            # Try multiple orderings for this candidate
            pairs = list(combinations(cand, 2))
            
            # Ordering 1: By degree sum
            order1 = sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]])
            # Ordering 2: By common neighbors (fewer first = harder first?)
            order2 = sorted(pairs, key=lambda p: len(adj[p[0]] & adj[p[1]]))
            # Ordering 3: Random
            order3 = list(pairs)
            rng.shuffle(order3)
            
            orders = [order1, order2, order3]
            
            found = False
            for order in orders:
                paths_dict = try_find_paths(cand, order)
                if paths_dict:
                    result = validate_and_format(cand, paths_dict)
                    if result:
                        return result
                    else:
                        # Validation failed, try next ordering
                        continue
            
            # If no ordering worked, try more random orderings if we have time
            if not time_up():
                for _ in range(5):
                    order_r = list(pairs)
                    rng.shuffle(order_r)
                    paths_dict = try_find_paths(cand, order_r)
                    if paths_dict:
                        result = validate_and_format(cand, paths_dict)
                        if result:
                            return result

    # Fallback: Return the greedy clique if nothing else worked
    # Or just the max clique from Bron-Kerbosch logic if we want to be safe
    # Let's just use the greedy clique result
    if best_result is None:
        # Find a valid subdivision for the greedy clique
        # The greedy clique is a clique, so paths are just edges
        branches = g_clique
        if len(branches) >= 2:
            paths = []
            for u, v in combinations(branches, 2):
                if v in adj[u]:
                    paths.append([u, v])
                else:
                    # Should not happen for a clique, but just in case
                    paths.append([u, v]) # Invalid, but we assume clique
            return {'branches': branches, 'paths': paths}
        else:
            # Fallback to first edge
            u, v = edges[0]
            return {'branches': [u, v], 'paths': [[u, v]]}

    return best_result
# EVOLVE-BLOCK-END
