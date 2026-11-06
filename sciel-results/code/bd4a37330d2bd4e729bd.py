import random
from itertools import combinations

def construct():
    # Target: Maximize n such that G is triangle-free and alpha(G) < 10.
    # Strategy: Start with Andrásfai graph AR_k (triangle-free, alpha=k).
    # We need alpha < 10, so start with k=10 (n=29, alpha=10) and try to reduce alpha to 9 by adding edges.
    # If that fails, fall back to k=9 (n=26, alpha=9).
    
    def generate_andrasfai(k):
        n = 3 * k - 1
        edges = set()
        # Differences for edges in AR_k: 1..k-1 and 2k
        diffs = set(range(1, k)) | {2 * k}
        for u in range(n):
            for d in diffs:
                v = (u + d) % n
                if u < v:
                    edges.add((u, v))
                else:
                    edges.add((v, u))
        return n, list(edges)

    def is_triangle_free_edges(edges, n):
        # Quick check using adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    # Check intersection of adj[u] and adj[v]
                    if adj[u].intersection(adj[v]):
                        return False
        return True

    def find_independent_set(edges, n, k=10):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Branch and bound
        # Use a heuristic: order vertices by degree?
        # Simple recursive search
        def dfs(candidates, current):
            if len(current) >= k:
                return current
            if not candidates:
                return None
            
            # Heuristic: pick vertex with min degree in induced subgraph? 
            # Or just first one. Let's try to be smart: 
            # If we pick a vertex with many non-neighbors, we might grow the set faster?
            # Actually, to find *any* set, standard B&B is fine.
            
            # Optimization: if len(candidates) + len(current) < k: return None
            if len(candidates) + len(current) < k:
                return None

            v = candidates[0]
            # Include v
            new_cands = [u for u in candidates[1:] if u not in adj[v]]
            res = dfs(new_cands, current + [v])
            if res:
                return res
            
            # Exclude v
            res = dfs(candidates[1:], current)
            if res:
                return res
            return None

        return dfs(list(range(n)), [])

    def try_improve(n, edges, max_iterations=1000):
        edges_set = set(tuple(e) for e in edges)
        
        for _ in range(max_iterations):
            indep_set = find_independent_set(list(edges_set), n, 10)
            if not indep_set:
                return list(edges_set) # Success!
            
            # Try to add an edge within indep_set to break it
            # We need an edge (u, v) in indep_set such that N(u) and N(v) are disjoint.
            found_edge = False
            for i in range(len(indep_set)):
                for j in range(i + 1, len(indep_set)):
                    u = indep_set[i]
                    v = indep_set[j]
                    if (u, v) in edges_set or (v, u) in edges_set:
                        continue
                    # Check if adding (u, v) creates triangle
                    # Triangle exists if there is w in N(u) intersect N(v)
                    # We need to check intersection of neighbors
                    # Note: u and v are not adjacent, so they are not in each other's neighbor set directly, 
                    # but they might share neighbors.
                    common = 0
                    # Efficient check:
                    # Since n is small (29), we can just check
                    for w in range(n):
                        if w != u and w != v:
                            if (u, w) in edges_set or (w, u) in edges_set:
                                if (v, w) in edges_set or (w, v) in edges_set:
                                    common += 1
                                    break
                    if common == 0:
                        # Safe to add
                        edges_set.add((u, v))
                        found_edge = True
                        break
                if found_edge:
                    break
            
            if not found_edge:
                # Cannot break this independent set without creating triangle
                # Try a different independent set? The B&B finds one. 
                # Maybe this one is "hard". 
                # For now, if we fail to find a safe edge for the *found* set, we might fail.
                # We could try randomizing the search for independent set to find an "easier" one.
                # But let's just break and return failure for this attempt.
                break
        
        return None

    # Attempt 1: n=29 (AR_10)
    n, edges = generate_andrasfai(10)
    improved = try_improve(n, edges)
    if improved:
        # Verify
        if is_triangle_free_edges(improved, n):
            if not find_independent_set(improved, n, 10):
                return {'n': n, 'edges': [list(e) for e in improved]}
    
    # Attempt 2: n=26 (AR_9) - Guaranteed valid
    n, edges = generate_andrasfai(9)
    # AR_9 has alpha=9, so it is valid.
    # Just to be safe, verify
    if is_triangle_free_edges(edges, n):
        if not find_independent_set(edges, n, 10):
            return {'n': n, 'edges': [list(e) for e in edges]}
            
    # Fallback (should not happen)
    return {'n': 1, 'edges': []}
