import time
import random

def _has_triangle(edges, n):
    """Return a triangle (u, v, w) if exists, else None."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for u in range(n):
        for v in adj[u]:
            if v > u:
                common = adj[u] & adj[v]
                if common:
                    w = next(iter(common))
                    return (u, v, w)
    return None

def _has_independent_set_10(edges, n):
    """Return an independent set of size 10 if exists, else None."""
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edges:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
                
    def dfs(clique, candidates):
        if len(clique) == 10:
            return clique
        if len(clique) + len(candidates) < 10:
            return None
        
        # Heuristic: pick vertex with highest degree in candidates
        # This often finds a clique faster if one exists
        # Or lowest degree to prune?
        # For finding a clique, high degree is usually better.
        v = max(candidates, key=lambda x: len(comp_adj[x] & candidates))
        
        new_clique = clique + [v]
        new_candidates = candidates & comp_adj[v]
        new_candidates.discard(v)
        
        res = dfs(new_clique, new_candidates)
        if res is not None:
            return res
        
        # If backtracking, we need to try other vertices.
        # The above code only tries the 'best' vertex. 
        # We need to iterate through all candidates to be exact.
        # But for performance, we can limit the branching factor or use a different strategy.
        # Let's stick to full backtracking but with a good ordering.
        
        # Correct DFS:
        for v in sorted(candidates, key=lambda x: len(comp_adj[x] & candidates), reverse=True):
            new_clique = clique + [v]
            new_candidates = candidates & comp_adj[v]
            new_candidates.discard(v)
            
            res = dfs(new_clique, new_candidates)
            if res is not None:
                return res
            candidates.discard(v) # This line is wrong, candidates is a set, modifying it during iteration is bad.
        return None

    # Rewrite DFS properly
    def dfs_exact(clique, candidates):
        if len(clique) == 10:
            return clique
        if len(clique) + len(candidates) < 10:
            return None
        
        # Order candidates by degree to find solution faster
        # We need to iterate over all
        cand_list = list(candidates)
        # Sort by degree in candidates
        cand_list.sort(key=lambda x: len(comp_adj[x] & candidates), reverse=True)
        
        for v in cand_list:
            if v not in candidates:
                continue
            new_clique = clique + [v]
            new_candidates = candidates & comp_adj[v]
            new_candidates.discard(v)
            
            res = dfs_exact(new_clique, new_candidates)
            if res is not None:
                return res
            # Remove v from candidates for subsequent iterations in this level
            candidates.discard(v)
            
        return None

    return dfs_exact([], set(range(n)))

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Time allocation for each n
    time_alloc = {
        40: 30,
        39: 30,
        38: 20,
        37: 10,
        36: 5,
        35: 3,
        34: 2,
        33: 1,
        32: 1,
        31: 1,
        30: 1,
        29: 1,
        28: 1,
        27: 1,
        26: 1
    }
    
    for n in range(40, 25, -1):
        t_start = time.time()
        t_limit = min(deadline, t_start + time_alloc.get(n, 1))
        
        # Try a few random restarts
        for attempt in range(100):
            if time.time() > t_limit:
                break
                
            # Initialize random graph
            edges = set()
            # Probability 0.3 is a guess. 
            # For triangle-free, max edges is n^2/4 = 400. 
            # For n=40, 0.3 * 780 = 234 edges. This is close to max.
            # Maybe 0.2 is safer.
            p = 0.2
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        edges.add((i, j))
            
            # Local search
            for step in range(5000):
                if time.time() > t_limit:
                    break
                    
                tri = _has_triangle(edges, n)
                if tri:
                    u, v, w = tri
                    e_to_remove = random.choice([(u, v), (u, w), (v, w)])
                    e_norm = (min(e_to_remove), max(e_to_remove))
                    if e_norm in edges:
                        edges.remove(e_norm)
                    continue
                    
                indep = _has_independent_set_10(edges, n)
                if indep:
                    u, v = random.sample(indep, 2)
                    e_new = (min(u, v), max(u, v))
                    if e_new not in edges:
                        edges.add(e_new)
                    continue
                
                # Valid graph found
                edge_list = [[u, v] for u, v in edges]
                return {'n': n, 'edges': edge_list}
                
    # Fallback
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
