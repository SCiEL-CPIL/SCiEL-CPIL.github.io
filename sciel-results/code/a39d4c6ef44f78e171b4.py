import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Helper functions
    
    def has_k4(adj, n):
        """Check if graph contains a K4 using bitmask operations."""
        masks = []
        for u in range(n):
            mask = 0
            for v in adj[u]:
                mask |= (1 << v)
            masks.append(mask)
        
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # Triangle u-v-w
                common_uv = adj[u] & adj[v]
                for w in common_uv:
                    if w <= v:
                        continue
                    # Check if any common neighbor of u and w is also adjacent to v
                    common_uw = adj[u] & adj[w]
                    if common_uw & masks[v]:
                        return True
        return False
    
    def has_independent_set(adj, n, k):
        """Check if graph contains an independent set of size k using backtracking with bitmasks."""
        if n < k:
            return False
        
        # Build non-neighbor masks (independent set candidates)
        non_neighbors = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if v != u and v not in adj[u]:
                    mask |= (1 << v)
            non_neighbors[u] = mask
        
        def backtrack(start, current_mask, size):
            if size == k:
                return True
            if start >= n:
                return False
            if (n - start) < (k - size):
                return False
            
            # Find candidates from start that are not in current_mask
            candidates = []
            for v in range(start, n):
                if not (current_mask & (1 << v)):
                    candidates.append(v)
            
            if len(candidates) < (k - size):
                return False
            
            for v in candidates:
                if backtrack(v + 1, current_mask | non_neighbors[v], size + 1):
                    return True
            return False
        
        return backtrack(0, 0, 0)
    
    def get_independent_sets_size_k(adj, n, k, max_sets=1000):
        """Return list of independent sets of size k."""
        if n < k:
            return []
        
        non_neighbors = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if v != u and v not in adj[u]:
                    mask |= (1 << v)
            non_neighbors[u] = mask
        
        sets = []
        
        def backtrack(start, current_set, forbidden_mask):
            if len(current_set) == k:
                sets.append(tuple(current_set))
                return
            if len(sets) >= max_sets:
                return
            remaining = k - len(current_set)
            if n - start < remaining:
                return
            
            for v in range(start, n):
                if not (forbidden_mask & (1 << v)):
                    current_set.append(v)
                    backtrack(v + 1, current_set, forbidden_mask | non_neighbors[v])
                    current_set.pop()
                    if len(sets) >= max_sets:
                        return
        
        backtrack(0, [], 0)
        return sets
    
    def try_add_vertex_fast(adj, n, deadline, ind_sets_5):
        """
        Find a valid neighbor set N for a new vertex.
        Constraints:
        1. N must be triangle-free (no K4 in G + new vertex)
        2. N must intersect every independent set of size 5 (no independent set of size 6)
        """
        if time.time() > deadline:
            return None
        
        if not ind_sets_5:
            # No independent sets of size 5, just need triangle-free N
            # Greedy: add vertices that don't form triangles
            N = set()
            remaining = list(range(n))
            random.shuffle(remaining)
            for v in remaining:
                valid = True
                neigh_in_N = [u for u in N if u in adj[v]]
                for i in range(len(neigh_in_N)):
                    for j in range(i + 1, len(neigh_in_N)):
                        if neigh_in_N[j] in adj[neigh_in_N[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    N.add(v)
            return N
        
        # Use greedy set cover with randomization
        best_N = None
        best_size = -1
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            N = set()
            active_sets = [set(s) for s in ind_sets_5]
            success = True
            
            while active_sets:
                if time.time() > deadline:
                    success = False
                    break
                
                # Pick the active set with fewest candidates
                best_idx = -1
                best_cands = None
                min_cands = float('inf')
                
                indices = list(range(len(active_sets)))
                random.shuffle(indices)
                
                for idx in indices:
                    target = active_sets[idx]
                    cands = []
                    for v in target:
                        # Check if v can be added to N without forming triangle
                        valid = True
                        neigh_v = [u for u in N if u in adj[v]]
                        for i in range(len(neigh_v)):
                            for j in range(i + 1, len(neigh_v)):
                                if neigh_v[j] in adj[neigh_v[i]]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            cands.append(v)
                    
                    if len(cands) < min_cands:
                        min_cands = len(cands)
                        best_idx = idx
                        best_cands = cands
                    
                    if min_cands == 0:
                        break
                
                if min_cands == 0:
                    success = False
                    break
                
                random.shuffle(best_cands)
                
                # Try each candidate
                found = False
                for v in best_cands:
                    new_N = N | {v}
                    new_active = []
                    for s in active_sets:
                        if v not in s:
                            new_active.append(s)
                    
                    # Check if this leads to a valid completion (quick heuristic)
                    # Just proceed with backtracking
                    N = new_N
                    active_sets = new_active
                    found = True
                    break
                
                if not found:
                    success = False
                    break
            
            if success and len(N) > best_size:
                best_size = len(N)
                best_N = N.copy()
            
            if best_size >= n - 1:
                break
        
        return best_N
    
    def local_search_polish(adj, n, deadline):
        """
        Try edge swaps to enable further vertex additions.
        """
        improved = False
        ind_sets_5 = get_independent_sets_size_k(adj, n, 5)
        
        for iteration in range(30):
            if time.time() > deadline:
                break
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            is_edge = v in adj[u]
            
            if is_edge:
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
            
            if has_k4(adj, n) or has_independent_set(adj, n, 6):
                if is_edge:
                    adj[u].add(v)
                    adj[v].add(u)
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
            else:
                ind_sets_5_new = get_independent_sets_size_k(adj, n, 5)
                new_N = try_add_vertex_fast(adj, n, deadline, ind_sets_5_new)
                if new_N is not None:
                    adj.append(set(new_N))
                    for u2 in new_N:
                        adj[u2].add(n)
                    n += 1
                    improved = True
                    ind_sets_5 = ind_sets_5_new
                else:
                    if is_edge:
                        adj[u].add(v)
                        adj[v].add(u)
                    else:
                        adj[u].remove(v)
                        adj[v].remove(u)
        
        return improved, n
    
    # Start with Paley 17
    q = 17
    squares = set()
    for i in range(1, q):
        squares.add((i * i) % q)
    
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    
    adj = [set() for _ in range(q)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    n = q
    
    # Verify initial graph
    if has_k4(adj, n) or has_independent_set(adj, n, 6):
        return {'n': 1, 'edges': []}
    
    # Main loop
    while n < 35 and time.time() < deadline:
        if time.time() > deadline - 5:
            break
        
        ind_sets_5 = get_independent_sets_size_k(adj, n, 5)
        new_N = try_add_vertex_fast(adj, n, deadline, ind_sets_5)
        
        if new_N is not None:
            adj.append(set(new_N))
            for u in new_N:
                adj[u].add(n)
            n += 1
        else:
            if time.time() < deadline - 10:
                improved, n = local_search_polish(adj, n, deadline)
                if improved:
                    continue
            break
    
    # Build edge list
    final_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                final_edges.append([u, v])
    
    return {'n': n, 'edges': final_edges}
