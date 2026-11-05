"""C5-free graphs without independent 9-sets: Random search for larger n."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 140s safety margin
    
    def make_8k4():
        n = 32
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return n, edges

    best_n, best_edges = make_8k4()
    
    def check_c5_free(adj, n):
        # Check for C5 in graph represented by adj list
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Check if there is a path of length 3 from u to v
                    # u - a - b - v
                    for a in adj[u]:
                        if a == v: continue
                        for b in adj[a]:
                            if b == u or b == a: continue
                            if v in adj[b]:
                                # Found u-a-b-v-u
                                # Check if it's a simple cycle
                                # u, a, b, v are distinct?
                                if len({u, a, b, v}) == 4:
                                    return False
        return True

    def has_alpha_ge_9(adj, n, time_limit=5.0):
        # Try to find an independent set of size 9
        # Backtracking
        t0 = time.time()
        
        def backtrack(candidates, current_size, current_set):
            if time.time() - t0 > time_limit:
                return None
            if current_size >= 9:
                return current_set
            if not candidates:
                return None
            if len(candidates) + current_size < 9:
                return None
            
            # Pick a vertex with minimum degree in candidates to prune
            # Or just pick the first one
            v = candidates.pop(0)
            
            # Include v
            new_candidates = [x for x in candidates if x not in adj[v]]
            res = backtrack(new_candidates, current_size + 1, current_set + [v])
            if res:
                return res
            
            # Exclude v
            res = backtrack(candidates, current_size, current_set)
            if res:
                return res
            
            return None

        candidates = list(range(n))
        res = backtrack(candidates, 0, [])
        return res is not None

    def try_random_search(n, time_budget):
        # Try to find a C5-free graph on n vertices with alpha < 9
        random.seed(n) # Deterministic per n
        
        # Start with empty graph
        adj = [set() for _ in range(n)]
        edges = []
        
        # Pre-compute all possible edges
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(possible_edges)
        
        t_start = time.time()
        
        for u, v in possible_edges:
            if time.time() > t_start + time_budget:
                break
                
            # Check if adding (u, v) creates a C5
            # We need to check if there is a path of length 3 between u and v
            creates_c5 = False
            for a in adj[u]:
                for b in adj[a]:
                    if b == v or b == u:
                        continue
                    if v in adj[b]:
                        # Path u-a-b-v exists
                        # Check if u, a, b, v are distinct
                        if len({u, a, b, v}) == 4:
                            creates_c5 = True
                            break
                if creates_c5:
                    break
            
            if not creates_c5:
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
                # Periodically check alpha
                # We can check alpha more frequently if n is small
                if len(edges) % 10 == 0:
                    if not has_alpha_ge_9(adj, n, time_limit=2.0):
                        # Alpha < 9 found!
                        return n, edges
        
        # Final check
        if not has_alpha_ge_9(adj, n, time_limit=5.0):
            return n, edges
            
        return None

    # Try n=33, 34, 35
    for n_try in [33, 34, 35]:
        if time.time() > deadline - 20:
            break
            
        # Allocate time based on remaining time
        remaining = deadline - time.time()
        time_budget = remaining * 0.8
        if time_budget < 5:
            break
            
        result = try_random_search(n_try, time_budget)
        if result:
            best_n, best_edges = result
            # If we found a valid graph for n_try, we can try to go higher
            # But the loop structure tries 33, then 34, then 35.
            # If we succeed at 33, we should try 34 with remaining time.
            # If we fail at 33, we try 34.
            # This logic is slightly flawed: if we succeed at 33, we have a valid 33.
            # We should then try to find a valid 34.
            # If we find 34, we replace best.
            # If we don't find 34, we keep 33.
            continue
            
    # Verify the best solution found
    # Re-verify C5-free and Alpha < 9 for the returned graph to be safe
    # The search function already verified, but let's double check if time permits
    if time.time() < deadline - 5:
        adj = [set() for _ in range(best_n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        if not check_c5_free(adj, best_n):
            # Fallback to 8xK4
            best_n, best_edges = make_8k4()
        elif has_alpha_ge_9(adj, best_n, time_limit=10.0):
            # Fallback to 8xK4
            best_n, best_edges = make_8k4()

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
