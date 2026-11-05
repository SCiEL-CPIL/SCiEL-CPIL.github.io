import random
import time

def construct():
    n = 16
    
    def is_3_colorable(adj):
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            used = 0
            mask = adj[v]
            for u in range(n):
                if mask & (1 << u):
                    if colors[u] != -1:
                        used |= (1 << colors[u])
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
            colors[v] = -1
            return False
        
        return backtrack(0)

    def get_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # Grötzsch graph initialization
    initial_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],
        [5, 6], [6, 7], [7, 8], [8, 9], [9, 5],
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
    ]
    
    current_edges = set((u, v) for u, v in initial_edges)
    current_adj = get_adj(current_edges)
    
    # Check initial state
    # Grötzsch is 4-chromatic, so not 3-colorable
    is_non_3_colorable = not is_3_colorable(current_adj)
    
    best_edges = list(current_edges)
    best_count = len(current_edges)
    
    start_time = time.time()
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    rng = random.Random(42)
    
    # Score function for SA acceptance
    # We maximize edge count, but strongly prefer non-3-colorable
    # Delta calculation is easier with explicit states
    
    while time.time() - start_time < 115:
        # Move generation
        if rng.random() < 0.8:
            # Add edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            # Normalize
            if u > v:
                u, v = v, u
            if (u, v) in current_edges:
                continue
            
            # Triangle check
            if current_adj[u] & current_adj[v]:
                continue
            
            # Apply move
            current_edges.add((u, v))
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
            # Evaluate
            if is_non_3_colorable:
                # Stays non-3-colorable
                new_is_non_3_colorable = True
                # Score change: +1 edge
                delta_score = 1
                # Always accept? 
                # In SA, we might reject to explore, but this is strictly better.
                # However, to escape local optima, we might want to remove edges later.
                # Accepting always leads to maximal graph.
                # We rely on removal moves to escape.
                accept = True
            else:
                # Was 3-colorable, might be non-3-colorable now
                new_is_non_3_colorable = not is_3_colorable(current_adj)
                if new_is_non_3_colorable:
                    # Huge improvement
                    delta_score = 1000 
                    accept = True
                else:
                    # Still 3-colorable, +1 edge
                    delta_score = 1
                    accept = rng.random() < pow(2.71828, delta_score / T)
            
            if accept:
                is_non_3_colorable = new_is_non_3_colorable
                if is_non_3_colorable and len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
            else:
                # Revert
                current_edges.remove((u, v))
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
        else:
            # Remove edge
            if not current_edges:
                continue
            e = rng.choice(list(current_edges))
            u, v = e
            current_edges.remove(e)
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
            # Evaluate
            if is_non_3_colorable:
                # Might become 3-colorable
                new_is_non_3_colorable = not is_3_colorable(current_adj)
                if new_is_non_3_colorable:
                    # Lost 1 edge, still non-3-colorable
                    delta_score = -1
                    accept = rng.random() < pow(2.71828, delta_score / T)
                else:
                    # Became 3-colorable
                    delta_score = -1001
                    accept = False # Very unlikely
            else:
                # Was 3-colorable, stays 3-colorable
                new_is_non_3_colorable = False
                delta_score = -1
                accept = rng.random() < pow(2.71828, delta_score / T)
            
            if accept:
                is_non_3_colorable = new_is_non_3_colorable
                # No best update on removal
            else:
                # Revert
                current_edges.add(e)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
        
        T *= cooling
        if T < T_min:
            T = T_min

    return [[u, v] for u, v in best_edges]
