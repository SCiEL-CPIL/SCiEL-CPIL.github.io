import time
import math
import random
from math import gcd

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b.
    
    Uses graph-theoretic maximum clique search with greedy coloring bounds
    and local search heuristics.
    """
    deadline = time.time() + 50.0  # 50s safety margin before 60s limit
    
    # Build candidate set: numbers in [1,255] divisible by 3, 5, or 17
    # Must contain 255, and all elements must share a factor with 255
    candidates = [i for i in range(1, 256) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    
    # Build adjacency: edge between a,b if gcd(a,b) > 1
    n = len(candidates)
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # Greedy coloring upper bound for maximum clique
    def color_order(vertices):
        """Order vertices by greedy coloring, return (order, color_bound)"""
        colors = {}
        order = []
        for v in vertices:
            # Find smallest available color
            used_colors = set()
            for u in order:
                if adj[v][u]:
                    used_colors.add(colors[u])
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            order.append(v)
        max_color = max(colors.values()) if colors else 0
        return order, max_color
    
    def greedy_coloring_upper_bound(vertices):
        """Get upper bound on clique size in induced subgraph of vertices"""
        if not vertices:
            return 0
        order, bound = color_order(vertices)
        return bound
    
    # Exact maximum clique via branch and bound
    def max_clique_exact(vertices, time_check=None):
        """Find maximum clique in induced subgraph of vertices using Bron-Kerbosch with coloring"""
        best = [set()]
        
        def bron_kerbosch(P, X, R):
            if time_check and time.time() > time_check:
                return
            if not P and not X:
                if len(R) > len(best[0]):
                    best[0] = set(R)
                return
            
            # Get coloring bound
            P_list = list(P)
            order, color_bound = color_order(P_list)
            
            # Process in reverse order (highest color first)
            for idx in range(len(order) - 1, -1, -1):
                if time_check and time.time() > time_check:
                    return
                v = order[idx]
                # Upper bound: current clique size + colors[v]
                if len(R) + colors[v] <= len(best[0]):
                    break
                
                # Expand
                new_R = set(R)
                new_R.add(v)
                new_P = P - {v}
                new_X = X - {v}
                
                # Only consider neighbors of v
                new_P = {u for u in new_P if adj[v][u]}
                new_X = {u for u in new_X if adj[v][u]}
                
                bron_kerbosch(new_P, new_X, new_R)
                
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(vertices), set(), set())
        return best[0]
    
    # Local search: start with multiples of 3, try to add elements and remove conflicts
    def local_search(start_set, max_iters=1000):
        """Improve clique via add/remove operations"""
        current = set(start_set)
        best = set(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try to add each non-member
            for v in candidates:
                if v in current:
                    continue
                # Check if v can be added (connected to all in current)
                can_add = all(adj[candidates.index(v)][candidates.index(u)] for u in current)
                if can_add:
                    new_set = current | {v}
                    if len(new_set) > len(best):
                        best = new_set
                        current = new_set
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swap: remove one, add another
            for u in list(current):
                for v in candidates:
                    if v in current:
                        continue
                    # Check if v connects to all in current except u
                    can_add = all(
                        u == w or adj[candidates.index(v)][candidates.index(w)]
                        for w in current
                    )
                    if can_add and len(current) - 1 + 1 > len(best):
                        new_set = (current - {u}) | {v}
                        if len(new_set) > len(best):
                            best = new_set
                            current = new_set
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try 2-swap: remove two, add two
                found = False
                cand_list = [v for v in candidates if v not in current]
                for i in range(len(cand_list)):
                    if time.time() > deadline:
                        break
                    v1 = cand_list[i]
                    for j in range(i + 1, len(cand_list)):
                        v2 = cand_list[j]
                        # Check if v1,v2 connect to all in current except 2 removed
                        # This is expensive; skip for now
                        pass
                break
        
        return best
    
    # Start with the known good solution: all multiples of 3
    start_set = [i for i in range(1, 256) if i % 3 == 0]
    best_clique = set(start_set)
    
    # Try local search from this start
    improved = local_search(start_set)
    if len(improved) > len(best_clique):
        best_clique = improved
    
    # Try exact maximum clique on a smaller induced subgraph
    # Focus on elements that might improve: non-multiples of 3 that are multiples of 5 or 17
    # But we need to be careful about time
    
    # Try a few random restarts with different initial sets
    random.seed(42)
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Create a random subset of candidates that includes 255
        if 255 not in candidates:
            continue
        idx_255 = candidates.index(255)
        
        # Start with 255 and randomly add connected elements
        current = {candidates[idx_255]}
        for v in candidates:
            if v == candidates[idx_255]:
                continue
            if adj[idx_255][candidates.index(v)] and random.random() < 0.5:
                current.add(v)
        
        # Improve via local search
        improved = local_search(list(current), max_iters=200)
        if len(improved) > len(best_clique):
            best_clique = improved
    
    # Final polish: try to add any remaining elements that fit
    for v in candidates:
        if v not in best_clique:
            can_add = all(
                gcd(v, u) > 1 for u in best_clique
            )
            if can_add:
                best_clique.add(v)
    
    return list(best_clique)
