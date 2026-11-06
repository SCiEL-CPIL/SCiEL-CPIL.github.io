import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    timeout = 55.0
    
    n = 256
    adj = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
                
    best_solution = []
    best_size = 0
    
    def greedy_mis(adj, n):
        """Fallback greedy algorithm"""
        remaining = set(range(1, n + 1))
        result = []
        while remaining:
            # Pick vertex with min degree in remaining subgraph?
            # Or just pick any? Min degree is better.
            min_v = None
            min_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            result.append(min_v)
            remaining.discard(min_v)
            remaining -= adj[min_v]
        return result

    def greedy_color(P):
        """
        P is a list of vertices.
        Returns max color used.
        """
        if not P:
            return 0
        colors = {}
        max_c = 0
        # Order P? The order affects the coloring quality (tightness of bound).
        # Sorting by degree descending often yields tighter bounds.
        # But sorting costs O(N log N).
        # Let's just use the given order or sort by degree.
        # Given P is already sorted by degree in expand, we can assume it's helpful.
        
        for v in P:
            used = set()
            for u in adj[v]:
                if u in colors:
                    used.add(colors[u])
            
            c = 1
            while c in used:
                c += 1
            
            colors[v] = c
            if c > max_c:
                max_c = c
                
        return max_c

    def expand(P, sol):
        nonlocal best_solution, best_size
        
        if time.time() - start_time > timeout:
            return
            
        if not P:
            if len(sol) > best_size:
                best_size = len(sol)
                best_solution = sol.copy()
            return
            
        # Compute UB
        ub = greedy_color(P)
        
        if len(sol) + ub <= best_size:
            return
            
        # Order P by degree descending to find good solutions early
        P = list(P)
        P.sort(key=lambda x: -len(adj[x]))
        
        for i in range(len(P)):
            v = P[i]
            
            # Include v
            new_P = [u for u in P if u != v and u not in adj[v]]
            
            # Recurse
            expand(new_P, sol + [v])
            
            # Exclude v
            P.pop(i)
            i -= 1
            
            # Check if remaining P can beat best
            # We need a bound for the remaining P.
            # Re-coloring is expensive.
            # Heuristic: If the max color of the *original* P minus 1 is <= best - len(sol) - 1, break?
            # No, colors change.
            # Let's just re-color. It's the safest.
            if len(sol) + greedy_color(P) <= best_size:
                break
                
    initial_P = list(range(1, n + 1))
    expand(initial_P, [])
    
    if not best_solution:
        best_solution = greedy_mis(adj, n)

    return list(best_solution)
