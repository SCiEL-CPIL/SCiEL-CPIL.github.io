import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency matrix for O(1) edge checks
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
        
    # Strategy: 
    # 1. Generate a pool of candidate maximal bicliques.
    # 2. Use a greedy set cover approach to select the minimum number of bicliques 
    #    that cover all edges exactly once (partition).
    #    Note: Standard set cover allows overlap. Here we need an EXACT partition.
    #    However, if we pick a biclique, we remove its edges. The remaining edges 
    #    must be partitioned by other bicliques from the pool (or new ones).
    #    A simpler robust approach for exact partition:
    #    Greedy: Pick the largest available biclique (from a generated set of maximal bicliques 
    #    restricted to remaining edges), remove its edges, repeat.
    #    To ensure optimality, we generate many candidates and use randomized restarts 
    #    with different tie-breaking orders.

    def expand_biclique(start_u, start_v, remaining_edges):
        """
        Expand a biclique starting with edge (start_u, start_v) using a greedy 
        maximal expansion strategy on the subgraph induced by remaining_edges.
        Returns (L, R, edges_covered)
        """
        if (min(start_u, start_v), max(start_u, start_v)) not in remaining_edges:
            return None, None, set()
            
        L = {start_u}
        R = {start_v}
        
        # We want to maximize the size of the biclique (number of edges)
        # Greedy expansion: try to add vertices that connect to all current opposite side vertices
        # and have edges in remaining_edges.
        
        changed = True
        while changed:
            changed = False
            # Try adding to L
            # Candidates for L: vertices x not in L such that x is connected to all in R
            # and all edges (x, r) for r in R are in remaining_edges
            candidates_L = []
            for x in range(n):
                if x in L: continue
                # Check connectivity to R
                ok = True
                for y in R:
                    if not adj[x][y]:
                        ok = False
                        break
                    e = (min(x, y), max(x, y))
                    if e not in remaining_edges:
                        ok = False
                        break
                if ok:
                    candidates_L.append(x)
            
            # Add all valid candidates to L? Or just one? 
            # Adding all might prevent future expansion? 
            # Actually, if x is connected to all R, adding x to L is always safe 
            # for the biclique property. It might block other vertices from joining R 
            # if they aren't connected to x. 
            # To get a MAXIMAL biclique, we should be careful. 
            # Standard maximal biclique expansion often adds one at a time or uses specific heuristics.
            # Let's try adding all possible L candidates first, then R.
            if candidates_L:
                for x in candidates_L:
                    L.add(x)
                changed = True
                
            # Try adding to R
            candidates_R = []
            for y in range(n):
                if y in R: continue
                ok = True
                for x in L:
                    if not adj[x][y]:
                        ok = False
                        break
                    e = (min(x, y), max(x, y))
                    if e not in remaining_edges:
                        ok = False
                        break
                if ok:
                    candidates_R.append(y)
            
            if candidates_R:
                for y in candidates_R:
                    R.add(y)
                changed = True
        
        # Calculate covered edges
        covered = set()
        for u in L:
            for v in R:
                covered.add((min(u, v), max(u, v)))
        
        # Verify all covered edges are actually in remaining_edges
        # (The expansion logic ensures this, but double check)
        for e in covered:
            if e not in remaining_edges:
                return None, None, set()
                
        return list(L), list(R), covered

    def solve_greedy(seed):
        """
        Greedy exact cover using a pool of bicliques generated on the fly 
        with randomization in expansion order.
        """
        random.seed(seed)
        remaining = set(edge_set)
        solution = []
        
        while remaining:
            # Pick a random edge from remaining to start a new biclique
            # This randomization helps escape local minima of "always pick the same big biclique"
            start_edge = random.choice(list(remaining))
            
            # Generate multiple candidate bicliques for this start edge with different expansion strategies?
            # Or just one? 
            # To be effective, we should probably generate a few candidates and pick the best one 
            # (largest coverage) or just use the greedy expansion.
            
            # Let's try a few expansions with different random perturbations?
            # Simplest: Just do the greedy expansion. 
            # But to improve, we can try to bias the expansion.
            
            L, R, covered = expand_biclique(start_edge[0], start_edge[1], remaining)
            
            if not covered:
                # Fallback: single edge biclique
                u, v = start_edge
                L = [u]
                R = [v]
                covered = {start_edge}
            
            solution.append([L, R])
            remaining -= covered
            
        return solution

    # We will run multiple randomized greedy attempts and keep the best (fewest bicliques)
    best_solution = None
    best_count = float('inf')
    
    # How many restarts? 
    # Each restart is fast (O(E * V^2) approx). 
    # Let's do as many as we can in the time budget.
    
    attempts = 0
    while time.time() < deadline - 2:  # Leave 2s for final check
        attempts += 1
        sol = solve_greedy(attempts)
        if len(sol) < best_count:
            best_count = len(sol)
            best_solution = sol
            
        # Early stop if we find a very good solution? 
        # Lower bound: max degree? No, biclique cover number is hard.
        # If best_count == 1, we are done.
        if best_count == 1:
            break
            
        # If we are getting stuck at a high number, maybe increase randomization?
        # The current randomization is just picking the start edge.
        # We could also randomize the order of adding vertices in expansion.
        # Let's add a bit more randomness to the expansion process in a new function?
        # No, keep it simple. The start edge randomization is the main driver.
        
    # Final validation
    if best_solution is None:
        return []
        
    covered_edges = set()
    for L, R in best_solution:
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in covered_edges:
                    return [] # Overlap
                covered_edges.add(e)
                if not adj[u][v]:
                    return [] # Not an edge
    
    if covered_edges != edge_set:
        return [] # Missing edges
        
    final_result = []
    for L, R in best_solution:
        final_L = [int(x) for x in L]
        final_R = [int(x) for x in R]
        final_result.append([final_L, final_R])
        
    return final_result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")
