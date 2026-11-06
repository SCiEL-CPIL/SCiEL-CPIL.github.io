import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    N = 256
    # 1. Identify vertices with self-conflicts (2v is a perfect square)
    invalid_vertices = set()
    for v in range(1, N + 1):
        s = 2 * v
        root = int(math.isqrt(s))
        if root * root == s:
            invalid_vertices.add(v)
            
    valid_vertices = [v for v in range(1, N + 1) if v not in invalid_vertices]
    # Map valid vertices to indices 0..M-1 for faster array access
    M = len(valid_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 2. Build conflict graph
    # adj[i] is a set of indices j such that valid_vertices[i] + valid_vertices[j] is a square
    adj = [set() for _ in range(M)]
    
    squares = set()
    max_sum = 2 * N
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
        
    for i in range(M):
        vi = valid_vertices[i]
        for j in range(i + 1, M):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Greedy heuristic for initial lower bound
    # Sort by degree descending
    degrees = [len(adj[i]) for i in range(M)]
    order = sorted(range(M), key=lambda x: -degrees[x])
    
    best_set = set()
    available = set(range(M))
    
    # Simple greedy: pick highest degree, remove neighbors
    # This might not be optimal but gives a baseline
    temp_available = set(range(M))
    greedy_set = set()
    while temp_available:
        # Find vertex in temp_available with max degree within temp_available
        best_v = None
        max_deg = -1
        for v in temp_available:
            deg = len(adj[v] & temp_available)
            if deg > max_deg:
                max_deg = deg
                best_v = v
        if best_v is None:
            break
        greedy_set.add(best_v)
        # Remove best_v and its neighbors from temp_available
        to_remove = {best_v} | (adj[best_v] & temp_available)
        temp_available -= to_remove
        
    current_best_size = len(greedy_set)
    best_solution = greedy_set.copy()
    
    # 4. Branch and Bound
    # State: current_set (set of indices), available (set of indices)
    # We use a list-based representation for speed if possible, but sets are easier for correctness first.
    # Given M=245, recursion depth can be up to 245.
    
    def bound(available_set):
        # Simple bound: size of available set
        # Better bound: Greedy coloring of the conflict graph restricted to available_set?
        # Actually, for Independent Set, we want an upper bound on the size of an independent set in available_set.
        # The size of a minimum vertex cover is an upper bound? No.
        # A simple upper bound is the size of the largest clique in the complement? No.
        # Let's use a simple greedy coloring on the *complement*? No, that's for clique.
        # For Independent Set, a good upper bound is the number of vertices in a minimum clique cover of the available subgraph?
        # Or simply: if we greedily pick vertices and remove neighbors, the number of picks is a *lower* bound on the independent set size? No, it's a heuristic solution.
        # Upper bound: The sum of degrees? No.
        # Let's stick to |available_set| as a weak bound, but improve vertex selection.
        # A stronger bound: Color the graph G[available]. The chromatic number is an upper bound on the clique number, not independent set.
        # Wait, the Independent Set in G is a Clique in G_complement.
        # So we can use a greedy coloring bound on G_complement to get an upper bound on the clique size in G_complement (which is IS in G).
        # Greedy coloring gives an upper bound on the chromatic number, which is an upper bound on the clique number? No, chromatic number >= clique number.
        # So a greedy coloring of G_complement gives an upper bound on the size of the independent set in G.
        
        if not available_set:
            return 0
            
        # Greedy coloring of the complement graph restricted to available_set
        # Colors assigned to vertices in available_set
        colors = {}
        uncolored = set(available_set)
        color_count = 0
        
        while uncolored:
            color_count += 1
            # Find a maximal independent set in G_complement restricted to uncolored
            # Which is a maximal clique in G restricted to uncolored? No.
            # Independent set in G_complement = set of vertices that are NOT connected in G_complement
            # Not connected in G_complement means connected in G? No.
            # Edge in G_complement (u,v) iff NO edge in G (u,v).
            # So Independent Set in G_complement = set of vertices where every pair HAS an edge in G.
            # That is a Clique in G.
            # So we are looking for a maximal clique in G[uncolored].
            # Greedy clique finding:
            clique = set()
            candidates = set(uncolored)
            while candidates:
                # Pick vertex with max degree in G restricted to candidates
                best_v = None
                max_deg = -1
                for v in candidates:
                    deg = len(adj[v] & candidates)
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                if best_v is None:
                    break
                clique.add(best_v)
                # Remove best_v and its non-neighbors in G?
                # In a clique, all must be connected.
                # So we keep only neighbors of best_v in G.
                candidates = candidates & adj[best_v]
                candidates.discard(best_v)
                
            # This clique is an independent set in G_complement.
            # It gets color 'color_count'.
            uncolored -= clique
            
        return color_count

    # Recursion
    def search(current_set, available_set):
        nonlocal current_best_size, best_solution, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if not available_set:
            if len(current_set) > current_best_size:
                current_best_size = len(current_set)
                best_solution = current_set.copy()
            return
            
        # Pruning
        if len(current_set) + len(available_set) <= current_best_size:
            return
            
        # Stronger pruning using coloring bound
        # This is expensive, so maybe use it only when available_set is small or periodically
        # For now, let's rely on the simple bound and good vertex ordering.
        
        # Select vertex to branch on
        # Heuristic: Choose vertex with maximum degree in available_set
        # This helps prune the "include" branch
        best_v = None
        max_deg = -1
        for v in available_set:
            deg = len(adj[v] & available_set)
            if deg > max_deg:
                max_deg = deg
                best_v = v
                
        if best_v is None:
            return
            
        # Branch 1: Include best_v
        # New available: available_set minus best_v and its neighbors
        neighbors = adj[best_v] & available_set
        new_available_include = available_set - {best_v} - neighbors
        new_current_set = current_set | {best_v}
        
        # Branch 2: Exclude best_v
        new_available_exclude = available_set - {best_v}
        
        # Order: Try Include first if it looks promising?
        # Usually, including high-degree nodes leads to smaller subproblems faster.
        search(new_current_set, new_available_include)
        
        search(current_set, new_available_exclude)
        
    search(set(), set(range(M)))
    
    # Map indices back to values
    final_values = [valid_vertices[i] for i in best_solution]
    
    return final_values

# EVOLVE-BLOCK-END
