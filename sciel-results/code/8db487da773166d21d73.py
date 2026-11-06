"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

def construct():
    """
    Find a maximum cardinality subset of [1, 256] such that a+b is never a perfect square
    for any a <= b in the set (including a=b).
    
    We model this as a maximum independent set problem on a conflict graph where:
    - Vertices: numbers 1..256
    - Edge between u and v if u+v is a perfect square (including u==v, i.e., 2u is a square)
    
    We use branch-and-bound with greedy coloring upper bounds.
    """
    N = 256
    
    # Precompute perfect squares up to 2*256 = 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Build conflict graph: for each vertex i (0-indexed, representing i+1),
    # find all j >= i such that (i+1) + (j+1) is a perfect square
    # Note: we need to include self-loops (a=b case): 2*(i+1) in squares means i conflicts with itself
    
    # Actually, let's think about this more carefully.
    # The condition is: for all a, b in S with a <= b, a+b is NOT a perfect square.
    # This includes a=b, so 2a must not be a perfect square for any a in S.
    
    # So vertex i (representing number i+1) has a self-conflict if 2*(i+1) is a perfect square.
    # If a vertex has a self-conflict, it cannot be in any valid set.
    
    # Let's identify which numbers can even be in the set:
    # Number x (1..256) is "self-valid" if 2*x is NOT a perfect square.
    
    self_valid = []
    for x in range(1, N + 1):
        if 2 * x not in squares:
            self_valid.append(x)
    
    # Now build the conflict graph among self-valid numbers.
    # Two numbers a, b (a != b) conflict if a+b is a perfect square.
    # We'll work with indices into the self_valid list.
    
    m = len(self_valid)
    # Map from number to index in self_valid
    num_to_idx = {x: i for i, x in enumerate(self_valid)}
    
    # Build adjacency list
    adj = [[] for _ in range(m)]
    for i in range(m):
        a = self_valid[i]
        for j in range(i + 1, m):
            b = self_valid[j]
            if a + b in squares:
                adj[i].append(j)
                adj[j].append(i)
    
    # Now solve maximum independent set on this graph using branch and bound.
    # We'll use a recursive approach with coloring-based upper bounds.
    
    best_set = []
    best_size = 0
    
    def color_bound(vertices):
        """
        Compute a greedy coloring upper bound on the maximum independent set size
        for the subgraph induced by 'vertices'.
        Returns a list of colors (one per vertex) and the number of colors used.
        The number of colors is an upper bound on the independence number.
        """
        if not vertices:
            return [], 0
        
        # Order vertices by degree in the subgraph (descending) for better coloring
        deg = {}
        for v in vertices:
            deg[v] = sum(1 for u in adj[v] if u in vertices)
        ordered = sorted(vertices, key=lambda v: -deg[v])
        
        colors = {}
        color_num = 0
        for v in ordered:
            # Find the smallest color not used by neighbors
            neighbor_colors = set()
            for u in adj[v]:
                if u in vertices and u in colors:
                    neighbor_colors.add(colors[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            color_num = max(color_num, c + 1)
        
        return colors, color_num
    
    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_set, best_size
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Upper bound: current_size + len(candidates)
        if current_size + len(candidates) <= best_size:
            return
        
        # Get coloring bound
        colors, num_colors = color_bound(candidates)
        if current_size + num_colors <= best_size:
            return
        
        # Choose a vertex to branch on: pick one with highest degree in candidates
        cand_list = list(candidates)
        deg = {}
        for v in cand_list:
            deg[v] = sum(1 for u in adj[v] if u in cand_list)
        
        # Pick vertex with max degree
        v = max(cand_list, key=lambda x: deg[x])
        
        # Branch 1: exclude v
        candidates_excl = [x for x in cand_list if x != v]
        branch_and_bound(candidates_excl, current_set, current_size)
        
        # Branch 2: include v
        # Remove v and all its neighbors from candidates
        neighbors = set(adj[v])
        candidates_incl = [x for x in cand_list if x != v and x not in neighbors]
        current_set.append(v)
        branch_and_bound(candidates_incl, current_set, current_size + 1)
        current_set.pop()
    
    # Start with all self-valid vertices
    all_vertices = list(range(m))
    branch_and_bound(all_vertices, [], 0)
    
    # Convert back to original numbers
    result = [self_valid[i] for i in best_set]
    
    # Ensure nonempty
    if not result:
        # Fallback: find any single valid number
        for x in range(1, N + 1):
            if 2 * x not in squares:
                return [x]
        return [1]
    
    return result

# EVOLVE-BLOCK-END
