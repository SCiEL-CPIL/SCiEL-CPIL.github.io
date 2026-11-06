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
    
    We use branch-and-bound with bitset-based operations and greedy coloring upper bounds.
    """
    N = 256
    
    # Precompute perfect squares up to 2*256 = 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Identify which numbers can even be in the set:
    # Number x (1..256) is "self-valid" if 2*x is NOT a perfect square.
    
    self_valid = []
    for x in range(1, N + 1):
        if 2 * x not in squares:
            self_valid.append(x)
    
    m = len(self_valid)
    
    # Build adjacency bitsets: adj[i] is a bitmask where bit j is set if i and j conflict
    adj = [0] * m
    for i in range(m):
        a = self_valid[i]
        mask = 0
        for j in range(m):
            if i == j:
                continue
            b = self_valid[j]
            if a + b in squares:
                mask |= (1 << j)
        adj[i] = mask
    
    # Full mask of all vertices
    full_mask = (1 << m) - 1
    
    # Greedy initial solution to get a good lower bound
    def greedy_solution(candidates_mask):
        """Greedy: repeatedly pick vertex with max degree, add to solution, remove it and neighbors."""
        sol = []
        remaining = candidates_mask
        while remaining:
            # Pick vertex with highest degree in remaining subgraph
            best_v = -1
            best_deg = -1
            for v in range(m):
                if (remaining >> v) & 1:
                    deg = bin(remaining & adj[v]).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            sol.append(best_v)
            # Remove best_v and all its neighbors from remaining
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return sol
    
    initial_sol = greedy_solution(full_mask)
    best_size = len(initial_sol)
    best_set = list(initial_sol)
    
    # Greedy coloring bound on subgraph induced by mask
    def color_bound(mask):
        """
        Compute a greedy coloring upper bound on the maximum independent set size
        for the subgraph induced by 'mask'.
        Returns the number of colors used (upper bound on independence number).
        """
        if mask == 0:
            return 0
        
        # Order vertices by degree in the subgraph (descending)
        vertices = [v for v in range(m) if (mask >> v) & 1]
        deg = {}
        for v in vertices:
            deg[v] = bin(mask & adj[v]).count('1')
        vertices.sort(key=lambda v: -deg[v])
        
        colors = {}
        color_num = 0
        for v in vertices:
            neighbor_colors = set()
            for u in vertices:
                if u != v and (adj[v] >> u) & 1 and u in colors:
                    neighbor_colors.add(colors[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_num:
                color_num = c + 1
        
        return color_num
    
    # Branch and bound using bitsets
    def branch_and_bound(candidates_mask, current_size, current_set):
        nonlocal best_set, best_size
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Upper bound: current_size + popcount(candidates_mask)
        if current_size + bin(candidates_mask).count('1') <= best_size:
            return
        
        # Get coloring bound
        cb = color_bound(candidates_mask)
        if current_size + cb <= best_size:
            return
        
        # Choose a vertex to branch on: pick one with highest degree in candidates
        best_v = -1
        best_deg = -1
        for v in range(m):
            if (candidates_mask >> v) & 1:
                deg = bin(candidates_mask & adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        # Remove best_v and all its neighbors from candidates
        new_mask_incl = candidates_mask & ~(1 << best_v) & ~adj[best_v]
        current_set.append(best_v)
        branch_and_bound(new_mask_incl, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: exclude best_v
        new_mask_excl = candidates_mask & ~(1 << best_v)
        branch_and_bound(new_mask_excl, current_size, current_set)
    
    branch_and_bound(full_mask, 0, [])
    
    # Convert back to original numbers
    result = [self_valid[i] for i in best_set]
    
    # Ensure nonempty
    if not result:
        for x in range(1, N + 1):
            if 2 * x not in squares:
                return [x]
        return [1]
    
    return result

# EVOLVE-BLOCK-END
