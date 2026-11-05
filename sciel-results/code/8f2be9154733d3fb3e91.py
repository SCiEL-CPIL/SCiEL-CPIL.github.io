"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build adjacency lists
    # adj[i] contains indices j (0-indexed) such that (i+1) + (j+1) is a square
    # Note: The problem says a<=b, so we consider pairs {a,b} with a<=b.
    # If a+b is a square, we cannot have both a and b in the set.
    # This is a standard graph problem: Max Independent Set.
    
    # First, identify vertices that are "self-conflicting" (2*a is a square).
    # These vertices cannot be in the set.
    valid_indices = []
    for i in range(N):
        a = i + 1
        if 2 * a not in squares:
            valid_indices.append(i)
            
    M = len(valid_indices)
    
    # Map valid_indices to 0..M-1 for the solver
    # adj[i] will be a set of neighbors in the 0..M-1 space
    adj = [set() for _ in range(M)]
    
    for i_idx in range(M):
        a = valid_indices[i_idx] + 1
        for j_idx in range(i_idx + 1, M):
            b = valid_indices[j_idx] + 1
            if a + b in squares:
                adj[i_idx].add(j_idx)
                adj[j_idx].add(i_idx)
                
    # Convert to lists for faster iteration if needed, but sets are fine for small M
    # M is at most 256.
    
    best_size = 0
    best_set = []
    
    # Branch and Bound for Maximum Independent Set
    # We use a recursive backtracking approach with coloring for upper bound.
    # To keep it simple and robust within time limits, we implement a standard B&B.
    
    # Sorting vertices by degree (descending) often helps B&B
    # Let's re-index based on degree to improve pruning? 
    # Actually, standard Tomita algorithm uses dynamic coloring.
    # Given M <= 256, a simple B&B might be too slow without good pruning.
    # However, the graph is sparse.
    
    # Let's try a greedy heuristic first to get a good lower bound.
    def greedy_mis():
        remaining = set(range(M))
        current_set = []
        while remaining:
            # Pick vertex with lowest degree in remaining subgraph? Or highest?
            # For MIS, usually pick vertex with min degree in the induced subgraph of remaining.
            min_deg = float('inf')
            min_v = -1
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            current_set.append(min_v)
            remaining.remove(min_v)
            # Remove neighbors of min_v from remaining
            for u in adj[min_v]:
                if u in remaining:
                    remaining.remove(u)
        return current_set
        
    initial_set = greedy_mis()
    best_size = len(initial_set)
    best_set = initial_set[:]
    
    # Simple Branch and Bound
    # P: set of candidates
    # X: set of vertices already in the independent set
    
    def expand(P, X, size):
        nonlocal best_size, best_set
        nonlocal start_time
        
        if time.time() - start_time > DEADLINE:
            return
            
        if not P:
            if size > best_size:
                best_size = size
                best_set = X[:]
            return
            
        # Upper bound: size + len(P)
        # If size + len(P) <= best_size, prune
        if size + len(P) <= best_size:
            return
            
        # Simple coloring bound? 
        # For small graphs, len(P) is a decent bound.
        # Better bound: Greedy coloring of the subgraph induced by P.
        # The number of colors is an upper bound on the size of the MIS in P.
        # Let's implement a quick greedy coloring for the bound.
        
        def color_bound(P_set):
            # Greedy coloring
            colors = {}
            for v in P_set:
                neighbor_colors = set()
                for u in adj[v]:
                    if u in P_set and u in colors:
                        neighbor_colors.add(colors[u])
                # Find smallest available color
                c = 1
                while c in neighbor_colors:
                    c += 1
                colors[v] = c
            return max(colors.values()) if colors else 0
            
        bound = color_bound(P)
        if size + bound <= best_size:
            return
            
        # Select a vertex to branch on
        # Heuristic: pick vertex with highest degree in P
        v = max(P, key=lambda x: len(adj[x] & P))
        
        # Branch 1: Include v
        X.append(v)
        P_new = P - {v}
        # Remove neighbors of v from P
        for u in adj[v]:
            if u in P_new:
                P_new.discard(u)
        expand(P_new, X, size + 1)
        X.pop()
        
        # Branch 2: Exclude v
        P.discard(v)
        expand(P, X, size)
        P.add(v) # Restore for next branch (though we passed P by value effectively via set copy in recursion? No, sets are mutable. Need to be careful.)
        
    # The above implementation modifies P in place which is dangerous.
    # Let's refactor to pass sets by copy or handle carefully.
    
    def expand_safe(P, X, size):
        nonlocal best_size, best_set
        nonlocal start_time
        
        if time.time() - start_time > DEADLINE:
            return
            
        if not P:
            if size > best_size:
                best_size = size
                best_set = X[:]
            return
            
        # Upper bound check
        if size + len(P) <= best_size:
            return
            
        # Coloring bound
        # To save time, we only compute the tighter coloring bound if len(P) is large enough to matter
        # or if the simple bound doesn't prune.
        # Actually, let's just use the simple bound first. If it passes, we might use coloring.
        
        # For M=256, coloring is O(M^2) which is fast.
        def color_bound(P_set):
            colors = {}
            # Order vertices by degree descending for better coloring?
            # Standard greedy is fine for bound.
            for v in P_set:
                neighbor_colors = set()
                for u in adj[v]:
                    if u in P_set and u in colors:
                        neighbor_colors.add(colors[u])
                c = 1
                while c in neighbor_colors:
                    c += 1
                colors[v] = c
            return max(colors.values()) if colors else 0
            
        bound = color_bound(P)
        if size + bound <= best_size:
            return
            
        # Select vertex
        # Heuristic: pick vertex with max degree in P
        v = max(P, key=lambda x: len(adj[x] & P))
        
        # Branch 1: Include v
        X.append(v)
        P_new = set(P)
        P_new.discard(v)
        for u in adj[v]:
            P_new.discard(u)
        expand_safe(P_new, X, size + 1)
        X.pop()
        
        # Branch 2: Exclude v
        P_new = set(P)
        P_new.discard(v)
        expand_safe(P_new, X, size)
        
    expand_safe(set(range(M)), [], 0)
    
    # Map back to original values
    result = [valid_indices[i] + 1 for i in best_set]
    return result

# EVOLVE-BLOCK-END
