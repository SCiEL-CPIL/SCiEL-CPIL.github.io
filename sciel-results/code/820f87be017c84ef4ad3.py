import time
from math import gcd

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = 85.0  # Keep 5s margin for safety

    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    idx_map = {val: i for i, val in enumerate(candidates)}
    
    # Precompute adjacency as bitmasks for speed
    adj = [0] * n
    for i in range(n):
        vi = candidates[i]
        mask = 0
        for j in range(n):
            if i != j and gcd(vi, candidates[j]) > 1:
                mask |= (1 << j)
        adj[i] = mask
    
    best_clique = []
    best_size = 0
    
    def bit_count(x):
        return bin(x).count('1')
    
    def get_bits(mask):
        """Yield indices of set bits in mask."""
        while mask:
            lb = mask & (-mask)
            yield lb.bit_length() - 1
            mask ^= lb
    
    def greedy_color(P):
        """
        Greedy coloring of vertices in P (bitmask).
        Returns (order, colors) where order is list of vertex indices
        sorted by color (ascending), and colors[i] is the color of order[i].
        The max color is an upper bound on clique size in P.
        """
        uncolored = P
        order = []
        colors = []
        color = 0
        while uncolored:
            color += 1
            # Find a maximal independent set in the subgraph induced by uncolored
            # For coloring: pick vertices that are NOT adjacent to each other
            # Greedy: repeatedly pick a vertex, remove it and its neighbors from consideration
            # Wait - for proper coloring, same-color vertices must be non-adjacent
            # So we want a maximal independent set
            remaining = uncolored
            color_class = 0
            while remaining:
                # Pick first vertex
                lb = remaining & (-remaining)
                v = lb.bit_length() - 1
                color_class |= lb
                # Remove v and all its neighbors from remaining
                remaining &= ~adj[v]
                remaining &= ~lb
            # All vertices in color_class get this color
            order.extend(get_bits(color_class))
            colors.extend([color] * bit_count(color_class))
            uncolored &= ~color_class
        
        return order, colors, color
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        # Bound: can't beat best
        if len(R) + bit_count(P) <= best_size:
            return
        
        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Greedy coloring for upper bound
        if P != 0:
            order, colors, max_color = greedy_color(P)
            if len(R) + max_color <= best_size:
                return
        
        # Pivot selection
        if P != 0:
            # Pick pivot u in P∪X maximizing |P ∩ N(u)|
            pivot = None
            max_deg = -1
            for u in get_bits(P):
                deg = bit_count(P & adj[u])
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            for u in get_bits(X):
                deg = bit_count(P & adj[u])
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            if pivot is None:
                pivot = next(get_bits(P))
            
            # Branch on P \ N(pivot)
            branch_mask = P & ~adj[pivot]
        else:
            branch_mask = 0
            pivot = None
        
        # Iterate over vertices in branch_mask in color order (higher color first)
        # This helps find large cliques earlier
        if P != 0:
            # Get vertices in branch_mask, ordered by color descending
            branch_vertices = []
            for v in get_bits(branch_mask):
                # Find color of v
                # We need to look up color from the greedy_color result
                # But we didn't store it per-vertex globally, so let's just use natural order
                branch_vertices.append(v)
        else:
            branch_vertices = []
        
        # Process in reverse color order for better pruning
        # Actually, let's just process all vertices in branch_mask
        for v in branch_vertices:
            if time.time() > deadline:
                return
            
            vbit = 1 << v
            
            # Recurse
            R.append(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            P &= ~vbit
            X |= vbit
    
    # Initial solution: multiples of 3
    init_clique = []
    for i in range(n):
        if candidates[i] % 3 == 0:
            init_clique.append(i)
    best_size = len(init_clique)
    best_clique = init_clique
    
    # Run Bron-Kerbosch
    R = []
    P = (1 << n) - 1  # All vertices
    X = 0
    
    bron_kerbosch(R, P, X)
    
    # Ensure 255 is in result
    idx_255 = idx_map[255]
    if idx_255 not in best_clique:
        best_clique.append(idx_255)
    
    result_vals = [candidates[i] for i in best_clique]
    
    # Verify
    valid = True
    for i in range(len(result_vals)):
        for j in range(i+1, len(result_vals)):
            if gcd(result_vals[i], result_vals[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        result_vals = [v for v in candidates if v % 3 == 0]
    
    return result_vals
