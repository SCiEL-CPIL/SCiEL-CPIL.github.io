import time
from math import gcd

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that
    gcd(a, b) > 1 for all distinct pairs a, b.
    
    Uses bitset-based Bron-Kerbosch with greedy coloring for exact max clique,
    then local search to extend the result.
    """
    deadline = time.time() + 80.0  # 80s safety margin before 90s limit
    t0 = time.time()
    
    # Build candidate set: numbers in [1,255] divisible by 3, 5, or 17
    candidates = [i for i in range(1, 256) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n = len(candidates)
    idx_map = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency bitsets: adj[i] is a bitset where bit j is set if gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Greedy coloring for upper bound (McPhee order)
    def greedy_coloring(P):
        """
        Greedy coloring of subgraph induced by P (bitset).
        Returns (order, colors) where order is list of vertices and colors[v] is color.
        Vertices are ordered by non-increasing color.
        """
        P_list = [i for i in range(n) if (P >> i) & 1]
        if not P_list:
            return [], {}
        
        # Sort by degree within P (descending) for better coloring
        P_list.sort(key=lambda v: bin((adj[v] & P).bit_count()), reverse=True)
        
        colors = {}
        order = []
        uncolored = set(P_list)
        
        while uncolored:
            # Greedy: pick vertex with max degree in uncolored
            v = max(uncolored, key=lambda x: bin((adj[x] & sum((1 << u) for u in uncolored) if u != x else 0).bit_count()) if len(uncolored) > 1 else 0)
            # Simpler: just iterate
            v = next(iter(uncolored))
            uncolored.discard(v)
            
            # Find smallest available color
            used = set()
            for u in order:
                if (adj[v] >> u) & 1:
                    used.add(colors[u])
            c = 1
            while c in used:
                c += 1
            colors[v] = c
            order.append(v)
        
        return order, colors
    
    def greedy_coloring_fast(P):
        """Faster greedy coloring using a simple sequential approach."""
        P_list = [i for i in range(n) if (P >> i) & 1]
        if not P_list:
            return [], {}
        
        # Sort by degree in P descending
        P_list.sort(key=lambda v: bin((adj[v] & P).bit_count()), reverse=True)
        
        colors = {}
        order = []
        for v in P_list:
            used = set()
            for u in order:
                if (adj[v] >> u) & 1:
                    used.add(colors[u])
            c = 1
            while c in used:
                c += 1
            colors[v] = c
            order.append(v)
        
        return order, colors
    
    # Bron-Kerbosch with coloring
    def bron_kerbosch(P, X, R, best):
        """
        P, X: bitsets of candidate and excluded vertices
        R: set of vertices in current clique
        best: [set] reference to best clique found
        """
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best[0]):
                best[0] = set(R)
            return
        
        # Greedy coloring
        order, colors = greedy_coloring_fast(P)
        
        # Process in reverse order (highest color first)
        for idx in range(len(order) - 1, -1, -1):
            if time.time() > deadline:
                return
            v = order[idx]
            # Upper bound: current clique size + color of v
            if len(R) + colors[v] <= len(best[0]):
                break
            
            # Expand: add v to clique, recurse on neighbors of v in P
            new_R = R | {v}
            new_P = P & adj[v] & ~(1 << v)
            new_X = X & adj[v] & ~(1 << v)
            
            bron_kerbosch(new_P, new_X, new_R, best)
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)
    
    # Start with a good initial solution: all multiples of 3
    mult3 = set(i for i in range(3, 256, 3))
    best = [mult3.copy()]
    
    # Run Bron-Kerbosch to find exact maximum clique
    # We need to include 255 in the result, so restrict to vertices connected to 255
    idx_255 = idx_map[255]
    # All candidates share a factor with 255, but not all are connected to 255 in the gcd graph
    # Actually, 255 = 3*5*17, so gcd(255, x) > 1 iff x is divisible by 3, 5, or 17
    # Which is exactly our candidate set. So all candidates are adjacent to 255.
    
    full_P = (1 << n) - 1
    bron_kerbosch(full_P, 0, set(), best)
    
    # Ensure 255 is in the result
    if 255 not in best[0]:
        # Add 255 and remove any element not connected to it (shouldn't happen)
        best[0].add(255)
        to_remove = [x for x in best[0] if x != 255 and gcd(x, 255) == 1]
        for x in to_remove:
            best[0].discard(x)
    
    # Local search: try to add elements that connect to all in current set
    def try_add(current, v):
        """Check if v can be added to current (connected to all elements in current)."""
        if v in current:
            return True
        vi = idx_map[v]
        for u in current:
            if u == v:
                continue
            ui = idx_map[u]
            if not ((adj[vi] >> ui) & 1):
                return False
        return True
    
    # Greedy addition pass
    current = set(best[0])
    improved = True
    while improved:
        improved = False
        for v in candidates:
            if v not in current and try_add(current, v):
                current.add(v)
                improved = True
                if len(current) > len(best[0]):
                    best[0] = current.copy()
    
    # Local search with swaps: try removing one element and adding another
    for iteration in range(50):
        if time.time() > deadline:
            break
        improved = False
        current = set(best[0])
        
        for u in list(current):
            if time.time() > deadline:
                break
            for v in candidates:
                if v in current:
                    continue
                # Check if v connects to all in current except u
                ok = True
                for w in current:
                    if w == u:
                        continue
                    wi = idx_map[w]
                    vi = idx_map[v]
                    if not ((adj[vi] >> wi) & 1):
                        ok = False
                        break
                if ok:
                    new_set = (current - {u}) | {v}
                    if len(new_set) > len(best[0]):
                        best[0] = new_set
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    # Final greedy addition
    current = set(best[0])
    for v in candidates:
        if v not in current and try_add(current, v):
            current.add(v)
    if len(current) > len(best[0]):
        best[0] = current
    
    # Verify 255 is present
    if 255 not in best[0]:
        best[0].add(255)
    
    return list(best[0])
