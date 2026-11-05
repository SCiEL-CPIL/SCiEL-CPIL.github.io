import math
import time
from itertools import combinations

def construct():
    n = 255
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    primes = [3, 5, 17]
    
    # Base: all multiples of 3 (size 85) is a valid clique containing 255
    base_set = list(range(3, n + 1, 3))
    best_set = base_set
    best_size = len(base_set)
    
    # Build candidate set: all numbers in [1,255] divisible by 3, 5, or 17
    candidates = set()
    for p in primes:
        for x in range(p, n + 1, p):
            candidates.add(x)
    candidates = sorted(candidates)
    
    if 255 not in candidates:
        candidates.append(255)
        candidates.sort()
    
    m = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build compatibility graph: edge (i,j) iff gcd(candidates[i], candidates[j]) > 1
    # Use bitmasks for efficient neighborhood operations
    neighbors = [0] * m  # bitmask of neighbors for each vertex
    
    for i in range(m):
        ci = candidates[i]
        for j in range(i + 1, m):
            cj = candidates[j]
            if math.gcd(ci, cj) > 1:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
    
    # For maximum clique containing 255, we need a clique in the subgraph
    # induced by {255} ∪ N(255) that includes 255.
    # Equivalently: find max clique in N(255) and add 255.
    
    # Get neighbors of 255 (as indices)
    nbrs_255_mask = neighbors[idx_255]
    nbrs_255 = [i for i in range(m) if i != idx_255 and (nbrs_255_mask >> i) & 1]
    
    if not nbrs_255:
        return base_set
    
    # Restrict to subgraph induced by nbrs_255
    # Remap: local index k -> global index nbrs_255[k]
    k = len(nbrs_255)
    
    # Build local adjacency bitmasks
    local_adj = [0] * k
    for a in range(k):
        ga = nbrs_255[a]
        for b in range(a + 1, k):
            gb = nbrs_255[b]
            if (neighbors[ga] >> gb) & 1:
                local_adj[a] |= (1 << b)
                local_adj[b] |= (1 << a)
    
    # Tomita's algorithm for maximum clique with coloring bound
    # This is exact and should handle ~100 vertices easily
    
    best_clique = []
    best_clique_size = 0
    
    def color_sort(R_set, P_set):
        """Color sort vertices in P to get an upper bound on clique size in P.
        Returns (sorted_vertices, color_bounds) where color_bounds[i] is the
        color assigned to the i-th vertex in sorted order (1-indexed colors).
        """
        if not P_set:
            return [], []
        
        # Greedy coloring: assign colors so that no two adjacent vertices share a color
        # This gives an upper bound: any clique in P has size <= max color
        uncolored = set(P_set)
        colors = {}
        color_classes = []  # list of sets, one per color
        
        while uncolored:
            # Greedily pick a maximal independent set for this color
            color_class = set()
            remaining = set(uncolored)
            while remaining:
                # Pick vertex with minimum degree in remaining (for better bounds)
                v = min(remaining, key=lambda x: sum(1 for y in remaining if y != x and (local_adj[x] >> y) & 1))
                color_class.add(v)
                # Remove v and its neighbors from remaining
                to_remove = {v}
                for y in list(remaining):
                    if y != v and (local_adj[v] >> y) & 1:
                        to_remove.add(y)
                remaining -= to_remove
            
            color_classes.append(color_class)
            for v in color_class:
                colors[v] = len(color_classes)
            uncolored -= color_class
        
        # Sort vertices by color (ascending) - vertices with lower color first
        sorted_vertices = sorted(color_classes, key=len, reverse=False)
        # Actually, we want to sort so that the coloring gives a valid upper bound
        # Standard approach: sort by color, then process in reverse for the bound
        
        # Return sorted list and the color of each vertex
        sorted_verts = []
        for cc in color_classes:
            for v in cc:
                sorted_verts.append(v)
        
        # Re-sort: process in order of increasing color for the bound
        # The bound at position i is the color of the vertex at position i
        # when vertices are sorted by color
        sorted_by_color = sorted(uncolored | set().union(*color_classes), key=lambda x: colors[x])
        
        # Actually let's just return the color assignment and let the caller sort
        return list(range(k)), colors
    
    # Simpler approach: Bron-Kerbosch with pivot and a simple coloring bound
    def max_clique(P, X, R, depth=0):
        """Bron-Kerbosch with pivot.
        P: set of candidates (bitmask or list)
        X: set of excluded (for completeness, not needed for max)
        R: current clique
        """
        nonlocal best_clique, best_clique_size
        
        # If R is larger than current best, update
        if len(R) > best_clique_size:
            best_clique_size = len(R)
            best_clique = list(R)
        
        if not P:
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        # Use a simple heuristic: pick vertex with max degree in P
        if P:
            # Pick pivot with max number of neighbors in P
            pivot = max(P, key=lambda u: sum(1 for v in P if v != u and (local_adj[u] >> v) & 1))
            # Branch on P \ N(pivot)
            candidates_to_branch = [v for v in P if not ((local_adj[pivot] >> v) & 1)]
        else:
            return
        
        for v in candidates_to_branch:
            # Include v in clique
            new_R = R | {v}
            new_P = P & set()  # need intersection with N(v)
            new_P = {u for u in P if u != v and (local_adj[v] >> u) & 1}
            
            max_clique(new_P, set(), new_R, depth + 1)
            
            P.discard(v)
            # X.add(v)  # not needed for maximum clique
        
    # Use a bitmask-based approach for speed
    # P is a bitmask of available vertices
    def max_clique_bitmask(P_mask, R_mask, R_size):
        nonlocal best_clique, best_clique_size
        
        if R_size > best_clique_size:
            best_clique_size = R_size
            # Reconstruct clique from R_mask
            clique = []
            for i in range(k):
                if (R_mask >> i) & 1:
                    clique.append(nbrs_255[i])
            best_clique = clique
        
        if not P_mask:
            return
        
        # Choose pivot
        # Count bits in P_mask
        # Pick pivot with max degree in P
        pivot = -1
        max_deg = -1
        for i in range(k):
            if (P_mask >> i) & 1:
                deg = bin(P_mask & local_adj[i]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    pivot = i
        
        if pivot == -1:
            return
        
        # Branch on P \ N(pivot)
        branch_mask = P_mask & ~(local_adj[pivot])
        
        while branch_mask:
            # Pick lowest set bit
            v = (branch_mask & (-branch_mask)).bit_length() - 1
            branch_mask &= branch_mask - 1
            
            new_R_mask = R_mask | (1 << v)
            new_P_mask = P_mask & local_adj[v]
            
            max_clique_bitmask(new_P_mask, new_R_mask, R_size + 1)
            
            P_mask &= ~(1 << v)
    
    # Run maximum clique on the subgraph induced by neighbors of 255
    initial_P = (1 << k) - 1  # all vertices
    max_clique_bitmask(initial_P, 0, 0)
    
    # The max clique in N(255) plus 255 gives a clique of size best_clique_size + 1
    if best_clique_size + 1 > best_size:
        # Convert back to original values
        result = [255] + [candidates[i] for i in best_clique]
        # Verify
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) == 1:
                    valid = False
                    break
            if not valid:
                break
        if valid and 255 in result:
            best_set = result
            best_size = len(result)
    
    # Verify final result
    if len(best_set) > 1:
        for i in range(len(best_set)):
            for j in range(i + 1, len(best_set)):
                if math.gcd(best_set[i], best_set[j]) == 1:
                    return base_set
    
    if 255 not in best_set:
        return base_set
    
    return best_set
