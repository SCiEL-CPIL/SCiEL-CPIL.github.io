"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    """
    Solve Maximum Weight Independent Set on the coprime-conflict graph for 1..511.
    Nodes: integers 1..511. Edge between i,j if gcd(i,j) > 1 (conflict).
    We want max sum of 1/(512 - a) over an independent set (pairwise coprime).
    
    Use branch-and-bound with greedy coloring upper bounds on the complement graph
    (where edges = coprime pairs), solving as Maximum Weight Clique.
    """
    start_time = time.time()
    TIME_LIMIT = 110.0  # safety margin before 120s limit
    
    n = 511  # integers 1..511
    nodes = list(range(1, 512))
    
    # Precompute weights: w[i] = 1/(512 - i)
    weights = [Fraction(1, 512 - i) for i in range(n + 1)]  # 1-indexed
    
    # Precompute coprime relation: coprime[i][j] = True if gcd(i,j)==1
    # For the conflict graph, edge exists if NOT coprime
    # For the complement (clique) graph, edge exists if coprime
    
    # Build adjacency bitsets for the coprime graph (complement of conflict)
    # Using Python integers as bitsets for efficiency
    coprime_adj = [0] * (n + 1)  # 1-indexed
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if gcd(i, j) == 1:
                coprime_adj[i] |= (1 << j)
                coprime_adj[j] |= (1 << i)
    
    # Precompute popcount helper
    def popcount(x):
        return bin(x).count('1')
    
    # Greedy coloring of a set of vertices (bitmask) in the coprime graph
    # Returns list of color classes (each a bitmask), giving upper bound = len(colors)
    def greedy_color(vertices_mask):
        """
        Greedy coloring: assign colors to vertices in the subgraph induced by vertices_mask.
        Returns number of colors used (upper bound on clique size).
        For weighted case, we use a weighted version.
        """
        if vertices_mask == 0:
            return 0
        
        # Order vertices by weight descending for better bounds
        verts = []
        for i in range(1, n + 1):
            if (vertices_mask >> i) & 1:
                verts.append(i)
        verts.sort(key=lambda x: -weights[x])
        
        # Greedy coloring
        color_of = {}
        num_colors = 0
        for v in verts:
            # Find available colors: colors not used by neighbors of v in already-colored set
            used_colors = set()
            for u in verts:
                if u == v:
                    break
                if u in color_of:
                    # Check if u is neighbor of v in coprime graph
                    if (coprime_adj[v] >> u) & 1:
                        used_colors.add(color_of[u])
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            num_colors = max(num_colors, c + 1)
        
        return num_colors
    
    # Weighted upper bound: sum of weights of vertices in each color class
    # For a clique, at most one vertex per color class can be chosen
    # So upper bound = sum over color classes of max weight in that class
    def weighted_color_bound(vertices_mask):
        """
        Returns upper bound on max weight clique in subgraph induced by vertices_mask.
        Uses greedy coloring: bound = sum of max weight per color class.
        """
        if vertices_mask == 0:
            return Fraction(0)
        
        verts = []
        for i in range(1, n + 1):
            if (vertices_mask >> i) & 1:
                verts.append(i)
        verts.sort(key=lambda x: -weights[x])
        
        # Greedy coloring
        color_classes = []  # list of lists of vertices
        color_of = {}
        for v in verts:
            used_colors = set()
            for u in verts:
                if u == v:
                    break
                if u in color_of and (coprime_adj[v] >> u) & 1:
                    used_colors.add(color_of[u])
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(v)
        
        # Upper bound: sum of max weight in each color class
        bound = Fraction(0)
        for cc in color_classes:
            bound += max(weights[v] for v in cc)
        
        return bound
    
    # Branch and bound for maximum weight clique
    best_weight = Fraction(0)
    best_clique = []
    node_count = [0]
    
    def bnb(candidates_mask, current_clique, current_weight):
        """
        candidates_mask: bitmask of vertices still available (all coprime with current_clique)
        current_clique: list of vertices in current clique
        current_weight: sum of weights of current_clique
        """
        nonlocal best_weight, best_clique
        
        # Time check
        if time.time() - start_time > TIME_LIMIT:
            return
        
        node_count[0] += 1
        
        if candidates_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = current_clique[:]
            return
        
        # Upper bound using weighted coloring
        ub = weighted_color_bound(candidates_mask)
        if current_weight + ub <= best_weight:
            return
        
        # Order candidates by weight descending
        cands = []
        for i in range(1, n + 1):
            if (candidates_mask >> i) & 1:
                cands.append(i)
        cands.sort(key=lambda x: -weights[x])
        
        for v in cands:
            if time.time() - start_time > TIME_LIMIT:
                return
            
            new_weight = current_weight + weights[v]
            if new_weight <= best_weight:
                break  # since sorted by weight descending, further vertices have lower weight
            
            new_clique = current_clique + [v]
            
            # New candidates: those in candidates_mask that are coprime with v
            new_candidates = candidates_mask & coprime_adj[v]
            # Remove v itself
            new_candidates &= ~(1 << v)
            
            bnb(new_candidates, new_clique, new_weight)
            
            # Remove v from candidates for subsequent iterations
            candidates_mask &= ~(1 << v)
            
            # Re-check bound after removing v
            if candidates_mask == 0:
                break
            ub = weighted_color_bound(candidates_mask)
            if current_weight + ub <= best_weight:
                break
    
    # Initialize with a greedy solution as lower bound
    greedy_clique = []
    greedy_weight = Fraction(0)
    remaining = (1 << (n + 1)) - 2  # all vertices 1..511
    for i in range(511, 0, -1):  # try high-weight vertices first (small i means large 1/(512-i))
        # Actually weight = 1/(512-i), so small i has large weight
        # Let's sort by weight descending
        pass
    
    # Better greedy: sort by weight descending
    all_nodes = list(range(1, 512))
    all_nodes.sort(key=lambda x: -weights[x])
    for v in all_nodes:
        if all(gcd(v, u) == 1 for u in greedy_clique):
            greedy_clique.append(v)
            greedy_weight += weights[v]
    
    best_weight = greedy_weight
    best_clique = greedy_clique[:]
    
    # Run B&B
    initial_mask = 0
    for i in range(1, n + 1):
        initial_mask |= (1 << i)
    
    bnb(initial_mask, [], Fraction(0))
    
    # Verify result
    result = best_clique
    # Ensure distinct and in range
    result = sorted(set(result))
    
    return result
# EVOLVE-BLOCK-END
