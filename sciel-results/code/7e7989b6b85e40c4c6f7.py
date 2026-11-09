"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [2, 256] such that lcm(a, b) > 256 for all distinct pairs a, b.
    Maximize the exact reciprocal sum (sum of 1/x for x in the subset).
    
    Approach:
    1. Build a conflict graph where an edge exists between a and b if lcm(a, b) <= 256.
       We want a subset with NO edges in this graph, i.e., an independent set.
    2. Equivalently, find a maximum-weight clique in the complement graph (where edges
       represent "compatible" pairs, i.e., lcm(a, b) > 256).
    3. Use Bron-Kerbosch with pivot selection and greedy coloring bounds to find the
       maximum-weight clique exactly.
    
    The weight of each vertex is its reciprocal 1/x. We maximize the sum of weights.
    """
    from math import lcm
    from fractions import Fraction
    
    # Candidate set: [2, 256]
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build compatibility matrix: compat[i][j] = True if lcm(candidates[i], candidates[j]) > 256
    # For i == j, we consider it compatible (self-loop not an issue for clique)
    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        compat[i][i] = True
        for j in range(i + 1, n):
            if lcm(candidates[i], candidates[j]) > 256:
                compat[i][j] = True
                compat[j][i] = True
    
    # Weights: reciprocal of each candidate
    weights = [Fraction(1, candidates[i]) for i in range(n)]
    
    # We'll use Bron-Kerbosch with pivot and coloring bounds.
    # Since weights are positive, we want to maximize the sum of weights in the clique.
    # The standard Bron-Kerbosch finds maximal cliques; with branch-and-bound using
    # upper bounds on the remaining weight, we can find the maximum-weight clique.
    
    # For the coloring bound, we need an ordering. We'll sort vertices by weight descending
    # to help with pruning. But actually, the coloring bound works on the subgraph induced
    # by the candidate set R. Let's implement a weighted version.
    
    # Actually, let's think about this differently. The number of vertices is 255, which is
    # manageable but we need efficient pruning. Let's use a recursive branch-and-bound.
    
    # State: 
    #   R: set of vertices in the current clique
    #   P: set of vertices that can be added to R (compatible with all in R)
    #   X: set of vertices already processed (not needed for correctness in maximal clique,
    #      but for maximum weight we don't need X if we're doing proper branch-and-bound)
    
    # We'll track the best weight found and the best clique.
    
    best_weight = Fraction(0)
    best_clique = []
    
    # Convert compat to adjacency lists for faster access
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if compat[i][j] and i != j:
                adj[i].add(j)
    
    # For the branch-and-bound, we need to compute an upper bound on the weight we can
    # add from a set of candidate vertices P. A simple upper bound is the sum of weights
    # of all vertices in P (since we can't take more than all of them). A tighter bound
    # would use the structure, but let's start with this.
    
    # Actually, a better approach: use the fact that we're looking for a clique in the
    # compatibility graph. The maximum weight clique problem is NP-hard, but with n=255
    # and good pruning, it might work. Let's try a different approach.
    
    # Alternative: since the problem has structure (lcm constraints), maybe we can exploit
    # it. But let's first try the general approach.
    
    # Let's implement Bron-Kerbosch with pivot for finding maximal cliques, and then
    # among all maximal cliques, find the one with maximum weight. But this could be
    # exponential.
    
    # Better: branch-and-bound where at each step we choose a vertex to include or exclude,
    # and use upper bounds to prune.
    
    # Let's sort vertices by weight descending for better pruning.
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    # Map from original index to position in order
    pos_in_order = {v: i for i, v in enumerate(order)}
    
    # We'll work with the ordered list. Let's redefine adj in terms of the ordered indices.
    # Actually, let's just use the original indices but process them in order.
    
    # For the upper bound, given a set of candidate vertices P, the maximum additional
    # weight is at most sum(weights[i] for i in P). We can improve this by noting that
    # not all of P can be in a clique together.
    
    # Let's use a greedy coloring bound. In the subgraph induced by P, we can compute a
    # proper coloring. The number of colors gives an upper bound on the clique size,
    # but for weights, we need a weighted version.
    
    # For weighted coloring bound: order vertices by weight, assign colors greedily,
    # and the upper bound is the sum of the maximum weight in each color class... no,
    # that's not quite right either.
    
    # Let's use a simpler approach: since n=255 is not too large, let's try a recursive
    # search with good pruning.
    
    # State: current clique (list of indices), candidate set (set of indices compatible
    # with all in current clique), and the starting index in the order to avoid duplicates.
    
    # We'll use a recursive function that tries to add vertices in order.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def search(clique, candidates_set, start_idx):
        """
        clique: list of vertex indices in the current clique
        candidates_set: set of vertex indices that are compatible with all in clique
                        and have position >= start_idx in the order
        start_idx: the minimum position in order to consider (to avoid duplicates)
        """
        nonlocal best_weight, best_clique
        
        current_weight = sum(weights[i] for i in clique)
        
        # Upper bound: current weight + sum of weights of all candidates
        upper_bound = current_weight + sum(weights[i] for i in candidates_set)
        if upper_bound <= best_weight:
            return
        
        if not candidates_set:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = list(clique)
            return
        
        # Choose pivot: the vertex in candidates_set with the most neighbors in candidates_set
        # This helps with branching
        pivot = None
        max_neighbors = -1
        for v in candidates_set:
            neighbors_in_candidates = len(adj[v] & candidates_set)
            if neighbors_in_candidates > max_neighbors:
                max_neighbors = neighbors_in_candidates
                pivot = v
        
        # Branch on vertices in candidates_set that are NOT neighbors of pivot
        # (plus pivot itself if we want, but standard Bron-Kerbosch branches on
        # candidates_set \ neighbors(pivot))
        if pivot is not None:
            branch_candidates = [v for v in candidates_set if v not in adj[pivot]]
        else:
            branch_candidates = list(candidates_set)
        
        # Sort branch_candidates by position in order
        branch_candidates.sort(key=lambda v: pos_in_order[v])
        
        for v in branch_candidates:
            # Add v to clique
            new_clique = clique + [v]
            # New candidates: those in candidates_set that are compatible with v and come after v in order
            new_candidates = {u for u in candidates_set if u in adj[v] and pos_in_order[u] > pos_in_order[v]}
            
            search(new_clique, new_candidates, pos_in_order[v] + 1)
            
            # Remove v from candidates_set for subsequent iterations
            candidates_set.discard(v)
    
    # Initial candidates: all vertices
    initial_candidates = set(range(n))
    search([], initial_candidates, 0)
    
    # Convert best_clique (indices) back to values
    result = [candidates[i] for i in best_clique]
    
    # Ensure nonempty
    if not result:
        # Fallback: pick the single element with largest reciprocal (smallest value >= 2)
        result = [2]
    
    return result
# EVOLVE-BLOCK-END
