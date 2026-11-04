"""Rewritten program: branch-and-bound clique on compatibility graph of 4-subsets,
then expand each selected support with all 16 sign patterns (Type D vectors)."""
import numpy as np
import time
import itertools

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Fallback seed: all vectors with two non-zero entries of value +/-1 (norm^2 = 2)
    # Total: C(11,2) * 4 = 220 vectors
    v_seed = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                v_seed.append(v)
    seed_arr = np.array(v_seed, dtype=np.int64)

    # Generate all 4-subsets (supports) of range(11)
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330

    # Build compatibility graph: edge (i,j) iff |support_i ∩ support_j| <= 2
    # For 4-subsets, intersection size is 0,1,2,3,4. Compatible iff <= 2.
    # Precompute support membership as bitmasks for fast intersection.
    masks = [0] * n_sup
    for idx, sup in enumerate(supports):
        m = 0
        for pos in sup:
            m |= (1 << pos)
        masks[idx] = m

    # Build adjacency as list of sets (for compatibility: intersection <= 2)
    # adj[i] = set of j > i such that |sup_i ∩ sup_j| <= 2
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        mi = masks[i]
        for j in range(i + 1, n_sup):
            mj = masks[j]
            inter = mi & mj
            # Count bits
            popcount = bin(inter).count('1')
            if popcount <= 2:
                adj[i].add(j)
                adj[j].add(i)

    # Now solve maximum clique on this compatibility graph using Tomita's
    # branch-and-bound with pivot-based coloring.
    #
    # Tomita's algorithm (MCQ):
    # - Maintain a candidate set P (nodes that can be added to current clique).
    # - Color P greedily to get upper bounds.
    # - Branch on nodes in reverse color order.
    # - Use pivot: pick a node u in P that maximizes |P ∩ N(u)|, then
    #   branch on P \ N(u) first (or use the pivot to reduce candidates).

    def maximum_clique(adj, n, deadline, start_time):
        # adj: list of sets (adjacency)
        # Returns the maximum clique as a list of node indices.
        
        best_clique = []
        best_size = 0

        # Convert adj to a list of lists for faster access
        adj_list = [list(s) for s in adj]
        adj_set = [set(s) for s in adj]

        # Greedy coloring of a list of nodes (in the order given)
        # Returns (colors, order) where colors[i] is the color of node order[i]
        # and order is a permutation of the input nodes sorted by color.
        def color_sort(nodes):
            # nodes: list of node indices
            # Greedy coloring: assign smallest available color
            # Returns (color_list, order) where color_list[i] = color of order[i]
            uncolored = set(nodes)
            color_of = {}
            order = []
            # We'll do a simple greedy: repeatedly pick the uncolored node
            # with the most already-colored neighbors, assign smallest color.
            # Actually, standard Tomita: process nodes in given order, assign
            # smallest color not used by its already-colored neighbors.
            # Then sort by color (stable).
            
            # Process in the given order
            color_of = {}
            for node in nodes:
                # Find colors used by neighbors of node that are already colored
                used = set()
                for nb in adj_set[node]:
                    if nb in color_of:
                        used.add(color_of[nb])
                # Assign smallest positive integer not in used
                c = 1
                while c in used:
                    c += 1
                color_of[node] = c
            
            # Sort by color (stable sort preserves original order within same color)
            order = sorted(nodes, key=lambda x: color_of[x])
            color_list = [color_of[node] for node in order]
            return color_list, order

        # Recursive branch-and-bound
        # P: list of candidate nodes (all compatible with current clique)
        # current_clique: list of nodes in current clique
        def expand(P, current_clique):
            nonlocal best_clique, best_size
            
            if time.time() > deadline:
                return
            
            if len(P) + len(current_clique) <= best_size:
                return
            
            if not P:
                if len(current_clique) > best_size:
                    best_size = len(current_clique)
                    best_clique = current_clique[:]
                return
            
            # Color sort P
            color_list, order = color_sort(P)
            
            # Process in reverse order (highest color first)
            # For each node in reverse order, if color >= potential improvement, branch
            for idx in range(len(order) - 1, -1, -1):
                if time.time() > deadline:
                    return
                
                node = order[idx]
                max_possible = len(current_clique) + idx + 1
                if max_possible <= best_size:
                    break
                
                # Pivot: find a node u in P[0:idx+1] that maximizes |P[0:idx+1] ∩ N(u)|
                # This helps prune: we only need to branch on nodes in P[0:idx+1] \ N(u)
                # but for simplicity and correctness, we'll use the standard approach:
                # branch on 'node', then remove 'node' from P and continue.
                
                # Actually, Tomita's MCQ uses the pivot to select which node to branch on.
                # Let's use a simpler but effective variant:
                # Branch on 'node': add it to clique, recurse on P ∩ N(node).
                # Then remove 'node' from P.
                
                # New P for recursion: nodes in P[0:idx] that are neighbors of 'node'
                new_P = [p for p in order[:idx] if p in adj_set[node]]
                expand(new_P, current_clique + [node])
                
                # Remove node from P (conceptually, we just continue with order[:idx])
                # The loop continues with idx-1, which effectively removes 'node'
            
        # Initial call: P = all nodes, current_clique = []
        all_nodes = list(range(n_sup))
        expand(all_nodes, [])
        
        return best_clique

    # Run the maximum clique algorithm
    clique = maximum_clique(adj, n_sup, deadline, start_time)
    
    if time.time() > deadline:
        return seed_arr

    # Expand each selected support with all 16 sign patterns
    # For a 4-subset (i,j,k,l), generate all 16 vectors with ±1 at those positions
    vectors = []
    for sup in clique:
        if not isinstance(sup, tuple):
            sup = supports[sup]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(sup, signs):
                v[pos] = s
            vectors.append(v)
    
    if len(vectors) == 0:
        return seed_arr
    
    result = np.array(vectors, dtype=np.int64)
    
    # Verify the result satisfies the constraint
    # min pairwise distance >= max norm
    if len(result) < 2:
        return result
    
    # Compute norms
    norms_sq = np.sum(result**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # For efficiency, check a sample or do full check if small enough
    # Full check: O(n^2 * d)
    n = len(result)
    if n <= 5000:
        # Compute all pairwise distances
        # dist^2 = ||a||^2 + ||b||^2 - 2 a.b
        # Use matrix multiplication
        dots = result @ result.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        # Check off-diagonal
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        
        if min_dist_sq < max_norm_sq:
            # Invalid, return seed
            return seed_arr
    
    if len(result) > len(seed_arr):
        return result
    else:
        return seed_arr

# EVOLVE-BLOCK-END
