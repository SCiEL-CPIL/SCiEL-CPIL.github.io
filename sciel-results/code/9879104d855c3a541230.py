# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares), i.e. forbidden differences are the nonzero
    quadratic residues modulo 205.

    This implementation uses a branch-and-bound algorithm (Tomita-style) to find the
    Maximum Independent Set (MIS) of the conflict graph. The conflict graph has vertices
    0..204, with an edge between u and v if (u-v) % 205 is a forbidden square.
    We find the MIS by finding the Maximum Clique in the complement graph.
    """
    m = 205
    deadline = time.time() + 590  # Safety margin for 600s limit

    # 1. Compute forbidden differences (nonzero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # 2. Build adjacency masks for the conflict graph G
    # adj_mask[u] has bit v set if (u-v) % m is in forbidden
    adj_mask = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v and (u - v) % m in forbidden:
                mask |= (1 << v)
        adj_mask[u] = mask

    # 3. Build adjacency masks for the complement graph H
    # comp_mask[u] has bit v set if u and v are NOT adjacent in G (and u != v)
    # This is the graph in which we seek the Maximum Clique.
    full_mask = (1 << m) - 1
    comp_mask = [0] * m
    for u in range(m):
        # Start with all nodes, remove self and neighbors in G
        comp_mask[u] = full_mask & ~(1 << u) & ~adj_mask[u]

    # 4. Tomita Algorithm for Maximum Clique
    best_clique_mask = 0
    best_size = 0

    # Helper to popcount efficiently (Python 3.8+ has int.bit_count)
    def popcount(x):
        return x.bit_count()

    def expand(clique_mask, cand_mask):
        nonlocal best_clique_mask, best_size

        # Pruning: if current clique size + max possible (cand size) <= best, return
        if popcount(clique_mask) + popcount(cand_mask) <= best_size:
            return

        if cand_mask == 0:
            if popcount(clique_mask) > best_size:
                best_size = popcount(clique_mask)
                best_clique_mask = clique_mask
            return

        # Greedy Coloring of the candidate subgraph in H
        # We assign colors to nodes in cand_mask.
        # The color of a node is an upper bound on the clique size that can be formed
        # from that node and its uncolored neighbors with higher or equal color?
        # Actually, standard Tomita:
        # We order candidates. We assign colors such that no two adjacent nodes in H
        # have the same color.
        # The color assigned to a node v is the smallest integer not used by its
        # already-colored neighbors in the candidate set?
        # No, the bound is: if we branch on node v, the max clique size we can add
        # is bounded by the color of v (if we process in reverse color order).
        
        # Let's implement a simple greedy coloring on the induced subgraph H[cand_mask].
        # We need to iterate through nodes in cand_mask.
        # To make the bound tight, we should process nodes in an order that minimizes
        # colors? Or just any order? Tomita usually uses a specific ordering (e.g. by degree).
        # For simplicity and speed, let's just iterate through set bits.
        
        # We need to store the color for each node in cand_mask.
        # Since m is small, we can use a dictionary or an array.
        # Let's use a dict for colored nodes.
        
        # Optimization: We can pre-compute the list of nodes in cand_mask.
        # But doing this at every recursion level might be expensive.
        # However, m=205 is small.
        
        # Let's extract nodes from cand_mask
        cand_nodes = []
        temp = cand_mask
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            node = lsb.bit_length() - 1
            cand_nodes.append(node)
            temp ^= lsb
        
        # Order by degree in H induced by cand_mask (descending)
        # deg(v) = popcount(comp_mask[v] & cand_mask)
        # This helps in coloring.
        # Note: This sort is O(N log N) per node. With N=205, this is ~2000 ops.
        # Recursion depth is up to 205. Total ops ~ 205 * 205 * log(205) ~ 300k per level?
        # No, the branching factor reduces quickly.
        
        # Let's try without sorting first for speed, or with a simple heuristic.
        # Actually, Tomita's efficiency relies heavily on the coloring bound.
        # A good coloring order is important.
        
        # Let's compute degrees
        degs = [0] * m
        for v in cand_nodes:
            degs[v] = popcount(comp_mask[v] & cand_mask)
            
        # Sort candidates by degree descending
        cand_nodes.sort(key=lambda x: degs[x], reverse=True)

        # Greedy Coloring
        # color[v] = smallest color not used by neighbors of v in H that are in cand_nodes
        # and have already been assigned a color.
        # Wait, the standard algorithm colors the graph.
        # If u and v are adjacent in H, they must have different colors.
        # We assign colors sequentially to cand_nodes.
        
        colors = {}
        # We need to check neighbors in H that are in cand_nodes.
        # To speed up, we can keep track of which colors are used by neighbors.
        
        # Simple approach:
        for v in cand_nodes:
            # Find neighbors of v in H that are in cand_nodes
            neigh_mask = comp_mask[v] & cand_mask
            used_colors = set()
            
            # Iterate through neighbors in cand_nodes
            # This is O(N) per node, so O(N^2) per level.
            # N=205, N^2 = 40k. This might be slow if deep.
            # But pruning should cut depth significantly.
            
            temp_neigh = neigh_mask
            while temp_neigh:
                lsb = temp_neigh & -temp_neigh
                u = lsb.bit_length() - 1
                temp_neigh ^= lsb
                if u in colors:
                    used_colors.add(colors[u])
            
            # Assign smallest available color
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c

        # Now, branch on nodes in reverse order of their color?
        # Actually, Tomita branches on nodes in the order they were colored?
        # No, it branches on nodes such that the bound is tight.
        # The bound for a node v is: current_clique_size + colors[v].
        # We should try nodes with higher colors first? Or lower?
        # Standard Tomita:
        # The candidates are partitioned into color classes.
        # We process color classes from highest to lowest?
        # Actually, we iterate through the nodes. For each node v,
        # if current_size + colors[v] <= best_size, we can prune the rest?
        
        # Let's iterate through cand_nodes.
        # To make the bound work, we should process nodes in decreasing order of color?
        # If we process v, and current_size + colors[v] <= best_size, then for any subsequent
        # node w with colors[w] <= colors[v], we can prune?
        # This requires the nodes to be sorted by color descending.
        
        # Let's sort cand_nodes by color descending
        cand_nodes.sort(key=lambda x: colors[x], reverse=True)

        for v in cand_nodes:
            # Pruning
            if popcount(clique_mask) + colors[v] <= best_size:
                break
            
            # Branch on v
            new_clique_mask = clique_mask | (1 << v)
            # New candidates are neighbors of v in H that are in the current cand_mask
            # AND have not been processed yet?
            # In Tomita, the new candidate set is (cand_mask & comp_mask[v]).
            # But we must ensure we don't include nodes that were "before" v in the ordering?
            # Actually, the standard recursive step is:
            # expand(clique + v, cand & N(v))
            # Where cand is the set of available vertices.
            # Since we are using a list of nodes and iterating, we need to be careful.
            
            # If we use the set-based approach:
            # new_cand_mask = cand_mask & comp_mask[v]
            # But this includes nodes that might have been "skipped" or "before" v in the list?
            # The coloring bound assumes that we are building a clique from the set of candidates.
            # If we pick v, the new candidates must be adjacent to v in H.
            # So new_cand_mask = cand_mask & comp_mask[v].
            
            # However, to maintain the pruning property based on the specific coloring,
            # we usually restrict the new candidates to those that were "after" v in the
            # specific ordering used for coloring?
            # No, the coloring is on the whole candidate set.
            # The bound colors[v] is valid for the clique size in the subgraph induced by
            # {v} U (cand_nodes that are adjacent to v and have color <= colors[v]?)
            
            # Let's stick to the standard recursive definition which is correct:
            # The new candidate set is the intersection of the current candidate set
            # and the neighbors of v in H.
            
            new_cand_mask = cand_mask & comp_mask[v]
            
            expand(new_clique_mask, new_cand_mask)
            
            # Remove v from cand_mask for subsequent iterations?
            # In the loop above, we are iterating over a static list `cand_nodes`.
            # But `cand_mask` is the set of all candidates available at this level.
            # When we branch on v, we explore that branch.
            # Then we move to the next node in the list.
            # The next node w should be considered as a candidate to form a clique
            # WITHOUT v.
            # So the `cand_mask` for the next iteration of the loop should ideally
            # exclude v?
            # Actually, the standard algorithm maintains a set of candidates.
            # When we pick v, we recurse with N(v) intersect Candidates.
            # Then we remove v from Candidates and continue with the next node.
            
            # In our bitmask implementation, `cand_mask` is passed by value (it's an int).
            # So modifying it in the loop affects the local variable.
            # We should remove v from cand_mask after processing?
            # Yes, because we are exploring the branch where v is NOT in the clique
            # (for the next iteration of the loop, we are effectively saying "what if we don't pick v?").
            # Wait, the loop iterates over possible vertices to ADD to the clique.
            # If we pick v, we recurse.
            # If we don't pick v, we continue to the next vertex.
            # So for the next vertex, the available candidates should be the original
            # candidates minus v.
            
            cand_mask &= ~(1 << v)

    # Initial call
    # Candidates are all nodes 0..204
    init_cand_mask = full_mask
    expand(0, init_cand_mask)

    # Extract the best clique
    result = []
    temp = best_clique_mask
    while temp:
        lsb = temp & -temp
        node = lsb.bit_length() - 1
        result.append(node)
        temp ^= lsb

    # If no clique found (should not happen, at least size 1), return [0]
    if not result:
        result = [0]

    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
