import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s buffer
    
    N = 256
    values = list(range(1, N + 1))
    
    # 1. Identify valid nodes (2*v is not a perfect square)
    def is_square(n):
        r = int(math.isqrt(n))
        return r * r == n
        
    valid_values = []
    for v in values:
        if not is_square(2 * v):
            valid_values.append(v)
            
    M = len(valid_values)
    val_to_idx = {v: i for i, v in enumerate(valid_values)}
    
    # 2. Build neighbor masks
    # neighbor_mask[i] is a bitmask where bit j is set if valid_values[i] + valid_values[j] is a square
    # Note: We only care about edges between valid nodes.
    # Since the graph is undirected, if i+j is square, j+i is square.
    # We exclude self-loops by definition of valid nodes (2v not square).
    
    neighbor_masks = [0] * M
    # Precompute squares up to 2*256 = 512
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    for i in range(M):
        vi = valid_values[i]
        mask = 0
        for j in range(M):
            if i == j:
                continue
            vj = valid_values[j]
            if vi + vj in squares:
                mask |= (1 << j)
        neighbor_masks[i] = mask
        
    # 3. Order nodes by degree (descending) to improve B&B performance
    # We want to branch on high-degree nodes first.
    # However, the bitset indices are fixed. We can reorder the nodes.
    # Let's create an order list.
    degrees = [bin(m).count('1') for m in neighbor_masks]
    order = sorted(range(M), key=lambda x: -degrees[x])
    
    # Remap indices to the new order?
    # Actually, it's easier to just use the original indices but pick the next node to branch on
    # based on the highest degree among the available nodes.
    
    # 4. Branch and Bound
    
    best_set_mask = 0
    best_size = 0
    best_nodes = []
    
    # Initial available mask: all valid nodes
    full_mask = (1 << M) - 1
    
    # Helper to update best
    def update_best(mask, size):
        nonlocal best_set_mask, best_size, best_nodes
        if size > best_size:
            best_size = size
            best_set_mask = mask
            # Decode mask to list of values
            nodes = []
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                nodes.append(valid_values[idx])
                m ^= lsb
            best_nodes = nodes
    
    # Greedy coloring bound for a given mask
    def greedy_color_bound(mask):
        colors = 0
        remaining = mask
        while remaining:
            colors += 1
            # Find a maximal independent set in the subgraph induced by 'remaining'
            # Greedy: pick node with lowest index (or highest degree?)
            # Let's pick the node with the lowest index for determinism and speed
            # Actually, picking a node with high degree might reduce the set faster?
            # Standard greedy coloring: pick vertex, add to color, remove it and neighbors.
            # To maximize the color class size (minimize colors), we should be careful.
            # But this is just an upper bound. Any valid coloring works.
            # A simple greedy:
            color_set = 0
            candidates = remaining
            while candidates:
                # Pick the lowest set bit
                lsb = candidates & -candidates
                idx = lsb.bit_length() - 1
                color_set |= lsb
                # Remove idx and its neighbors from candidates
                # neighbors of idx in the context of 'remaining'
                # We need neighbor_mask[idx] & candidates
                # But wait, neighbor_mask[idx] contains all neighbors.
                # We remove idx and all its neighbors that are in candidates.
                to_remove = lsb | (neighbor_masks[idx] & candidates)
                candidates &= ~to_remove
            remaining &= ~color_set
        return colors

    # B&B Function
    # current_mask: set of nodes included so far
    # avail_mask: set of nodes that can still be included (no conflicts with current_mask)
    # current_size: number of nodes in current_mask
    
    def branch_and_bound(current_mask, avail_mask, current_size):
        nonlocal best_size, best_nodes, best_set_mask
        if time.time() > deadline:
            return
        
        # Pruning: if even taking all available nodes doesn't beat best, stop
        if current_size + avail_mask.bit_count() <= best_size:
            return
            
        # Stronger Pruning: Greedy Color Bound
        # The size of the MIS in the subgraph of avail_mask is <= chromatic number
        # chromatic number >= clique number, but we use greedy coloring which is an upper bound on chromatic number?
        # No, greedy coloring produces a valid coloring with C colors.
        # The chromatic number chi(G) <= C.
        # The size of the Maximum Independent Set alpha(G) is NOT bounded by chi(G) directly in a simple way for upper bound of search.
        # Wait. If we can color the graph with C colors, the graph is partitioned into C independent sets.
        # The MIS can take at most 1 node from each color class? No.
        # Actually, if the graph is C-colorable, it doesn't mean alpha(G) <= C.
        # Example: Complete graph K_n has chi=n and alpha=1.
        # Example: Empty graph has chi=1 and alpha=n.
        # So greedy coloring bound is NOT an upper bound on the size of the Independent Set.
        # It is an upper bound on the size of the *Clique*? No.
        
        # Correction:
        # We are looking for an Independent Set.
        # The standard bound for MIS B&B is:
        # 1. |Available| (trivial)
        # 2. Sum of sizes of Maximal Independent Sets? No.
        # 3. Greedy Coloring is used in Tomita's algorithm for Maximum Clique.
        #    For Maximum Clique, the graph is colored, and the size of the clique is bounded by the number of colors.
        #    For Maximum Independent Set, we can apply the same logic to the *complement* graph.
        #    An Independent Set in G is a Clique in G_bar.
        #    So, we should compute the greedy coloring bound on the COMPLEMENT graph of the available subgraph.
        
        # Let's implement the bound for the complement graph.
        # A node is in the complement neighborhood if it is NOT a neighbor in G.
        # comp_neighbor_mask[i] = avail_mask & ~(neighbor_masks[i] | (1<<i))
        
        # To avoid computing full comp masks, we can do:
        # In the greedy coloring loop for the bound:
        # When picking a node v, we remove v and its NEIGHBORS in the COMPLEMENT graph.
        # Neighbors in complement = All available nodes EXCEPT v and its neighbors in G.
        
        def greedy_color_bound_complement(avail_mask):
            colors = 0
            remaining = avail_mask
            while remaining:
                colors += 1
                color_set = 0
                candidates = remaining
                while candidates:
                    lsb = candidates & -candidates
                    idx = lsb.bit_length() - 1
                    color_set |= lsb
                    # In complement graph, neighbors of idx are:
                    # remaining nodes that are NOT neighbors of idx in G and not idx itself.
                    # So, to_remove = lsb | (remaining & ~(neighbor_masks[idx] | lsb))
                    # Wait, if u is not a neighbor of idx in G, then (u, idx) is an edge in G_bar.
                    # So in G_bar, idx is connected to all u in remaining \ {idx, neighbors_of_idx_in_G}.
                    # We want to remove idx and its G_bar neighbors from candidates.
                    # G_bar neighbors = remaining & ~(neighbor_masks[idx] | lsb)
                    g_bar_neighbors = remaining & ~(neighbor_masks[idx] | lsb)
                    to_remove = lsb | g_bar_neighbors
                    candidates &= ~to_remove
                remaining &= ~color_set
            return colors

        bound = greedy_color_bound_complement(avail_mask)
        if current_size + bound <= best_size:
            return

        # If no available nodes, we are done
        if avail_mask == 0:
            if current_size > best_size:
                update_best(current_mask, current_size)
            return

        # Choose next node to branch on
        # Heuristic: Pick node with highest degree in the available subgraph?
        # Or just the lowest index?
        # Let's pick the node with the highest degree in the induced subgraph of avail_mask.
        # This requires counting bits of (neighbor_masks[i] & avail_mask) for each i in avail_mask.
        # This is O(M) per node. M is ~233.
        
        best_node_idx = -1
        max_deg = -1
        m = avail_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = (neighbor_masks[idx] & avail_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_node_idx = idx
            m ^= lsb
            
        idx = best_node_idx
        bit = 1 << idx
        
        # Branch 1: Include idx
        # New current: current_mask | bit
        # New avail: avail_mask & ~(neighbor_masks[idx] | bit)
        # Note: We remove idx and its neighbors from available.
        new_avail_include = avail_mask & ~(neighbor_masks[idx] | bit)
        branch_and_bound(current_mask | bit, new_avail_include, current_size + 1)
        
        # Branch 2: Exclude idx
        # New current: current_mask
        # New avail: avail_mask & ~bit
        new_avail_exclude = avail_mask & ~bit
        branch_and_bound(current_mask, new_avail_exclude, current_size)

    branch_and_bound(0, full_mask, 0)
    
    if not best_nodes:
        # Fallback: return a single valid node
        return [valid_values[0]] if valid_values else [1]
        
    return best_nodes
