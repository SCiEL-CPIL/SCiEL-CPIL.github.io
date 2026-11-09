import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n = 16
    num_edges_total = n * (n - 1) // 2  # 120
    
    # Map edge (u, v) to an index 0..119
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    # Precompute edge indices for all pairs
    edge_idx_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = edge_index(u, v)
            
    # Inverse map: index -> (u, v)
    inv_edge_map = [None] * num_edges_total
    for (u, v), idx in edge_idx_map.items():
        inv_edge_map[idx] = (u, v)

    # Precompute local Q3 masks on 8 vertices (0..7)
    # Vertices 0..7
    # Q3 edges in abstract space (using standard labeling 000..111)
    # 0:000, 1:001, 2:010, 3:011, 4:100, 5:101, 6:110, 7:111
    q3_abstract_edges = [
        (0,1), (0,2), (0,4),
        (1,3), (1,5),
        (2,3), (2,6),
        (3,7),
        (4,5), (4,6),
        (5,7),
        (6,7)
    ]
    
    # We need to find all unique sets of 12 edges in K8 that form Q3.
    # We can iterate over all permutations of 0..7 and generate the edge set.
    # To avoid duplicates, we store them in a set of frozensets of local edge indices.
    
    # Local edge index mapping for K8
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

    local_q3_masks = set()
    
    # To speed up, we can just check all 8! permutations?
    # 40320 is small.
    from itertools import permutations
    for p in permutations(range(8)):
        # p[i] is the vertex in the subset corresponding to abstract node i
        # We need to check if the edges of Q3 mapped by p form a valid Q3?
        # Actually, any permutation of the vertices of Q3 maps Q3 to a subgraph of K8.
        # The set of edges is determined by the image of the abstract edges.
        # Since Q3 is connected and rigid, different permutations that differ by automorphisms
        # will produce the same edge set.
        
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        
        # There should be exactly 12 edges
        if len(edges_set) == 12:
            local_q3_masks.add(frozenset(edges_set))
            
    # Convert to list of bitmasks (integers) for fast operations
    # 28 bits fit in an int
    local_q3_bitmasks = []
    for mask_set in local_q3_masks:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)
        
    # Generate global Q3 masks
    # For each subset of 8 vertices from 0..15
    # Map local 0..7 to global vertices
    # Translate local edge bitmask to global edge bitmask
    
    global_q3_masks = []
    
    all_subsets = list(combinations(range(n), 8))
    
    for subset in all_subsets:
        # subset is a tuple of 8 global vertex IDs
        # Map local index 0..7 to global vertex
        # local i -> subset[i]
        
        # Precompute local edge to global edge index map for this subset
        # local_edge_idx -> global_edge_idx
        l2g_edge = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_index(u_loc, v_loc)
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g_edge[le] = ge
                
        for local_mask in local_q3_bitmasks:
            global_mask = 0
            # Iterate through bits in local_mask
            # 28 bits, so we can just loop 0..27
            for le in range(28):
                if local_mask & (1 << le):
                    ge = l2g_edge[le]
                    global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)
            
    num_q3s = len(global_q3_masks)
    
    # Simulated Annealing
    # State: current edge mask (120 bits)
    # Energy: number of violated Q3s (Q3s where all 12 edges are present)
    
    # We want to MINIMIZE energy.
    # Ideally energy = 0.
    # We start with a dense graph and try to reduce violations.
    # Or start with a valid graph and try to add edges.
    
    # Strategy: Start with a known valid graph (e.g., from greedy) or a random dense graph?
    # Random dense graph will have MANY violations.
    # Greedy graph has 0 violations.
    # Let's run Greedy first to get a starting point, then SA to improve.
    
    # --- Phase 1: Greedy Set Cover (Fast initialization) ---
    active = [True] * num_q3s
    counts = [0] * num_edges_total
    
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    removed_edges = set()
    current_edge_mask = (1 << num_edges_total) - 1  # All edges present initially
    
    # Greedy Loop
    while True:
        if time.time() > deadline:
            break
            
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if e in removed_edges:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        removed_edges.add(best_edge)
        current_edge_mask &= ~(1 << best_edge)
        
        for q3_idx in q3s_containing_edge[best_edge]:
            if active[q3_idx]:
                active[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb

    # Now current_edge_mask is a valid graph (0 violations)
    # Let's run Simulated Annealing to see if we can add back some edges
    # or find a different valid configuration with more edges.
    
    # State: current_edge_mask
    # Move: Flip a random edge bit.
    # If we add an edge (0->1), energy might increase (more violations).
    # If we remove an edge (1->0), energy might decrease (fewer violations).
    
    # We want to maximize edges subject to Energy = 0.
    # So we can define Cost = Energy * BigPenalty - NumEdges.
    # Or simply: Accept move if NewEnergy < OldEnergy.
    # If NewEnergy == OldEnergy, accept if NewEdges > OldEdges.
    # If NewEnergy > OldEnergy, accept with probability exp(-(DeltaE)/T).
    # But we MUST end with Energy = 0.
    # So we can run SA to minimize Energy, and among min energy states, maximize edges.
    # Or, run SA on the space of valid graphs? That's hard to navigate.
    
    # Better approach:
    # Objective: Minimize (Violations * 1000 - EdgeCount)
    # This heavily prioritizes removing violations, but among valid graphs, maximizes edges.
    
    def get_energy_and_edges(mask):
        # Count violations
        violations = 0
        for q3_mask in global_q3_masks:
            if (mask & q3_mask) == q3_mask:
                violations += 1
        # Count edges
        edges = bin(mask).count('1')
        return violations, edges

    # Initial state
    curr_mask = current_edge_mask
    curr_viol, curr_edges = get_energy_and_edges(curr_mask)
    curr_cost = curr_viol * 1000 - curr_edges
    
    best_mask = curr_mask
    best_cost = curr_cost
    best_edges = curr_edges
    best_viol = curr_viol
    
    # SA Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute list of edges for random selection
    all_edge_indices = list(range(num_edges_total))
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        # Random move: flip a random edge
        e = random.choice(all_edge_indices)
        bit = 1 << e
        new_mask = curr_mask ^ bit
        
        new_viol, new_edges = get_energy_and_edges(new_mask)
        new_cost = new_viol * 1000 - new_edges
        
        delta_cost = new_cost - curr_cost
        
        if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
            curr_mask = new_mask
            curr_viol = new_viol
            curr_edges = new_edges
            curr_cost = new_cost
            
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_mask = curr_mask
                best_edges = curr_edges
                best_viol = curr_viol
                
        T *= cooling_rate
        
    # If best_viol is not 0, we might have an invalid graph.
    # The task requires a valid graph.
    # If SA ends with violations, we must fall back to the greedy result or repair it.
    # Given the penalty of 1000, it's very unlikely to stay in a high-violation state
    # if a valid state is reachable, but let's be safe.
    
    if best_viol > 0:
        # Fall back to greedy result which is known valid
        final_mask = current_edge_mask
    else:
        final_mask = best_mask
        
    # Construct edge list from final_mask
    result_edges = []
    for e in range(num_edges_total):
        if final_mask & (1 << e):
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
