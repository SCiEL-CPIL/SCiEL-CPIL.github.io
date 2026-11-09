"""Rewritten program using weighted Bron-Kerbosch + greedy coloring for maximum weight clique."""
# EVOLVE-BLOCK-START
import time
from math import gcd

_DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Build compatibility: edge (a,b) exists iff lcm(a,b) > 256
    # i.e., a * b // gcd(a,b) > 256
    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)
    
    # Precompute conflicts: conflict[i][j] = True if lcm(vi, vj) <= 256 (incompatible)
    # We want cliques in the compatibility graph, so we work with the conflict graph's complement
    # Actually, let's define: adj[i] = set of j != i such that lcm(vi,vj) > 256 (compatible)
    
    # For efficiency, precompute lcm compatibility
    # Two numbers a, b are compatible iff a*b/gcd(a,b) > 256
    # i.e., a*b > 256 * gcd(a,b)
    
    # Build adjacency as bitmasks for speed
    # n = 255 vertices
    adj = [0] * n  # bitmask of compatible vertices
    
    for i in range(n):
        a = vertices[i]
        for j in range(i + 1, n):
            b = vertices[j]
            g = gcd(a, b)
            # lcm = a * b // g
            # compatible if a * b // g > 256
            if a * b > 256 * g:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Weights: w[i] = 1/vertices[i]
    weights = [1.0 / vertices[i] for i in range(n)]
    
    # Order vertices by weight descending (smaller numbers first, higher 1/a)
    order = sorted(range(n), key=lambda i: -weights[i])
    
    # Map: new_index -> old_index
    # Reorder so that vertex 0 has highest weight
    new_order = [order[k] for k in range(n)]
    # new_adj[i] corresponds to old vertex new_order[i]
    new_adj = [0] * n
    new_weights = [0.0] * n
    for new_i in range(n):
        old_i = new_order[new_i]
        new_weights[new_i] = weights[old_i]
        # Convert adjacency
        old_mask = adj[old_i]
        new_mask = 0
        for new_j in range(n):
            old_j = new_order[new_j]
            if old_j != old_i and (old_mask >> old_j) & 1:
                new_mask |= (1 << new_j)
        new_adj[new_i] = new_mask
    
    # Bron-Kerbosch with coloring for maximum weight clique
    # We use a branch-and-bound: at each node, compute an upper bound on the
    # remaining weight using greedy coloring on the subgraph of candidates.
    
    best_weight = [0.0]
    best_clique = [0]  # bitmask of best clique
    
    def greedy_color_upper_bound(candidates_mask):
        """
        Greedy coloring to get an upper bound on the sum of weights of any clique
        in the subgraph induced by candidates_mask.
        
        Standard approach: color the graph, then the max clique weight <= sum of
        (max weight per color class) ... actually for weighted case we need to be careful.
        
        For weighted maximum clique with B&B, a common approach:
        - Order vertices by weight descending
        - Assign colors greedily; the color number of vertex v gives an upper bound
          on how many more vertices of weight >= w[v] can be added.
        
        Actually, a simpler valid upper bound: sort candidates by weight desc,
        then for each vertex in that order, if it's compatible with all previously
        "selected" in this bound computation... no, that's just greedy.
        
        Let me use the standard Bron-Kerbosch coloring bound:
        - Greedily color the candidate subgraph
        - The color assigned to vertex v (1-indexed, in the order of coloring) 
          means at most color[v] vertices from v onward can form a clique.
        - For weighted case: the upper bound on additional weight is the sum of
          the top color[v] weights among vertices from v onward? No...
        
        Standard weighted B&B for max weight clique:
        - Process vertices in order of decreasing weight
        - Assign colors greedily (color 1, 2, ...)
        - For vertex at position i with color c, the upper bound on weight of any
          clique containing vertex i and only vertices from i onward is:
          w[i] + sum of the (c-1) largest weights among vertices i+1..end that are
          compatible... this gets complex.
        
        Simpler valid bound: the greedy coloring gives that any clique in the
        subgraph has size at most max_color. For weighted case, a valid upper bound
        on the total weight of any clique in the candidate set is:
        sum of the top max_color weights in the candidate set.
        This is valid because any clique has size <= max_color, and the max possible
        weight for a clique of size k is the sum of the k largest weights.
        """
        if candidates_mask == 0:
            return 0.0
        
        # Get list of candidate vertices
        cands = []
        m = candidates_mask
        while m:
            low = m & (-m)
            i = low.bit_length() - 1
            cands.append(i)
            m ^= low
        
        # Greedy coloring: assign colors so that adjacent vertices get different colors
        # Process in order of decreasing weight
        cands_sorted = sorted(cands, key=lambda i: -new_weights[i])
        
        color_of = {}
        color_classes = []  # list of sets, color_classes[c] = set of vertices with color c+1
        
        for v in cands_sorted:
            # Find the smallest color not used by any neighbor of v in cands
            used = set()
            vm = new_adj[v] & candidates_mask
            mm = vm
            while mm:
                low = mm & (-mm)
                nb = low.bit_length() - 1
                if nb in color_of:
                    used.add(color_of[nb])
                mm ^= low
            
            c = 1
            while c in used:
                c += 1
            color_of[v] = c
            if c > len(color_classes):
                color_classes.append(set())
            color_classes[c].add(v)
        
        max_color = max(color_of.values()) if color_of else 0
        
        # Upper bound: sum of top max_color weights among candidates
        # This is valid because any clique has at most max_color vertices,
        # and the best possible weight for a clique of size <= max_color is
        # the sum of the max_color largest weights.
        cands_by_weight = sorted(cands, key=lambda i: -new_weights[i])
        ub = sum(new_weights[v] for v in cands_by_weight[:max_color])
        return ub
    
    def bron_kerbosch(R_mask, P_mask, best_weight_ref, best_clique_ref):
        """
        R: current clique (bitmask)
        P: candidates that can be added (bitmask)
        """
        if time.time() > _DEADLINE:
            return
        
        if P_mask == 0:
            # Check if R is a valid clique (it should be by construction)
            w = 0.0
            rm = R_mask
            while rm:
                low = rm & (-rm)
                i = low.bit_length() - 1
                w += new_weights[i]
                rm ^= low
            if w > best_weight_ref[0]:
                best_weight_ref[0] = w
                best_clique_ref[0] = R_mask
            return
        
        # Upper bound: current weight + upper bound from P
        current_w = 0.0
        rm = R_mask
        while rm:
            low = rm & (-rm)
            i = low.bit_length() - 1
            current_w += new_weights[i]
            rm ^= low
        
        ub_p = greedy_color_upper_bound(P_mask)
        if current_w + ub_p <= best_weight_ref[0]:
            return
        
        # Process vertices in P in order of decreasing weight
        p_list = []
        pm = P_mask
        while pm:
            low = pm & (-pm)
            i = low.bit_length() - 1
            p_list.append(i)
            pm ^= low
        p_list.sort(key=lambda i: -new_weights[i])
        
        for v in p_list:
            if time.time() > _DEADLINE:
                return
            
            # Check if adding v can still beat best
            # Quick bound: current_w + w[v] + ub of (P & adj[v]) 
            new_R = R_mask | (1 << v)
            new_P = P_mask & new_adj[v] & ((1 << v) - 1)  # only vertices after v in ordering... 
            # Actually we should remove v from P and only keep common neighbors
            # But we need to be careful: new_P should be vertices in P that are
            # compatible with v AND come after v in our processing order to avoid
            # duplicate cliques. Since we process in a fixed order, we restrict to
            # vertices with index > v in p_list order... 
            # Standard BK: new_P = P ∩ N(v), and we remove v from P for subsequent iterations.
            # But to avoid duplicates, we process in order and only consider P vertices
            # that come after v in the processing order.
            
            # Since p_list is sorted, "after v" means vertices in p_list that come after v.
            # Let's build a mask of vertices in P that come after v in p_list order.
            v_idx_in_plist = p_list.index(v)
            after_mask = 0
            for j in range(v_idx_in_plist + 1, len(p_list)):
                after_mask |= (1 << p_list[j])
            
            new_P = P_mask & new_adj[v] & after_mask
            
            # Quick upper bound check
            current_w_new = current_w + new_weights[v]
            ub_new_p = greedy_color_upper_bound(new_P)
            if current_w_new + ub_new_p <= best_weight_ref[0]:
                # Remove v from P for subsequent iterations
                P_mask &= ~(1 << v)
                continue
            
            bron_kerbosch(new_R, new_P, best_weight_ref, best_clique_ref)
            
            # Remove v from P
            P_mask &= ~(1 << v)
            
            # Update bound check for remaining
            # (already handled in recursive calls)
    
    # Initial call: R = empty, P = all vertices
    all_mask = (1 << n) - 1
    bron_kerbosch(0, all_mask, best_weight, best_clique)
    
    # Extract the best clique
    if best_clique[0] == 0:
        # Should not happen, but fallback
        return [vertices[0]]
    
    result = []
    cm = best_clique[0]
    while cm:
        low = cm & (-cm)
        i = low.bit_length() - 1
        result.append(vertices[i])
        cm ^= low
    
    # Verify the result
    from math import lcm as _lcm
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if _lcm(result[i], result[j]) <= 256:
                # Invalid, fallback to greedy
                selected = []
                for x in range(2, 257):
                    if all(_lcm(x, y) > 256 for y in selected):
                        selected.append(x)
                return selected
    
    return result

# EVOLVE-BLOCK-END
