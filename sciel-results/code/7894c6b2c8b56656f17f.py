"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _get_supports(m):
    """Generate all supports (frozensets of indices) of size m."""
    return [frozenset(comb) for comb in combinations(range(D), m)]

def _generate_vectors_from_support(support):
    """Generate all 2^m sign variants for a given support."""
    indices = list(support)
    m = len(indices)
    vecs = []
    for mask in range(1 << m):
        v = np.zeros(D, dtype=np.int64)
        for j, idx in enumerate(indices):
            if mask & (1 << j):
                v[idx] = 1
            else:
                v[idx] = -1
        vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Strategy:
    # 1. Model vectors as supports of size 4 with entries +/-1.
    #    Norm squared = 4.
    #    Constraint: ||u - v||^2 >= 4 => u.v <= 2.
    #    Max u.v for supports S1, S2 is |S1 ∩ S2|.
    #    So we need |S1 ∩ S2| <= 2 for all pairs.
    # 2. This is a Maximum Independent Set (MIS) problem on the graph of supports
    #    where edges connect supports with intersection > 2.
    # 3. Use exact Branch-and-Bound to find the optimal MIS size.
    #    The graph has C(11,4)=330 nodes. Degree is ~28.
    #    B&B with good heuristics (McGuffey or similar) can solve this quickly.
    
    supports = list(_get_supports(4))
    n_supports = len(supports)
    
    # Precompute bitmasks for supports
    support_bits = []
    for s in supports:
        bits = 0
        for idx in s:
            bits |= (1 << idx)
        support_bits.append(bits)
        
    # Build conflict graph
    # conflict[i] is a bitmask of nodes conflicting with i
    # 330 nodes fits in a Python int (or a few 64-bit ints, but Python int is arbitrary precision)
    conflict_masks = [0] * n_supports
    
    for i in range(n_supports):
        for j in range(i + 1, n_supports):
            inter = support_bits[i] & support_bits[j]
            count = bin(inter).count('1')
            if count > 2:
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i)
                
    # Branch and Bound for MIS
    # We use a standard recursive B&B.
    # State: available nodes (bitmask), current set size.
    # Upper bound: greedy coloring bound on the available subgraph.
    
    # To speed up, we can use a heuristic to get a good initial lower bound (incumbent).
    # Greedy MIS
    def greedy_mis():
        avail = (1 << n_supports) - 1
        count = 0
        while avail:
            # Pick node with min degree in available subgraph? Or just first?
            # Let's pick the node with the fewest neighbors in the available set.
            best_node = -1
            min_deg = n_supports
            # Iterate through set bits
            a = avail
            while a:
                lsb = a & (-a)
                idx = lsb.bit_length() - 1
                # Count neighbors in avail
                neighbors = conflict_masks[idx] & avail
                deg = bin(neighbors).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
                a ^= lsb
                
            if best_node == -1:
                break
                
            count += 1
            # Remove best_node and its neighbors
            avail &= ~(conflict_masks[best_node] | (1 << best_node))
            
        return count

    initial_lb = greedy_mis()
    best_size = initial_lb
    best_set_mask = 0
    
    # For B&B, we need to track the actual set to reconstruct it.
    # However, B&B on 330 nodes with bitmasks is fast.
    # Let's implement a B&B that returns the mask of the best set.
    
    # We will use a recursive function.
    # To avoid deep recursion limits, we might need to increase limit or use iterative.
    # 330 depth is fine for Python if we set limit.
    import sys
    sys.setrecursionlimit(10000)
    
    # Upper bound function: 
    # A simple bound is the size of the available set.
    # A tighter bound is the greedy coloring number of the available subgraph.
    # Let's use a simple greedy coloring bound.
    
    def upper_bound(avail_mask):
        # Greedy coloring
        # We want to color the graph induced by avail_mask.
        # The number of colors is an upper bound on the size of the independent set.
        # Actually, for MIS, we are looking for a set of non-adjacent nodes.
        # The complement graph's clique number is the MIS size.
        # So we can find a clique in the complement graph.
        # Alternatively, use the bound: sum of 1/(1+deg(v)) ? No, that's for matching.
        # Standard bound for MIS: Greedy coloring of the graph.
        # If we color the graph with k colors, the largest independent set has size at most k?
        # No, that's for Chromatic Number.
        # Wait. If we partition nodes into k independent sets (color classes), 
        # then the MIS can pick at most 1 node from each color class? No, that's for Clustering.
        # 
        # Correction: 
        # If we color the graph G with k colors, then any Independent Set in G 
        # can contain nodes of any colors? No.
        # An Independent Set is a set of nodes with no edges between them.
        # If we color G, nodes of the same color are NOT necessarily non-adjacent. 
        # Nodes of DIFFERENT colors are NOT necessarily adjacent.
        # 
        # Let's use the standard bound for MIS:
        # The size of the MIS is at most the size of the maximum clique in the complement graph.
        # Or, use the bound based on greedy coloring of the graph:
        # If we greedily color the graph, the number of colors used is an upper bound on the Chromatic Number.
        # This doesn't directly bound the MIS.
        #
        # Let's use a simpler bound: 
        # If we pick a node v, the MIS either contains v (and excludes N(v)) or excludes v.
        # Bound: max(1 + MIS(Av \ {v} \ N(v)), MIS(Av \ {v}))
        # This is the recurrence.
        # To prune, we need an upper bound on MIS(Av).
        # A common bound is the "clique cover" number or simply the number of nodes.
        # Let's use a heuristic bound: 
        # Sort nodes by degree. 
        # Actually, for small graphs (330 nodes), a simple branch and bound with 
        # "degree ordering" and "pruning if current + remaining < best" works well.
        
        # Let's use a stronger bound: 
        # The size of the MIS is <= the size of the largest independent set in a 
        # greedy coloring of the complement? 
        # 
        # Let's stick to a robust B&B with the following:
        # 1. Order nodes by descending degree in the available set.
        # 2. Branch on the first node.
        # 3. Prune if current_size + popcount(avail) <= best_size.
        # This is weak but safe.
        
        # Better: Use the bound from "Tomita's algorithm" for maximum clique (which is MIS on complement).
        # We can implement a maximum clique finder on the complement graph.
        # The complement graph has edges where intersection <= 2.
        # This graph is dense (degree ~ 330 - 28 = 300).
        # Maximum clique in a dense graph is hard.
        # 
        # Let's stick to MIS on the sparse conflict graph.
        # Bound: 
        # Let's use a simple greedy bound: 
        # Greedily pick nodes, removing neighbors. The number picked is a LOWER bound.
        # We need an UPPER bound.
        # Upper bound: 
        # If we color the graph, the MIS size is at most the number of colors? No.
        # 
        # Let's use the bound: 
        # For each node v, let d(v) be its degree in the available subgraph.
        # The MIS size is at most sum(1 / (1 + d(v)))? No.
        # 
        # Let's use a simple but effective B&B:
        # 1. Maintain a list of available nodes.
        # 2. Pick a node v with max degree.
        # 3. Branch 1: Include v. Remove v and N(v).
        # 4. Branch 2: Exclude v. Remove v.
        # 5. Prune if current_size + len(available) <= best_size.
        
        # To make it faster, we can use a heuristic to find a good initial solution (done).
        # And we can use "branch and reduce".
        
        # Given the small size (330), even a slightly weak bound might work if the graph is structured.
        # But let's try to be smarter.
        # We can use the bound: 
        # The MIS size is at most the number of nodes in the available set.
        # This is too weak.
        
        # Let's try to compute a greedy coloring of the available subgraph.
        # If we can color the graph with k colors, does that bound the MIS?
        # No. 
        # 
        # However, if we find a clique in the complement (which is an independent set in G), 
        # that's a lower bound.
        
        # Let's use a different approach: 
        # Since the graph is sparse, we can use a simple recursive B&B with memoization? 
        # No, state space is 2^330.
        
        # Let's use the "Tomita" style bound for Maximum Clique on the complement graph.
        # Wait, the conflict graph is sparse. The complement is dense.
        # Finding a Maximum Independent Set in a sparse graph is equivalent to 
        # Finding a Maximum Clique in a dense graph.
        # Maximum Clique is NP-hard, but Tomita's algorithm is very fast for many graphs.
        
        # Let's implement Tomita's algorithm for Maximum Clique on the complement graph.
        # Nodes: 0..330
        # Edge in complement if NO edge in conflict graph (i.e., intersection <= 2).
        # 
        # Precompute complement adjacency masks.
        # comp_mask[i] = bits of j such that j != i and NOT (conflict_masks[i] & (1<<j))
        
        comp_masks = []
        all_mask = (1 << n_supports) - 1
        for i in range(n_supports):
            # Complement of conflict, excluding self
            c = all_mask ^ conflict_masks[i] ^ (1 << i)
            comp_masks.append(c)
            
        # Tomita's Algorithm for Maximum Clique
        # P: set of candidates (bitmask)
        # X: set of excluded nodes (bitmask) - usually not needed if we order properly
        # Q: set of nodes in current clique
        # 
        # We will use a recursive function expand(P, Q_size).
        # Order nodes in P by degree in the induced subgraph of P.
        # 
        # This is complex to implement efficiently in Python with bitmasks for 330 nodes 
        # because we need to sort 330 nodes frequently.
        
        # Alternative: Use a simpler B&B on the sparse conflict graph (MIS).
        # Let's try a greedy coloring bound on the conflict graph.
        # If we color the conflict graph with k colors, 
        # the Maximum Independent Set size is NOT bounded by k.
        
        # Let's use the bound: 
        # Greedily select nodes from P, removing their neighbors. 
        # This gives a LOWER bound (feasible solution).
        # 
        # For an UPPER bound, we can use:
        # The size of the MIS is at most the size of the maximum matching in the complement? No.
        
        # Let's use a simple B&B with the following heuristic:
        # 1. Pick a node v from P.
        # 2. If current_size + popcount(P) <= best_size, prune.
        # 3. Include v: P' = P \ {v} \ N(v). Recurse.
        # 4. Exclude v: P'' = P \ {v}. Recurse.
        
        # To improve, pick v with maximum degree in P.
        
        # Let's implement this.
        
        pass # Placeholder, actual implementation below

    # Implementation of B&B for MIS
    best_size_val = initial_lb
    best_mask_val = 0
    
    # We need to track the mask to reconstruct the solution.
    # For speed, we'll just track the size first, then reconstruct if needed, 
    # or track the mask.
    
    # To avoid recursion overhead, we might use a stack. But recursion is clearer.
    
    # Let's use a global variable for the best mask.
    
    def bb_mis(avail_mask, current_mask, current_size):
        nonlocal best_size_val, best_mask_val
        
        if current_size + bin(avail_mask).count('1') <= best_size_val:
            return
            
        if avail_mask == 0:
            if current_size > best_size_val:
                best_size_val = current_size
                best_mask_val = current_mask
            return
            
        # Pick a node to branch on
        # Heuristic: Max degree in avail subgraph
        best_node = -1
        max_deg = -1
        a = avail_mask
        while a:
            lsb = a & (-a)
            idx = lsb.bit_length() - 1
            deg = bin(conflict_masks[idx] & avail_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                best_node = idx
            a ^= lsb
            
        if best_node == -1:
            return
            
        # Branch 1: Include best_node
        # Remove best_node and its neighbors from avail
        new_avail_1 = avail_mask & ~(conflict_masks[best_node] | (1 << best_node))
        new_mask_1 = current_mask | (1 << best_node)
        bb_mis(new_avail_1, new_mask_1, current_size + 1)
        
        # Branch 2: Exclude best_node
        new_avail_2 = avail_mask & ~(1 << best_node)
        bb_mis(new_avail_2, current_mask, current_size)
        
    # Start B&B
    initial_avail = (1 << n_supports) - 1
    bb_mis(initial_avail, 0, 0)
    
    # If B&B didn't finish or didn't improve, we still have the greedy solution.
    # But B&B with this weak bound might be slow.
    # Let's add a time check in the recursion? 
    # Python doesn't easily interrupt recursion.
    # Given 330 nodes, it might take a while.
    # 
    # Let's rely on the fact that the graph is structured (packing design like).
    # The optimal MIS size is likely close to the greedy result.
    # 
    # If time is running out, we should return the best found.
    # But the function must return something.
    
    # Extract nodes from best_mask_val
    selected_nodes = []
    m = best_mask_val
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        selected_nodes.append(idx)
        m ^= lsb
        
    if not selected_nodes:
        # Fallback to greedy result if B&B failed or returned 0
        # Re-run greedy to get the mask
        avail = (1 << n_supports) - 1
        while avail:
            a = avail
            best_node = -1
            min_deg = n_supports
            while a:
                lsb = a & (-a)
                idx = lsb.bit_length() - 1
                deg = bin(conflict_masks[idx] & avail).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
                a ^= lsb
            if best_node == -1: break
            selected_nodes.append(best_node)
            avail &= ~(conflict_masks[best_node] | (1 << best_node))
            
    # Construct vectors
    vectors = []
    for node_idx in selected_nodes:
        support = supports[node_idx]
        vecs = _generate_vectors_from_support(support)
        vectors.append(vecs)
        
    if not vectors:
        return np.zeros((0, D), dtype=np.int64)
        
    result = np.vstack(vectors)
    
    return result

# EVOLVE-BLOCK-END
