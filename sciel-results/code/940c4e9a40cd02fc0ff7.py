import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for final checks and return
    
    N = 256
    max_sum = 2 * N
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # 1. Filter valid nodes (diagonal constraint: 2*v must not be a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    M = len(valid_nodes)
    if M == 0:
        return [1]
        
    # 2. Build Conflict Graph using Bitmasks
    # adj[i] is a bitmask of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    adj = [0] * M
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 3. Greedy Heuristic to find an initial Independent Set (Lower Bound)
    # Strategy: Repeatedly pick the node with the fewest conflicts in the remaining set
    def greedy_is():
        remaining_mask = (1 << M) - 1
        is_set = []
        while remaining_mask:
            # Find node with min degree in remaining subgraph
            best_node = -1
            min_deg = M + 1
            # Iterate through bits in remaining_mask
            m = remaining_mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                # Count neighbors in remaining_mask
                deg = (adj[idx] & remaining_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
                m ^= lsb
                
            if best_node == -1:
                break
                
            is_set.append(best_node)
            # Remove best_node and its neighbors from remaining
            remaining_mask &= ~(adj[best_node] | (1 << best_node))
            
        return is_set

    initial_is = greedy_is()
    best_size = len(initial_is)
    best_set = initial_is[:]

    # 4. Branch and Bound with Bitmasks
    # We use a static ordering for vertices to guide the search
    # Order by initial degree descending (high degree nodes are more constraining)
    degrees = [adj[i].bit_count() for i in range(M)]
    order = sorted(range(M), key=lambda x: degrees[x], reverse=True)
    
    # Precompute bit for each index
    bits = [1 << i for i in range(M)]
    
    # Greedy Coloring Bound Function
    # Returns an upper bound on the MIS size of the subgraph induced by 'candidates'
    def color_bound(candidates):
        if candidates == 0:
            return 0
        
        # Simple greedy coloring to get an upper bound
        # We assign colors to vertices in 'candidates'
        # The max color used is an upper bound on the size of the maximum independent set?
        # No, the size of the maximum independent set is at most the chromatic number.
        # Chromatic number >= clique number.
        # Actually, for MIS, the upper bound is the number of colors in a proper coloring of the *conflict* graph?
        # Yes. If we color the conflict graph with k colors, any independent set can contain at most one vertex of each color?
        # No, an independent set contains no edges. In a coloring, adjacent vertices have different colors.
        # An independent set can contain multiple vertices of the same color (since they are not adjacent).
        # So coloring does NOT directly bound MIS size.
        
        # Correction: 
        # The size of the Maximum Independent Set (MIS) is equal to the size of the Maximum Clique (MaxClique) in the *complement* graph.
        # Tomita's algorithm finds MaxClique.
        # Alternatively, for MIS directly:
        # Upper bound: The number of vertices in 'candidates' is a trivial bound.
        # A tighter bound: The size of a *clique cover* of the conflict graph? 
        # If we cover the vertices of the conflict graph with k cliques, the MIS can pick at most 1 vertex from each clique.
        # So, MIS size <= number of cliques in a clique cover.
        # Greedy clique cover is equivalent to greedy coloring of the *complement* graph.
        
        # Let's implement a simple bound: 
        # Bound = number of vertices in candidates. (Too loose)
        # Let's use a heuristic bound: 
        # Pick a vertex v. MIS(candidates) <= 1 + MIS(candidates \ N[v])
        # This is the recursion itself.
        
        # For a fast bound in Python, we can use:
        # Bound = sum(1 / (deg(u) + 1) for u in candidates)? No.
        
        # Let's rely on the recursion with the trivial bound `popcount` but order the branching to prune fast.
        # To improve pruning, we can use a "lookahead" or a simple greedy MIS size on the candidates as a *lower* bound for the *subproblem*?
        # If we can find a greedy MIS of size G in candidates, then MIS(candidates) >= G.
        # This doesn't help prune if current + G <= best? No, we need an UPPER bound.
        
        # Let's use a simple upper bound: 
        # If the graph is sparse, popcount is okay initially.
        # We can also use: MIS <= floor(popcount / 2) if min degree >= 1? No.
        
        # Let's just use popcount for now, but rely on the strong initial lower bound and good branching order.
        return candidates.bit_count()

    # Recursive B&B
    # To avoid deep recursion issues and speed up, we use an iterative stack or careful recursion.
    # Python recursion limit is usually 1000. M=245, so depth is fine.
    
    def search(candidates, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
            
        # Bound
        upper_bound = current_size + color_bound(candidates)
        if upper_bound <= best_size:
            return
            
        # Select a vertex to branch on
        # Heuristic: Pick the vertex with the highest degree in the 'candidates' subgraph
        # This maximizes the reduction in the search space when including it.
        
        # Find vertex with max degree in candidates
        max_deg = -1
        best_v = -1
        m = candidates
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = (adj[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = idx
            m ^= lsb
            
        if best_v == -1:
            return
            
        v_bit = bits[best_v]
        
        # Branch 1: Include best_v
        # New candidates: remove best_v and its neighbors
        new_candidates_include = candidates & ~(adj[best_v] | v_bit)
        current_set.append(best_v)
        search(new_candidates_include, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        # New candidates: remove best_v
        new_candidates_exclude = candidates & ~v_bit
        search(new_candidates_exclude, current_size, current_set)

    # Start search
    # We can prune the initial search space if we already have a good solution
    # But we need to explore all possibilities to ensure optimality.
    
    # Optimization: If the initial greedy solution is very large, we can skip branches that can't beat it.
    # The search function handles this.
    
    # Initial candidates is all nodes
    initial_candidates = (1 << M) - 1
    search(initial_candidates, 0, [])
    
    # 5. Final Verification and Fallback
    # Verify the best_set found
    result_values = [valid_nodes[i] for i in best_set]
    
    # Check validity
    is_valid = True
    res_set = set(result_values)
    for a in res_set:
        if 2 * a in squares:
            is_valid = False
            break
        for b in res_set:
            if a < b and (a + b) in squares:
                is_valid = False
                break
        if not is_valid:
            break
            
    if not is_valid:
        # Fallback to the greedy solution which is always valid
        result_values = [valid_nodes[i] for i in initial_is]
        
    return result_values
