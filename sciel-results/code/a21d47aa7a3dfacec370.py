"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import lcm, gcd
    from fractions import Fraction
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 85.0

    # Nodes are integers 2..256
    nodes = list(range(2, 257))
    N = len(nodes)
    
    # Index mapping
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_node = {i: v for i, v in enumerate(nodes)}
    
    # Weights: 1/x for each node
    weights = [1.0 / x for x in nodes]
    
    # Build conflict graph: edge (i, j) if lcm(nodes[i], nodes[j]) <= 256
    # Use bitsets for fast conflict checking
    # conflict_bits[i] is a bitmask of all nodes j that conflict with i
    
    # First, precompute which pairs conflict
    conflict_pairs = []
    for i in range(N):
        ai = nodes[i]
        for j in range(i + 1, N):
            aj = nodes[j]
            if lcm(ai, aj) <= 256:
                conflict_pairs.append((i, j))
    
    # Build adjacency as sets for quick lookup
    adj = [set() for _ in range(N)]
    for i, j in conflict_pairs:
        adj[i].add(j)
        adj[j].add(i)
    
    # Build bitset representation for fast conflict checks
    # Each node's conflicts as a Python int bitmask
    conflict_bits = [0] * N
    for i in range(N):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        conflict_bits[i] = mask
    
    # Greedy coloring bound for independent set
    # We want to find an upper bound on the weight of any independent set in a subgraph
    # Greedy coloring: color the complement graph (where edges = no conflict)
    # Actually, for independent set in G, we can use greedy coloring on G itself
    # The chromatic number of G gives an upper bound on the size of max independent set
    # But we want weighted version. Let's use a simpler bound:
    # For a set of candidate nodes, sum of weights of nodes in each color class
    # gives an upper bound (since independent set can take at most one per color class... 
    # wait, that's for clique. For independent set, we need the complement.)
    
    # Actually, let's use a different approach:
    # Upper bound = sum of all weights in the candidate set (trivial)
    # Better: use greedy sequential coloring on the conflict graph
    # The independence number <= chromatic number of the complement
    # But for weighted case, let's use: 
    # For each node in order, assign it to the first color class where it has no conflict
    # with any node already in that class. Then the sum of max weight per color class
    # is NOT a valid upper bound for weighted independent set.
    
    # Let me use a simpler but effective bound:
    # For a candidate set S, the upper bound is sum(w_i for i in S)
    # But we can improve by noting that if two nodes conflict, we can't take both.
    # A better bound: greedy matching or using the fact that the graph is sparse.
    
    # For now, let's use a practical approach:
    # Branch and bound with:
    # - Upper bound: sum of weights of all remaining candidates
    # - Ordering: by weight descending (pick high-weight nodes first)
    # - Pruning: if upper bound <= current best, prune
    
    # Actually, let me think about this differently.
    # The key insight is that the conflict graph is very sparse.
    # Most pairs have lcm > 256, so most nodes are compatible.
    # The conflicts are concentrated among small numbers and their multiples.
    
    # Let's identify the "conflict core" - nodes that have many conflicts
    degrees = [len(adj[i]) for i in range(N)]
    
    # Strategy: 
    # 1. Find a good initial solution using greedy
    # 2. Use local search to improve
    # 3. Use B&B to prove optimality or find better
    
    # Step 1: Greedy initial solution
    # Sort by weight descending (i.e., by value ascending)
    order = sorted(range(N), key=lambda i: -weights[i])
    
    selected = []
    selected_set = set()
    for i in order:
        # Check if i conflicts with any selected node
        compatible = True
        for j in selected_set:
            if j in adj[i]:
                compatible = False
                break
        if compatible:
            selected.append(i)
            selected_set.add(i)
    
    current_weight = sum(weights[i] for i in selected)
    best_solution = list(selected)
    best_weight = current_weight
    
    # Step 2: Local search - try to improve by swapping
    def get_conflict_mask(node_set):
        """Get the set of nodes that conflict with any node in node_set"""
        mask = set()
        for i in node_set:
            mask.update(adj[i])
        return mask
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        selected_set = set(best_solution)
        
        # Try adding a node
        for i in range(N):
            if i in selected_set:
                continue
            # Check if i conflicts with any selected node
            if any(j in adj[i] for j in selected_set):
                continue
            # Can add i
            new_weight = best_weight + weights[i]
            if new_weight > best_weight + 1e-15:
                best_solution.append(i)
                best_weight = new_weight
                improved = True
        
        # Try swapping: remove one, add another
        if time.time() < deadline - 5.0:
            best_swap_gain = 0
            best_swap = None
            for i in list(best_solution):
                remaining = set(best_solution) - {i}
                # Find best node j not in best_solution that doesn't conflict with remaining
                for j in range(N):
                    if j in set(best_solution):
                        continue
                    # Check if j conflicts with any node in remaining
                    if any(k in adj[j] for k in remaining):
                        continue
                    gain = weights[j] - weights[i]
                    if gain > best_swap_gain + 1e-15:
                        best_swap_gain = gain
                        best_swap = (i, j)
            
            if best_swap is not None:
                i, j = best_swap
                best_solution.remove(i)
                best_solution.append(j)
                best_weight += best_swap_gain
                improved = True
        
        # Try 2-for-1: remove two, add one
        if time.time() < deadline - 10.0 and len(best_solution) >= 2:
            best_2for1_gain = 0
            best_2for1 = None
            sol_set = set(best_solution)
            for i, j in combinations(best_solution, 2):
                remaining = sol_set - {i, j}
                for k in range(N):
                    if k in sol_set:
                        continue
                    if any(m in adj[k] for m in remaining):
                        continue
                    gain = weights[k] - weights[i] - weights[j]
                    if gain > best_2for1_gain + 1e-15:
                        best_2for1_gain = gain
                        best_2for1 = (i, j, k)
            
            if best_2for1 is not None:
                i, j, k = best_2for1
                best_solution.remove(i)
                best_solution.remove(j)
                best_solution.append(k)
                best_weight += best_2for1_gain
                improved = True
        
        # Try 1-for-2: remove one, add two
        if time.time() < deadline - 10.0:
            best_1for2_gain = 0
            best_1for2 = None
            sol_set = set(best_solution)
            for i in best_solution:
                remaining = sol_set - {i}
                # Find two nodes k, l not in sol_set that don't conflict with remaining or each other
                candidates = [k for k in range(N) if k not in sol_set and not any(m in adj[k] for m in remaining)]
                for k, l in combinations(candidates, 2):
                    if l in adj[k]:
                        continue
                    gain = weights[k] + weights[l] - weights[i]
                    if gain > best_1for2_gain + 1e-15:
                        best_1for2_gain = gain
                        best_1for2 = (i, k, l)
            
            if best_1for2 is not None:
                i, k, l = best_1for2
                best_solution.remove(i)
                best_solution.append(k)
                best_solution.append(l)
                best_weight += best_1for2_gain
                improved = True
    
    # Step 3: Branch and bound for further improvement
    # Use a recursive B&B with greedy upper bound
    
    if time.time() < deadline - 3.0:
        # B&B: explore nodes in order of decreasing weight
        # State: (candidate_set, selected_set, current_weight)
        # Upper bound: current_weight + sum(weights[i] for i in candidate_set)
        # This is a valid upper bound since we can't get more than all remaining weights
        
        # For efficiency, we'll do a limited B&B
        # Only explore if we can potentially beat the current best
        
        # Sort candidates by weight descending
        all_nodes_order = sorted(range(N), key=lambda i: -weights[i])
        
        # Use a stack-based DFS
        # State: (candidates_remaining, selected_so_far, weight_so_far)
        # We'll use a simple approach: try to add nodes in greedy order,
        # and use the sum of remaining weights as upper bound
        
        # Actually, for this problem size, local search should be sufficient.
        # Let's do a quick verification that our solution is valid.
        
        pass  # Local search should have converged
    
    # Verify the solution is valid
    sol_set = set(best_solution)
    for i in best_solution:
        for j in best_solution:
            if i == j:
                continue
            if j in adj[i]:
                # Invalid! This shouldn't happen, but let's fix it
                # Remove one of the conflicting nodes
                if weights[i] < weights[j]:
                    best_solution.remove(i)
                else:
                    best_solution.remove(j)
                break
    
    # Final verification
    final_set = set(best_solution)
    valid = True
    for i in best_solution:
        for j in best_solution:
            if i != j and j in adj[i]:
                valid = False
                break
        if not valid:
            break
    
    if not valid or not best_solution:
        # Fallback to greedy
        selected = []
        selected_set = set()
        for i in sorted(range(N), key=lambda i: -weights[i]):
            if not any(j in adj[i] for j in selected_set):
                selected.append(i)
                selected_set.add(i)
        best_solution = selected
    
    # Convert to node values
    result = [nodes[i] for i in best_solution]
    
    if not result:
        result = [2]
    
    return result
# EVOLVE-BLOCK-END
