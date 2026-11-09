import time
import random
from math import gcd

def construct():
    """
    Finds the maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    Uses branch-and-bound maximum independent set search on the conflict graph.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    max_modulus = 128
    moduli_list = list(range(2, max_modulus + 1))
    n_moduli = len(moduli_list)
    
    # Precompute gcds for all pairs of moduli
    gcd_cache = {}
    for i in range(n_moduli):
        m1 = moduli_list[i]
        for j in range(i, n_moduli):
            m2 = moduli_list[j]
            if m1 == m2:
                continue
            g = gcd(m1, m2)
            gcd_cache[(m1, m2)] = g
            gcd_cache[(m2, m1)] = g

    def get_gcd(m1, m2):
        return gcd_cache.get((m1, m2), 1)
    
    # Build conflict graph: nodes are (modulus_idx, residue) pairs
    # Two nodes conflict if they are in the same modulus OR their residue classes intersect
    
    # For each modulus, list all valid residues
    all_residues = {}
    for i, m in enumerate(moduli_list):
        all_residues[i] = list(range(m))
    
    # Total number of nodes
    total_nodes = sum(len(all_residues[i]) for i in range(n_moduli))
    
    # Map each node to an index
    node_to_idx = {}
    idx_to_node = []
    idx = 0
    for i in range(n_moduli):
        for r in all_residues[i]:
            node_to_idx[(i, r)] = idx
            idx_to_node.append((i, r))
            idx += 1
    
    # Build adjacency lists using bitsets (Python ints as bitsets)
    # adj[i] is a bitset of nodes that conflict with node i
    adj = [0] * total_nodes
    
    # Precompute: for each pair of moduli (i, j), which residues conflict
    # Residue r_i in modulus m_i conflicts with residue r_j in modulus m_j
    # if r_i % gcd(m_i, m_j) == r_j % gcd(m_i, m_j)
    
    conflict_cache = {}
    
    def get_conflict_mask(i, r_i, j):
        """Get bitset of residues in modulus j that conflict with (i, r_i)"""
        if (i, j) in conflict_cache:
            return conflict_cache[(i, j)]
        
        m_i = moduli_list[i]
        m_j = moduli_list[j]
        g = get_gcd(m_i, m_j)
        target = r_i % g
        
        # Build bitset for modulus j
        mask = 0
        for r_j in range(m_j):
            if r_j % g == target:
                # Find the global index for (j, r_j)
                node_idx = node_to_idx[(j, r_j)]
                mask |= (1 << node_idx)
        
        conflict_cache[(i, j)] = mask
        return mask
    
    # Build adjacency for all pairs
    for i in range(n_moduli):
        m_i = moduli_list[i]
        for r_i in all_residues[i]:
            node_i = node_to_idx[(i, r_i)]
            
            # Same modulus: all other residues conflict
            for r_i2 in all_residues[i]:
                if r_i2 != r_i:
                    node_i2 = node_to_idx[(i, r_i2)]
                    adj[node_i] |= (1 << node_i2)
            
            # Different moduli: check conflicts
            for j in range(n_moduli):
                if i == j:
                    continue
                m_j = moduli_list[j]
                g = get_gcd(m_i, m_j)
                target = r_i % g
                
                for r_j in all_residues[j]:
                    if r_j % g == target:
                        node_j = node_to_idx[(j, r_j)]
                        adj[node_i] |= (1 << node_j)
    
    # Now we have a conflict graph. We want to find the maximum independent set.
    # An independent set is a set of nodes with no edges between them.
    # Each node in the independent set corresponds to a (residue, modulus) pair.
    # Since we can only pick one residue per modulus, and we need disjoint classes,
    # this is exactly what we want.
    
    # Branch and bound for maximum independent set
    best_size = 0
    best_solution = []
    
    # For upper bound, use greedy coloring on the candidate set
    def greedy_color_bound(candidates_bitset):
        """
        Upper bound on independent set size using greedy coloring.
        candidates_bitset: bitset of candidate nodes
        Returns: upper bound on size
        """
        if candidates_bitset == 0:
            return 0
        
        # Greedy coloring: assign colors to nodes such that adjacent nodes have different colors
        # The number of colors is an upper bound on the independent set size
        uncolored = candidates_bitset
        color_count = 0
        
        while uncolored != 0:
            color_count += 1
            # Find a maximal independent set in uncolored
            # Greedily pick nodes
            current_color = 0
            # Pick a node with fewest uncolored neighbors (MRV heuristic)
            best_node = -1
            min_neighbors = float('inf')
            
            # Iterate through uncolored nodes
            temp = uncolored
            while temp != 0:
                # Get lowest set bit
                lsb = temp & (-temp)
                node = lsb.bit_length() - 1
                temp ^= lsb
                
                # Count uncolored neighbors
                neighbors = adj[node] & uncolored
                neighbor_count = bin(neighbors).count('1')
                
                if neighbor_count < min_neighbors:
                    min_neighbors = neighbor_count
                    best_node = node
            
            if best_node == -1:
                break
            
            # Remove this node and its neighbors from uncolored
            uncolored &= ~(1 << best_node)
            uncolored &= ~adj[best_node]
        
        return color_count
    
    def branch_and_bound(candidates_bitset, current_set, depth):
        """
        Branch and bound search for maximum independent set.
        candidates_bitset: bitset of nodes that can still be added
        current_set: list of (node_idx, modulus_idx, residue) in current solution
        depth: current depth (for timeout checking)
        """
        nonlocal best_size, best_solution, start_time, deadline
        
        # Check timeout
        if time.time() > deadline - 2.0:
            return
        
        # Update best if current is better
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_solution = current_set.copy()
        
        # If no candidates, return
        if candidates_bitset == 0:
            return
        
        # Upper bound: current size + greedy coloring bound
        bound = len(current_set) + greedy_color_bound(candidates_bitset)
        if bound <= best_size:
            return
        
        # Choose a node to branch on (MRV: fewest candidates that conflict with it)
        # Actually, for independent set, we should pick a node that maximizes pruning
        # Use degree heuristic: pick node with highest degree in candidate subgraph
        
        best_node = -1
        max_degree = -1
        
        temp = candidates_bitset
        while temp != 0:
            lsb = temp & (-temp)
            node = lsb.bit_length() - 1
            temp ^= lsb
            
            # Count neighbors in candidate set
            neighbors_in_candidates = adj[node] & candidates_bitset
            degree = bin(neighbors_in_candidates).count('1')
            
            if degree > max_degree:
                max_degree = degree
                best_node = node
        
        if best_node == -1:
            return
        
        # Branch 1: Include best_node
        # Remove best_node and all its neighbors from candidates
        new_candidates = candidates_bitset & ~(1 << best_node)
        new_candidates &= ~adj[best_node]
        
        current_set.append((best_node, idx_to_node[best_node][0], idx_to_node[best_node][1]))
        branch_and_bound(new_candidates, current_set, depth + 1)
        current_set.pop()
        
        # Branch 2: Exclude best_node
        new_candidates = candidates_bitset & ~(1 << best_node)
        branch_and_bound(new_candidates, current_set, depth + 1)
    
    # Initial candidates: all nodes
    all_candidates = (1 << total_nodes) - 1
    
    # Run branch and bound
    branch_and_bound(all_candidates, [], 0)
    
    # Convert best_solution to required format
    result = []
    for node_idx, mod_idx, residue in best_solution:
        m = moduli_list[mod_idx]
        result.append([residue, m])
    
    # Ensure non-empty
    if not result:
        result = [[0, 2]]
    
    return result

# EVOLVE-BLOCK-END
