import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin
    
    # 1. Define Moduli and Weights
    moduli = list(range(20, 201))
    n = len(moduli)
    weights = [1.0 / m for m in moduli]
    
    # 2. Build Non-Coprime Graph
    # adj[i][j] is True if gcd(moduli[i], moduli[j]) > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(moduli[i], moduli[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True

    # Sort nodes by weight descending (small moduli first) to improve B&B pruning
    # We will work with indices sorted by weight
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    # Re-index or just use the sorted list for selection
    # It's easier to keep the original indices but iterate in sorted order
    
    best_clique = []
    best_weight = 0.0
    
    # Global variables for B&B
    # We use a recursive search
    # To avoid recursion depth issues, we can use a stack, but 181 depth is fine for Python if optimized?
    # Actually, max clique size is likely < 100. Recursion is fine.
    
    # Pre-compute neighbor lists for fast access
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    # Convert to lists for faster iteration if needed, but sets are good for intersection
    # For B&B, we maintain a set of candidates
    
    def solve_clique():
        nonlocal best_clique, best_weight
        
        # Initial candidates: all nodes
        # We process nodes in order of sorted_indices
        # But B&B on graph usually picks a node, branches on include/exclude.
        
        # A more efficient approach for Max Weight Clique:
        # Recursive expansion of a clique.
        
        def expand(clique, candidates):
            nonlocal best_clique, best_weight
            
            if time.time() > deadline:
                return
                
            if not candidates:
                if len(clique) > 0:
                    w = sum(weights[i] for i in clique)
                    if w > best_weight:
                        best_weight = w
                        best_clique = clique[:]
                return
            
            # Bound: current weight + sum of weights of all candidates
            # This is a weak bound.
            # Better bound: Greedy coloring on the induced subgraph of candidates?
            # For now, let's try a simple bound and rely on the structure.
            
            current_w = sum(weights[i] for i in clique)
            # Simple upper bound
            upper_bound = current_w + sum(weights[i] for i in candidates)
            if upper_bound <= best_weight:
                return
            
            # Select a node to branch on
            # Heuristic: Pick the node with highest weight in candidates
            # Or the one with most neighbors in candidates?
            # Highest weight is good for weight-based B&B.
            
            # Find max weight node in candidates
            max_w_node = None
            max_w_val = -1
            for i in candidates:
                if weights[i] > max_w_val:
                    max_w_val = weights[i]
                    max_w_node = i
            
            v = max_w_node
            
            # Branch 1: Include v
            # New candidates must be neighbors of v in the current candidates
            new_candidates_include = {i for i in candidates if i != v and adj[v][i]}
            clique.append(v)
            expand(clique, new_candidates_include)
            clique.pop()
            
            # Branch 2: Exclude v
            new_candidates_exclude = {i for i in candidates if i != v}
            expand(clique, new_candidates_exclude)

        # Start with empty clique and all nodes as candidates
        # To speed up, we can order the initial candidates by weight
        # The recursive function picks the max weight node each time, so initial order doesn't matter as much,
        # but the set operations do.
        
        # Optimization: Use a list for candidates and filter?
        # Sets are O(1) for membership, O(N) for creation.
        # With N=181, it's fine.
        
        initial_candidates = set(range(n))
        expand([], initial_candidates)

    # Run the solver
    solve_clique()
    
    # 3. Assign Residues
    # We have best_clique (list of indices)
    # Convert to moduli
    selected_moduli = [moduli[i] for i in best_clique]
    
    # We need to assign residues r_i for each m_i in selected_moduli
    # such that r_i != r_j mod gcd(m_i, m_j)
    
    # Backtracking search for residues
    # Order moduli by size? Or by degree in conflict graph?
    # Let's just try all residues for each modulus in order.
    
    classes = []
    
    def is_disjoint(m, r, existing_classes):
        for m2, r2 in existing_classes:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def assign_residues(moduli_list):
        # Sort moduli to help backtracking? 
        # Moduli with fewer options should be first.
        # Options depend on existing classes.
        # Let's just iterate.
        
        if not moduli_list:
            return []
        
        m = moduli_list[0]
        rest = moduli_list[1:]
        
        # Try all residues for m
        for r in range(m):
            if is_disjoint(m, r, classes):
                classes.append([m, r])
                result = assign_residues(rest)
                if result is not None:
                    return result
                classes.pop()
        return None

    # The backtracking might be slow if the order is bad.
    # We can use a heuristic: pick the modulus with the most constraints (highest gcd with others) first?
    # Or just try.
    
    # To be safe, let's shuffle if it fails? No, deterministic is better for reproducibility.
    # Let's order by modulus size ascending (smaller moduli have fewer residues, but also smaller gcds? No).
    # Actually, smaller moduli have fewer residues to try.
    
    # Let's try assigning in the order of the clique found.
    # If it fails, we might need to remove a node from the clique.
    # But likely it succeeds.
    
    # Optimization: If the clique is large, backtracking might be deep.
    # We can use a greedy assignment with backtracking.
    
    # Let's implement a simple recursive backtracker.
    
    sys.setrecursionlimit(10000)
    
    # Sort selected moduli to potentially help
    # No, let's keep the order from the clique solver.
    
    assigned_classes = assign_residues(selected_moduli)
    
    if assigned_classes is None:
        # Fallback: Just take the first few moduli
        # This shouldn't happen often
        assigned_classes = [[selected_moduli[0], 0]] if selected_moduli else []
        # Try to add more greedily
        for m in selected_moduli[1:]:
            for r in range(m):
                if is_disjoint(m, r, assigned_classes):
                    assigned_classes.append([m, r])
                    break

    return assigned_classes
# EVOLVE-BLOCK-END
