import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety

    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i*i <= 2*N:
        squares.add(i*i)
        i += 1
    
    # Identify valid nodes (no self-conflict)
    # If 2*v is a square, v cannot be in the set (diagonal constraint a=b)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2*v not in squares:
            valid_nodes.append(v)
            
    n_vars = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict adjacency list for faster local search
    # adj[i] = set of j != i such that valid_nodes[i] + valid_nodes[j] is a square
    adj = [set() for _ in range(n_vars)]
    conflict_pairs = []
    for i in range(n_vars):
        v = valid_nodes[i]
        for j in range(i + 1, n_vars):
            u = valid_nodes[j]
            if v + u in squares:
                adj[i].add(j)
                adj[j].add(i)
                conflict_pairs.append((i, j))
                
    # If no conflicts, we can take all valid nodes
    if not conflict_pairs:
        return valid_nodes

    # Prepare ILP
    # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
    c = -np.ones(n_vars)
    
    # Constraints: x_i + x_j <= 1 for all conflicts
    num_constraints = len(conflict_pairs)
    A = np.zeros((num_constraints, n_vars))
    b = np.ones(num_constraints)
    
    for k, (i, j) in enumerate(conflict_pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0
        
    # Bounds: x_i in [0, 1]
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(n_vars)
    
    best_solution = None
    
    # Solve ILP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, ub=b),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            # Extract solution
            solution = res.x
            # Since it's binary, values are close to 0 or 1
            selected_indices = [i for i in range(n_vars) if solution[i] > 0.5]
            best_solution = set(selected_indices)
    except Exception as e:
        pass

    # If ILP didn't give a solution, use greedy
    if best_solution is None or len(best_solution) == 0:
        # Greedy: pick nodes with lowest degree first
        degrees = [len(adj[i]) for i in range(n_vars)]
        order = sorted(range(n_vars), key=lambda x: degrees[x])
        best_solution = set()
        for idx in order:
            # Check if idx conflicts with any already selected
            conflict = False
            for nb in adj[idx]:
                if nb in best_solution:
                    conflict = True
                    break
            if not conflict:
                best_solution.add(idx)

    # Local search: try swaps to improve
    # A swap removes one node from the set and adds one not in the set,
    # keeping the set size the same but potentially enabling further additions.
    # Actually, for MIS, a better move is: remove a set of nodes and add a larger set.
    # Simple 1-swap: remove node i, add node j (j not in set). Size stays same.
    # Better: try to add a node by removing its neighbors.
    
    # Let's do: for each node not in set, check if we can add it by removing its neighbors.
    # If number of neighbors in set < 1, we can add it directly (but that's just greedy).
    # If number of neighbors in set == k, we remove those k and add 1, so we lose k-1.
    # Only beneficial if k == 0 (already tried) or if we can chain.
    
    # Alternative: 2-opt style. Remove two nodes, add three.
    # This is complex. Let's try a simpler approach:
    # Iteratively try to add a node. If blocked, try removing one of its neighbors
    # and see if we can then add the target node plus possibly another node.
    
    # Actually, let's try a different local search:
    # For each pair (i, j) where i is in set and j is not in set,
    # try swapping: remove i, add j. Check validity.
    # This keeps size same. Then try to add more.
    
    # Better approach: Start with ILP solution. Try to add nodes.
    # If can't add any, try 2-exchange: remove 2, add 3.
    
    def is_valid_set(idx_set):
        """Check if idx_set is an independent set."""
        for i in idx_set:
            for j in adj[i]:
                if j in idx_set:
                    return False
        return True
    
    # Local search phase
    improved = True
    max_iter = 1000
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        # Try to add a single node
        added = False
        for j in range(n_vars):
            if j in best_solution:
                continue
            # Check if j can be added
            can_add = True
            for nb in adj[j]:
                if nb in best_solution:
                    can_add = False
                    break
            if can_add:
                best_solution.add(j)
                improved = True
                added = True
                break
        
        if added:
            continue
        
        # Try 2-exchange: remove 2 nodes, add 3 nodes
        # For each pair of nodes (i1, i2) in set, try removing them
        # and adding 3 nodes not in set that don't conflict with remaining set
        # and don't conflict with each other.
        
        # This is expensive. Let's limit the search.
        # Try: for each node j not in set, find its neighbors in set.
        # If |N(j) ∩ set| <= 2, try removing those neighbors and adding j
        # plus possibly other nodes.
        
        for j in range(n_vars):
            if j in best_solution:
                continue
            
            # Get neighbors of j that are in the set
            neighbors_in_set = [nb for nb in adj[j] if nb in best_solution]
            
            if len(neighbors_in_set) > 2:
                continue
            
            # Try removing neighbors_in_set and adding j
            new_set = best_solution - set(neighbors_in_set)
            new_set.add(j)
            
            # Try to add more nodes to new_set
            for k in range(n_vars):
                if k in new_set:
                    continue
                can_add_k = True
                for nb in adj[k]:
                    if nb in new_set:
                        can_add_k = False
                        break
                if can_add_k:
                    new_set.add(k)
                    improved = True
                    break
            
            if improved:
                best_solution = new_set
                break
        
        if time.time() > deadline - 10:
            break
    
    # Convert back to node values
    selected_nodes = [valid_nodes[i] for i in best_solution]
    
    # Final validation
    if selected_nodes:
        s_set = set(selected_nodes)
        valid = True
        for v in selected_nodes:
            if 2*v in squares:
                valid = False
                break
        if valid:
            for i, v in enumerate(selected_nodes):
                for u in selected_nodes[i+1:]:
                    if v + u in squares:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            return selected_nodes
    
    # Fallback: Simple Greedy MIS
    seed_set = [v for v in valid_nodes if v % 3 == 1]
    
    valid_seed = []
    s_set = set()
    for v in seed_set:
        ok = True
        if 2*v in squares:
            ok = False
        if ok:
            for u in s_set:
                if v + u in squares:
                    ok = False
                    break
        if ok:
            valid_seed.append(v)
            s_set.add(v)
            
    if valid_seed:
        return valid_seed
    else:
        # Absolute fallback: pick first valid node
        return [valid_nodes[0]] if valid_nodes else [1]
