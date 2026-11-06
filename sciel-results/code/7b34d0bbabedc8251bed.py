"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from math import lcm
    from scipy.optimize import milp, LinearConstraint, Bounds
    import scipy.sparse as sp

    start_time = time.time()
    deadline = 50.0  # Safety margin

    nodes = list(range(2, 257))
    n = len(nodes)
    weights = np.array([1.0 / x for x in nodes])
    
    # Build conflict edges
    edges = []
    for i in range(n):
        ai = nodes[i]
        for j in range(i + 1, n):
            aj = nodes[j]
            if lcm(ai, aj) <= 256:
                edges.append((i, j))
                
    m = len(edges)
    
    # Fallback result (Greedy)
    greedy_selected = []
    # Sort by weight descending (smallest numbers first)
    for x in sorted(nodes, key=lambda v: -1.0/v):
        if all(lcm(x, y) > 256 for y in greedy_selected):
            greedy_selected.append(x)
            
    if not greedy_selected:
        greedy_selected = [256]
        
    best_score = sum(1.0/x for x in greedy_selected)
    best_solution = greedy_selected

    # Try MILP
    try:
        if time.time() - start_time > deadline:
            return best_solution
            
        c = -weights
        
        # If no edges, return all
        if m == 0:
            return nodes

        # Use sparse matrix for constraints
        row_indices = []
        col_indices = []
        data = []
        for k, (i, j) in enumerate(edges):
            row_indices.append(k)
            col_indices.append(i)
            data.append(1)
            row_indices.append(k)
            col_indices.append(j)
            data.append(1)
            
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(m, n))
        
        cons = LinearConstraint(A, lb=0, ub=1)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        
        # Run MILP with a timeout
        # HiGHS doesn't have a direct timeout parameter in scipy wrapper easily, 
        # but we can rely on the global deadline.
        # To be safe, we can run in a thread or just trust it's fast.
        # Given n=255, m~32k, it should be fast.
        
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.success:
            x = res.x
            selected_indices = [i for i in range(n) if x[i] > 0.5]
            selected_nodes = [nodes[i] for i in selected_indices]
            
            # Verify
            valid = True
            for i in range(len(selected_nodes)):
                for j in range(i + 1, len(selected_nodes)):
                    if lcm(selected_nodes[i], selected_nodes[j]) <= 256:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and len(selected_nodes) > 0:
                score = sum(1.0/v for v in selected_nodes)
                if score > best_score:
                    best_score = score
                    best_solution = selected_nodes
                
    except Exception:
        pass

    # Local Search Polish
    # Try to add nodes or swap nodes to improve
    current_set = set(best_solution)
    improved = True
    while improved and (time.time() - start_time < deadline - 5.0):
        improved = False
        # Try adding a node
        for node in nodes:
            if node in current_set:
                continue
            # Check if node can be added
            can_add = True
            for y in current_set:
                if lcm(node, y) <= 256:
                    can_add = False
                    break
            if can_add:
                current_set.add(node)
                improved = True
                break # Restart loop to check others
        if improved:
            continue
        
        # Try swapping: remove one, add one to improve score
        current_score = sum(1.0 / x for x in current_set)
        for remove_node in list(current_set):
            for add_node in nodes:
                if add_node in current_set:
                    continue
                # Check if add_node is compatible with all remaining nodes
                can_add = True
                for y in current_set:
                    if y == remove_node:
                        continue
                    if lcm(add_node, y) <= 256:
                        can_add = False
                        break
                if can_add:
                    new_score = current_score - 1.0 / remove_node + 1.0 / add_node
                    if new_score > current_score + 1e-12:
                        current_set.discard(remove_node)
                        current_set.add(add_node)
                        improved = True
                        break
            if improved:
                break
        
        # Try removing a node that allows adding multiple better nodes
        # This is more complex, skip for now
        
    return list(current_set)
# EVOLVE-BLOCK-END
