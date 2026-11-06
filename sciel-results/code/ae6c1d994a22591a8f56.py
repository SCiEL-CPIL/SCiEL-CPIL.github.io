"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import lcm
    from fractions import Fraction
    from itertools import combinations
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    start_time = time.time()
    deadline = start_time + 85.0

    # Nodes are integers 2..256
    nodes = list(range(2, 257))
    N = len(nodes)
    
    # Weights are 1/x
    weights = [Fraction(1, x) for x in nodes]
    
    # Build the complement graph adjacency: 
    # Edge (i, j) in complement graph exists if lcm(nodes[i], nodes[j]) <= 256.
    # We want an Independent Set in this complement graph.
    # This is equivalent to a Clique in the original graph (lcm > 256).
    
    # List of edges in the complement graph (conflict edges)
    conflict_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if lcm(nodes[i], nodes[j]) <= 256:
                conflict_edges.append((i, j))
                
    # If there are no conflict edges, we can pick all nodes.
    # But usually there are many.
    
    # Formulate MILP
    # Variables: x_i for i in 0..N-1
    # Maximize sum(w_i * x_i)
    # Subject to:
    #   x_i + x_j <= 1 for all (i,j) in conflict_edges
    #   x_i in {0, 1}
    
    # Objective coefficients (to be maximized, so we negate for minimization if needed, 
    # but scipy.optimize.milp minimizes by default. 
    # So we minimize -sum(w_i * x_i).
    # Note: weights are Fractions. MILP requires numeric types. 
    # Using float might lose precision for the "exact" requirement? 
    # The task says "Maximize exact reciprocal sum". 
    # However, the evaluator measures the raw value. 
    # If we use floats, we might pick a suboptimal set if two sets have very close sums.
    # Given N=255 and weights 1/x, the sums are distinct rationals.
    # The difference between any two distinct sums of reciprocals of integers <= 255 is at least 1/(255!)? No, much larger.
    # The minimum difference between two distinct subsets sums is likely > 1e-15.
    # Float64 has ~15-17 decimal digits of precision.
    # Sum of 255 terms of ~1/2 to 1/256 is ~2.
    # Precision should be sufficient to distinguish the optimal set.
    
    # To be safer, we can scale weights to integers? 
    # LCM(1..255) is huge. 
    # Let's stick to floats for the solver, but verify the final set.
    
    c = np.array([-float(w) for w in weights])
    
    # Constraints matrix A for A @ x <= b
    # Each edge (i, j) gives a row: x_i + x_j <= 1
    A = np.zeros((len(conflict_edges), N))
    b = np.ones(len(conflict_edges))
    
    for k, (i, j) in enumerate(conflict_edges):
        A[k, i] = 1.0
        A[k, j] = 1.0
        
    # Bounds: x_i in [0, 1]
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer (binary)
    integrality = np.ones(N)
    
    # 1. Greedy Initialization
    # Sort nodes by weight descending (i.e., value ascending)
    # We want to pick small numbers first because 1/x is larger.
    # But we must respect constraints.
    # Greedy strategy: Iterate through nodes sorted by value ascending.
    # If a node doesn't conflict with any already selected node, add it.
    
    # Precompute conflict masks using bitsets for O(1) compatibility checks
    # mask[i] is an integer where bit j is set if node i conflicts with node j
    masks = [0] * N
    for i in range(N):
        m = 0
        for j in range(N):
            if i != j and (conflict_edges and any((e[0]==i and e[1]==j) or (e[0]==j and e[1]==i) for e in conflict_edges)):
                 # This inner loop is slow. Let's build masks more efficiently.
                 pass
        
    # Rebuild masks efficiently
    # Since we already have conflict_edges, we can use them.
    # But for N=255, N^2 is ~65k, which is fine.
    for i in range(N):
        m = 0
        for j in range(N):
            if i != j:
                if lcm(nodes[i], nodes[j]) <= 256:
                    m |= (1 << j)
        masks[i] = m

    selected_indices = []
    selected_mask = 0
    # Sort indices by node value (ascending) to prioritize higher weights
    sorted_indices = sorted(range(N), key=lambda i: nodes[i])
    
    for i in sorted_indices:
        # Check if i conflicts with any selected node
        # If (masks[i] & selected_mask) == 0, no conflict
        if (masks[i] & selected_mask) == 0:
            selected_indices.append(i)
            selected_mask |= (1 << i)

    # Calculate objective value of greedy solution
    greedy_value = sum(weights[i] for i in selected_indices)
    
    # Prepare x0 for MILP
    x0 = np.zeros(N)
    for i in selected_indices:
        x0[i] = 1.0

    # 2. Solve MILP with x0
    # Check time
    if time.time() > deadline - 5:
        return [nodes[i] for i in selected_indices]

    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, ub=b),
            integrality=integrality,
            bounds=bounds,
            x0=x0
        )
        
        if res.success:
            # Extract solution
            x_sol = res.x
            sol_indices = [i for i in range(N) if x_sol[i] > 0.5]
            
            # Verify validity of solver solution
            valid = True
            sol_mask = 0
            for i in sol_indices:
                if (masks[i] & sol_mask) != 0:
                    valid = False
                    break
                sol_mask |= (1 << i)
            
            if valid and sol_indices:
                # Check if solver solution is better than greedy
                sol_value = sum(weights[i] for i in sol_indices)
                if sol_value > greedy_value:
                    return [nodes[i] for i in sol_indices]
                else:
                    return [nodes[i] for i in selected_indices]
            else:
                # Solver returned invalid or empty, use greedy
                return [nodes[i] for i in selected_indices]
        else:
            # Solver failed, use greedy
            return [nodes[i] for i in selected_indices]
    except Exception:
        # Any error, use greedy
        return [nodes[i] for i in selected_indices]

    # Fallback if we somehow reach here
    return [nodes[i] for i in selected_indices]
# EVOLVE-BLOCK-END
