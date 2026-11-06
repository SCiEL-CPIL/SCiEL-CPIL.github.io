"""Program using ILP to find the minimum threshold K for a Hamiltonian cycle."""
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 24
    # Generate the pinned edge list with random.Random(614)
    rng = random.Random(614)
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    rng.shuffle(all_edges)
    
    # Build edge index lookup: edge -> position (1-based)
    edge_pos = {}
    for idx, (u, v) in enumerate(all_edges, 1):
        edge_pos[(u, v)] = idx
        edge_pos[(v, u)] = idx
        
    # Precompute edge indices for quick lookup in ILP
    # We will use a matrix or dict for available edges in G_K
    # To speed up ILP construction, we can pre-define the variable indices
    
    def solve_ilp(K):
        """
        Solve ILP for threshold K.
        Returns the cycle as a list of vertices or None if infeasible.
        """
        # Identify available edges
        available_edges = set()
        for (u, v) in all_edges:
            if edge_pos[(u, v)] <= K:
                available_edges.add((u, v))
                available_edges.add((v, u))
                
        # Quick check: degree
        deg = [0] * n
        for u, v in available_edges:
            deg[u] += 1
            deg[v] += 1
        for i in range(n):
            if deg[i] < 2:
                return None

        # Variables:
        # x_{ij} for i != j. Index: i * n + j
        # u_i for i = 1..n-1. Index: n*n + (i-1)
        num_x = n * n
        num_u = n - 1
        total_vars = num_x + num_u
        
        # Objective: minimize 0
        c = np.zeros(total_vars)
        
        # Constraints
        A_rows = []
        lb = []
        ub = []
        
        # 1. Out-degree constraints
        for i in range(n):
            row = np.zeros(total_vars)
            for j in range(n):
                if i != j:
                    if (i, j) in available_edges:
                        row[i * n + j] = 1
            A_rows.append(row)
            lb.append(1)
            ub.append(1)
            
        # 2. In-degree constraints
        for i in range(n):
            row = np.zeros(total_vars)
            for j in range(n):
                if i != j:
                    if (j, i) in available_edges:
                        row[j * n + i] = 1
            A_rows.append(row)
            lb.append(1)
            ub.append(1)
            
        # 3. MTZ Constraints: u_j - u_i + n * x_{ij} >= 1
        # Rewrite: -u_j + u_i - n * x_{ij} <= -1
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (i, j) not in available_edges:
                    continue # x_{ij} is forced 0, constraint is -u_j + u_i <= -1 which might be violated?
                    # Wait. If x_{ij} is 0, the constraint is u_j >= u_i + 1 - n.
                    # -u_j + u_i - 0 <= -1 => u_i - u_j <= -1 => u_j >= u_i + 1.
                    # This is NOT always true.
                    # We must include the constraint even if edge is unavailable?
                    # If edge is unavailable, x_{ij} = 0.
                    # The constraint becomes u_j - u_i >= 1 - n.
                    # This is a valid constraint on u.
                    # However, if we skip it, we might allow subtours.
                    # But if x_{ij}=0, the edge doesn't exist.
                    # The MTZ constraint prevents subtours by enforcing the order.
                    # If the edge doesn't exist, we don't traverse it.
                    # The constraint $u_j \ge u_i + 1 - n(1-x_{ij})$ with $x_{ij}=0$ becomes $u_j \ge u_i + 1 - n$.
                    # This is a weak constraint.
                    # We should include it to be safe?
                    # Actually, if x_{ij}=0, it doesn't participate in the cycle.
                    # The subtour elimination relies on the edges IN the cycle.
                    # So we only need constraints for edges that COULD be in the cycle?
                    # No, the variables u are global.
                    # If we omit the constraint for an unavailable edge, does it matter?
                    # The constraint links u_i and u_j.
                    # If the edge is not used, the link is broken (relaxed).
                    # So we MUST include the constraint for ALL i,j pairs to properly constrain u?
                    # Or only for available edges?
                    # If we only include for available edges, and a subtour forms using only available edges,
                    # the constraints for those edges will be active.
                    # If a subtour forms, the u values will wrap around, violating the bounds.
                    # So we only need constraints for edges that are in the graph G_K.
                    # Edges not in G_K have x=0.
                    # The constraint for x=0 is $u_j \ge u_i + 1 - n$.
                    # This is always satisfiable given bounds?
                    # $u_j \ge u_i - n + 1$.
                    # Min $u_j = 1$, Max $u_i = n-1$.
                    # $1 \ge n-1 - n + 1 = 0$. True.
                    # So constraints for unavailable edges are redundant (always satisfied).
                    # We can skip them to reduce problem size.
                    continue
                
                row = np.zeros(total_vars)
                # -u_j + u_i - n * x_{ij} <= -1
                # Coeff of x_{ij}: -n
                row[i * n + j] = -n
                
                # Coeff of u_i
                if i > 0:
                    row[num_x + (i - 1)] = 1
                # u_0 is 0, so no variable
                
                # Coeff of u_j
                if j > 0:
                    row[num_x + (j - 1)] = -1
                # u_0 is 0
                
                A_rows.append(row)
                lb.append(-np.inf)
                ub.append(-1)
        
        A = np.array(A_rows)
        
        # Bounds for variables
        bounds = Bounds(0, 1, shape=total_vars) # Default all 0..1
        # Set u variables bounds to 1..n-1
        for i in range(1, n):
            idx = num_x + (i - 1)
            bounds.lower[idx] = 1
            bounds.upper[idx] = n - 1
            
        # Integrality
        integrality = np.ones(total_vars) # All integer
        # x are binary (integers 0..1)
        # u are integers
        
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=bounds)
        
        if res.status != 0: # 0 is optimal
            return None
            
        x = res.x[:num_x].reshape(n, n)
        
        # Extract cycle
        cycle = [0]
        current = 0
        for _ in range(n - 1):
            # Find next node
            next_node = -1
            for j in range(n):
                if j != current and x[current, j] > 0.5:
                    next_node = j
                    break
            if next_node == -1:
                return None
            cycle.append(next_node)
            current = next_node
            
        # Check if cycle closes
        if x[current, 0] < 0.5:
            return None
            
        return cycle

    # Binary search for minimum K
    lo = n
    hi = n * (n - 1) // 2
    
    # Check feasibility at hi
    if solve_ilp(hi) is None:
        return list(range(n))
        
    while lo < hi:
        mid = (lo + hi) // 2
        if solve_ilp(mid) is not None:
            hi = mid
        else:
            lo = mid + 1
            
    K = lo
    cycle = solve_ilp(K)
    
    if cycle is None:
        return list(range(n))
        
    return cycle
