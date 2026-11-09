import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict adjacency list
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # First, get a good initial solution using greedy
    def greedy_mis():
        available = set(range(n))
        chosen = []
        while available:
            # Pick vertex with lowest degree in remaining subgraph
            best_v = min(available, key=lambda v: len(conflicts[v] & available))
            chosen.append(best_v)
            available -= set([best_v])
            available -= conflicts[best_v]
        return chosen
    
    greedy_result = greedy_mis()
    greedy_size = len(greedy_result)
    
    # Now use MILP to find optimal solution
    # Decision variables: x_i in {0, 1} for each vertex i
    # Objective: maximize sum(x_i)
    # Constraints: x_i + x_j <= 1 for each edge (i,j)
    
    # Build constraint matrix
    edges = []
    for i in range(n):
        for j in conflicts[i]:
            if i < j:
                edges.append((i, j))
    
    m = len(edges)
    
    # Create A matrix (sparse for efficiency)
    # For each edge (i, j): x_i + x_j <= 1
    rows = []
    cols = []
    values = []
    
    for e_idx, (i, j) in enumerate(edges):
        rows.append(e_idx)
        cols.append(i)
        values.append(1.0)
        
        rows.append(e_idx)
        cols.append(j)
        values.append(1.0)
    
    # Build sparse matrix
    from scipy.sparse import csr_matrix
    A = csr_matrix((values, (rows, cols)), shape=(m, n))
    
    # Objective: maximize sum(x_i), so c = -1 for all (minimizing -sum)
    c = -np.ones(n)
    
    # Constraints: A @ x <= 1 for all edges
    lb = np.zeros(m)
    ub = np.ones(m)
    
    # Variable bounds: x_i in [0, 1], integer
    integrality = np.ones(n)
    lb_var = np.zeros(n)
    ub_var = np.ones(n)
    
    # Initial solution for warm start
    x0 = np.zeros(n)
    for idx in greedy_result:
        x0[idx] = 1.0
    
    # Time limit in seconds
    time_remaining = deadline - time.time() - 5  # Leave 5s buffer
    if time_remaining < 1:
        # Not enough time for MILP, return greedy result
        return [valid_nodes[i] for i in greedy_result]
    
    # Solve MILP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb=lb, ub=ub),
            integrality=integrality,
            bounds=Bounds(lb=lb_var, ub=ub_var),
            options={"time_limit": max(1, time_remaining), "presolve": True}
        )
        
        if res.status == 0 or res.status == 2:  # Optimal or time limit reached
            if res.x is not None:
                milp_result = [i for i in range(n) if res.x[i] > 0.5]
                if len(milp_result) > greedy_size:
                    greedy_result = milp_result
                    greedy_size = len(milp_result)
        elif res.status == 1:  # Infeasible
            pass
    except Exception:
        # If MILP fails, fall back to greedy
        pass
    
    # Final local search polish: try to add any vertex that doesn't conflict
    if time.time() < deadline - 2:
        current_set = set(greedy_result)
        
        # Try adding vertices (greedy, lowest degree first)
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(n):
                if i not in current_set:
                    # Check if i conflicts with any vertex in current_set
                    if not (conflicts[i] & current_set):
                        current_set.add(i)
                        changed = True
        
        greedy_result = list(current_set)
        greedy_size = len(greedy_result)
        
        # Try 1-swap moves: remove one vertex, try to add multiple non-conflicting vertices
        if time.time() < deadline - 1:
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                current_list = list(greedy_result)
                for rem in current_list:
                    if time.time() > deadline - 1:
                        break
                    # Remove rem
                    new_set = set(greedy_result) - {rem}
                    # Try to greedily add as many non-conflicting vertices as possible
                    for add in range(n):
                        if add not in new_set:
                            if not (conflicts[add] & new_set):
                                new_set.add(add)
                    if len(new_set) > len(greedy_result):
                        greedy_result = list(new_set)
                        greedy_size = len(greedy_result)
                        improved = True
                        break
    
    # Extract the result
    result_nodes = [valid_nodes[i] for i in greedy_result]
    
    # Final verification: ensure no pair (a,b) with a<=b has a+b as a perfect square
    def is_valid(nodes):
        for idx_a in range(len(nodes)):
            a = nodes[idx_a]
            if 2 * a in squares:
                return False
            for idx_b in range(idx_a + 1, len(nodes)):
                b = nodes[idx_b]
                if a + b in squares:
                    return False
        return True
    
    if not result_nodes or not is_valid(result_nodes):
        # Fallback: greedy valid set
        result_nodes = []
        for v in valid_nodes:
            ok = True
            for u in result_nodes:
                if u + v in squares:
                    ok = False
                    break
            if ok:
                result_nodes.append(v)
    
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
    
    return result_nodes
