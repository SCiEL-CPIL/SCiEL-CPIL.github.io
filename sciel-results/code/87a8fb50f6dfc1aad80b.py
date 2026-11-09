import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # 1. Precompute squares
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # 2. Filter self-conflicts
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    valid_set = set(valid_nodes)
    M = len(valid_nodes)
    
    if M == 0:
        return [1]
    
    # 3. Build conflict structure
    conflict = {v: set() for v in valid_nodes}
    for i, v in enumerate(valid_nodes):
        for u in valid_nodes[i+1:]:
            if v + u in squares:
                conflict[v].add(u)
                conflict[u].add(v)
    
    def is_valid_set(S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            if 2 * a in squares:
                return False
            for j in range(i + 1, n):
                b = S_list[j]
                if a + b in squares:
                    return False
        return True
    
    def local_search_safe(initial_set):
        S = set(initial_set)
        current_size = len(S)
        
        while True:
            improved = False
            
            # Greedy Add
            for v in valid_nodes:
                if v in S:
                    continue
                if not conflict[v].intersection(S):
                    S.add(v)
                    current_size += 1
                    improved = True
            
            if not improved:
                # Try Swap
                swap_improved = False
                for r in list(S):
                    S.remove(r)
                    S_temp = set(S)
                    for v in valid_nodes:
                        if v == r:
                            continue
                        if v in S_temp:
                            continue
                        if not conflict[v].intersection(S_temp):
                            S_temp.add(v)
                    
                    new_size = len(S_temp)
                    if new_size > current_size:
                        S = S_temp
                        current_size = new_size
                        improved = True
                        swap_improved = True
                        break
                    else:
                        S.add(r)
                
                if not improved:
                    break
        
        return S
    
    best_size = 0
    best_set = set()
    
    # Generate multiple random initial sets for local search
    num_restarts = 5000
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:  # Leave 5s for ILP attempt
            break
        
        random.shuffle(valid_nodes)
        init_set = set()
        for v in valid_nodes:
            if not conflict[v].intersection(init_set):
                init_set.add(v)
        
        S = local_search_safe(init_set)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Try ILP to find exact solution
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix, csc_matrix
        
        # Variables x[0..M-1]
        c = -np.ones(M)
        
        # Collect conflict edges
        edges = []
        for i, v in enumerate(valid_nodes):
            for u in valid_nodes[i+1:]:
                if v + u in squares:
                    edges.append((i, valid_nodes.index(u)))
        
        if not edges:
            # No conflicts, take all valid nodes
            return valid_nodes
        
        # Build constraint matrix
        A = lil_matrix((len(edges), M), dtype=int)
        b = np.ones(len(edges))
        
        for k, (i, j) in enumerate(edges):
            A[k, i] = 1
            A[k, j] = 1
            
        A = csc_matrix(A)
        
        lb = np.zeros(M)
        ub = np.ones(M)
        integrality = np.ones(M)
        
        # Solve with time awareness
        ilp_start = time.time()
        res = milp(c, constraints=LinearConstraint(A, -np.inf, b), 
                   integrality=integrality, bounds=Bounds(lb, ub))
        ilp_time = time.time() - ilp_start
        
        if res.success and res.x is not None:
            # Extract solution
            sol_indices = [i for i in range(M) if res.x[i] > 0.5]
            ilp_set = set(valid_nodes[i] for i in sol_indices)
            
            if is_valid_set(ilp_set) and len(ilp_set) > best_size:
                best_size = len(ilp_set)
                best_set = ilp_set
    
    except Exception as e:
        # ILP failed, continue with local search result
        pass
    
    # Final verification and fallback
    if best_set and not is_valid_set(best_set):
        fallback_set = set()
        for v in valid_nodes:
            if not conflict[v].intersection(fallback_set):
                fallback_set.add(v)
        best_set = fallback_set
    
    if not best_set:
        # Ultimate fallback
        best_set = {valid_nodes[0]} if valid_nodes else {1}
    
    return list(best_set)
