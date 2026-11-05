import itertools
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def check_violations(adj_matrix, n):
        """
        Returns a list of (mask, size) for all induced subgraphs of size >= 6 that are regular.
        Uses vectorized numpy for size 6, and fallback for larger sizes if needed.
        For n=20, C(20,6)=38760. C(21,6)=54264.
        Checking all sizes >= 6 is expensive. 
        However, if we minimize size-6 violations, larger ones often disappear or are rarer.
        But we MUST check all sizes >= 6 for final validity.
        
        Optimization: 
        1. Quickly check size 6 using numpy. If any exist, return them (or a sample) to guide search.
        2. For final verification, do a full check.
        
        For the search objective, we can just count size-6 violations. 
        Why? A regular subgraph of size k >= 7 contains regular subgraphs of size 6? 
        No, not necessarily. A 7-regular graph induced on 7 vertices might not contain a 6-regular induced subgraph.
        Example: K7 is 6-regular. Remove one vertex -> K6 is 5-regular. Not regular? 
        Wait, "regular induced subgraph" means the induced subgraph itself is regular.
        K7 is 6-regular. Its induced subgraph on 6 vertices is K6, which is 5-regular. So K7 contains a regular subgraph of size 6.
        In general, if a graph is d-regular on k vertices, and we remove a vertex, the degrees of neighbors drop by 1. 
        If the graph is connected and d>0, it's unlikely to remain regular.
        
        Actually, minimizing size-6 violations is a good proxy. 
        Let's stick to counting size-6 violations for the cost function, 
        but perform a FULL check before returning a candidate.
        """
        if n < 6:
            return 0, []
            
        combos_6 = list(itertools.combinations(range(n), 6))
        if not combos_6:
            return 0, []
            
        idx_6 = np.array(combos_6)
        num_combos = len(idx_6)
        
        # Build incidence matrix X (num_combos x n)
        X = np.zeros((num_combos, n), dtype=np.int8)
        for i in range(num_combos):
            for v in idx_6[i]:
                X[i, v] = 1
                
        # Degrees D (num_combos x n)
        # D[i, v] = sum_u A[v, u] * X[i, u]
        # A is n x n. X is num_combos x n.
        # D = X @ A.T ? 
        # (X @ A.T)[i, v] = sum_u X[i, u] * A.T[u, v] = sum_u X[i, u] * A[v, u]. Correct.
        D = X @ adj_matrix.T
        
        # Extract degrees for vertices in each combo
        # D_subset[i, j] = D[i, idx_6[i, j]]
        D_subset = D[np.arange(num_combos)[:, None], idx_6]
        
        maxes = np.max(D_subset, axis=1)
        mins = np.min(D_subset, axis=1)
        
        violations_mask = (maxes == mins)
        num_violations = np.sum(violations_mask)
        
        # Get indices of violations for potential repair (optional)
        viol_indices = np.where(violations_mask)[0]
        
        return num_violations, viol_indices

    def full_validity_check(adj_matrix, n):
        """
        Strict check for any regular induced subgraph of size >= 6.
        """
        # We already know size 6 is clean if num_violations == 0.
        # Check sizes 7..n
        for k in range(7, n + 1):
            for subset in itertools.combinations(range(n), k):
                # Check if induced subgraph on subset is regular
                # Get degrees
                degs = []
                for v in subset:
                    d = 0
                    for u in subset:
                        if u != v and adj_matrix[v, u]:
                            d += 1
                    degs.append(d)
                
                if len(set(degs)) == 1:
                    return False
        return True

    def masks_to_edges(adj_matrix, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j]:
                    edges.append([i, j])
        return edges

    # Simulated Annealing Parameters
    target_n = 21  # Try to jump to 21
    
    # Initialize
    adj_matrix = (np.random.rand(target_n, target_n) < 0.5).astype(np.int8)
    adj_matrix = np.triu(adj_matrix, 1)
    adj_matrix = adj_matrix + adj_matrix.T
    np.fill_diagonal(adj_matrix, 0)
    
    best_adj = adj_matrix.copy()
    best_cost = float('inf')
    
    current_cost, _ = check_violations(adj_matrix, target_n)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    steps = 0
    max_steps = 50000  # Adjust based on time
    
    while T > T_min and time.time() < deadline - 10.0:
        steps += 1
        
        # Pick a random edge to flip
        i = random.randint(0, target_n - 1)
        j = random.randint(0, target_n - 1)
        if i == j:
            continue
            
        # Flip edge
        old_val = adj_matrix[i, j]
        adj_matrix[i, j] = 1 - old_val
        adj_matrix[j, i] = adj_matrix[i, j]
        
        new_cost, _ = check_violations(adj_matrix, target_n)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = adj_matrix.copy()
                
                # If we found a valid graph (0 violations of size 6), 
                # we must check full validity before accepting as final answer
                # But for search, 0 is the goal.
                if new_cost == 0:
                    # Check full validity
                    if full_validity_check(adj_matrix, target_n):
                        edges = masks_to_edges(adj_matrix, target_n)
                        return {'n': target_n, 'edges': edges}
        else:
            # Metropolis criterion
            delta = new_cost - current_cost
            if delta < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta / T)
                
            if accept:
                current_cost = new_cost
            else:
                # Revert
                adj_matrix[i, j] = old_val
                adj_matrix[j, i] = old_val
                
        T *= alpha
        
    # If we didn't find a valid 21-vertex graph, try to salvage a 20-vertex one 
    # or return the best 21 found if it passes full check (unlikely if cost>0)
    
    # Check if best 21 is valid
    if best_cost == 0:
        if full_validity_check(best_adj, target_n):
            edges = masks_to_edges(best_adj, target_n)
            return {'n': target_n, 'edges': edges}
            
    # Fallback: Try n=20 with same method, or just return a known valid n=20 if we have one?
    # The prompt says "Graphs with 20 vertices are known". 
    # If we fail 21, we should try to find a valid 20 quickly.
    
    target_n = 20
    adj_matrix = (np.random.rand(target_n, target_n) < 0.5).astype(np.int8)
    adj_matrix = np.triu(adj_matrix, 1)
    adj_matrix = adj_matrix + adj_matrix.T
    np.fill_diagonal(adj_matrix, 0)
    
    best_adj_20 = adj_matrix.copy()
    best_cost_20 = float('inf')
    current_cost_20, _ = check_violations(adj_matrix, target_n)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    while T > T_min and time.time() < deadline - 5.0:
        i = random.randint(0, target_n - 1)
        j = random.randint(0, target_n - 1)
        if i == j:
            continue
            
        old_val = adj_matrix[i, j]
        adj_matrix[i, j] = 1 - old_val
        adj_matrix[j, i] = adj_matrix[i, j]
        
        new_cost_20, _ = check_violations(adj_matrix, target_n)
        
        if new_cost_20 < current_cost_20:
            current_cost_20 = new_cost_20
            if new_cost_20 < best_cost_20:
                best_cost_20 = new_cost_20
                best_adj_20 = adj_matrix.copy()
                
                if new_cost_20 == 0:
                    if full_validity_check(adj_matrix, target_n):
                        edges = masks_to_edges(adj_matrix, target_n)
                        return {'n': target_n, 'edges': edges}
        else:
            delta = new_cost_20 - current_cost_20
            if delta < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta / T)
                
            if accept:
                current_cost_20 = new_cost_20
            else:
                adj_matrix[i, j] = old_val
                adj_matrix[j, i] = old_val
                
        T *= alpha
        
    # If we still don't have a valid 20, try a few random restarts for 20
    for _ in range(5):
        if time.time() > deadline - 2.0:
            break
        adj_matrix = (np.random.rand(target_n, target_n) < 0.5).astype(np.int8)
        adj_matrix = np.triu(adj_matrix, 1)
        adj_matrix = adj_matrix + adj_matrix.T
        np.fill_diagonal(adj_matrix, 0)
        
        cost, _ = check_violations(adj_matrix, target_n)
        if cost == 0:
            if full_validity_check(adj_matrix, target_n):
                edges = masks_to_edges(adj_matrix, target_n)
                return {'n': target_n, 'edges': edges}

    # Final fallback: n=5
    return {'n': 5, 'edges': []}
