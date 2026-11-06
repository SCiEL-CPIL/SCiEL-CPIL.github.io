"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
import numpy as np

def _is_regular_subset_np(adj_matrix, subset):
    """Check if induced subgraph on subset is regular using numpy."""
    sub = adj_matrix[np.ix_(subset, subset)]
    degrees = sub.sum(axis=1)
    return np.all(degrees == degrees[0])

def _count_reg_subgraphs_np(adj_matrix, n, k):
    """Count regular induced subgraphs of size k using numpy vectorization."""
    count = 0
    for subset in itertools.combinations(range(n), k):
        if _is_regular_subset_np(adj_matrix, subset):
            count += 1
    return count

def _has_violation_full_np(adj_matrix, n):
    """Check if any induced subgraph of size >= 6 is regular."""
    for k in range(6, n + 1):
        if _count_reg_subgraphs_np(adj_matrix, n, k) > 0:
            return True
    return False

def _find_best_edge_flip(adj, n, violating_subsets, deadline):
    """
    Find the edge flip that resolves the most violating 6-subsets.
    violating_subsets: list of 6-tuples that are currently regular.
    Returns (u, v) of the best edge to flip, or None.
    """
    if not violating_subsets:
        return None
    
    edge_score = {}
    
    for subset in violating_subsets:
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                u, v = subset[i], subset[j]
                key = (min(u,v), max(u,v))
                edge_score[key] = edge_score.get(key, 0) + 1
    
    if not edge_score:
        return None
    
    max_score = max(edge_score.values())
    candidates = [k for k, v in edge_score.items() if v == max_score]
    return random.choice(candidates)

def _constraint_propagation_search(adj, n, deadline, max_iter=2000):
    """
    Iteratively fix violations by choosing edge flips that resolve most conflicts.
    Returns True if a valid graph is found, False otherwise.
    """
    # Convert to numpy matrix for fast subset checks
    adj_matrix = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    
    for iteration in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        # Find all violating 6-subsets using numpy
        violating = []
        for subset in itertools.combinations(range(n), 6):
            if _is_regular_subset_np(adj_matrix, subset):
                violating.append(subset)
        
        if not violating:
            # No 6-vertex violations; check larger sizes
            if not _has_violation_full_np(adj_matrix, n):
                return True
            else:
                # Has larger violation but no 6-violation; try random flip
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj_matrix[u, v] = 0
                    adj_matrix[v, u] = 0
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj_matrix[u, v] = 1
                    adj_matrix[v, u] = 1
                continue
        
        # Find best edge flip
        best_edge = _find_best_edge_flip(adj, n, violating, deadline)
        if best_edge is None:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
        else:
            u, v = best_edge
        
        # Flip the edge
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            adj_matrix[u, v] = 0
            adj_matrix[v, u] = 0
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Target n=20 first, then 19, 18...
    for n in [20, 19, 18, 17]:
        if time.time() > deadline - 10:
            break
            
        # Multiple restarts for this n
        max_restarts = 20 if n == 20 else 12
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Use constraint propagation to fix violations
            if _constraint_propagation_search(adj, n, deadline):
                # Valid graph found!
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                return {'n': best_n, 'edges': best_edges}
            
            # If constraint propagation didn't find a solution, try simulated annealing as backup
            if time.time() < deadline - 8:
                # Build numpy matrix for fast cost evaluation
                adj_matrix = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            adj_matrix[i, j] = 1
                            adj_matrix[j, i] = 1
                
                current_cost = _count_reg_subgraphs_np(adj_matrix, n, 6)
                T = 5.0
                T_min = 0.1
                cooling = 0.995
                max_iter_sa = 400
                
                it = 0
                found = False
                while it < max_iter_sa and T > T_min and time.time() < deadline - 3:
                    it += 1
                    
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        adj_matrix[u, v] = 0
                        adj_matrix[v, u] = 0
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        adj_matrix[u, v] = 1
                        adj_matrix[v, u] = 1
                    
                    new_cost = _count_reg_subgraphs_np(adj_matrix, n, 6)
                    delta = new_cost - current_cost
                    
                    if delta < 0:
                        current_cost = new_cost
                    else:
                        if delta == 0 or random.random() < pow(2.71828, -delta / T):
                            current_cost = new_cost
                        else:
                            # Revert
                            if adj[u] & (1 << v):
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                adj_matrix[u, v] = 0
                                adj_matrix[v, u] = 0
                            else:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                adj_matrix[u, v] = 1
                                adj_matrix[v, u] = 1
                    
                    T *= cooling
                    
                    if current_cost == 0 and not _has_violation_full_np(adj_matrix, n):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        if n > best_n:
                            best_n = n
                            best_edges = edges
                        found = True
                        break
                
                if found:
                    return {'n': best_n, 'edges': best_edges}

    # Fallback: return best found or default
    if best_n == 5:
        for n in [15, 14, 13, 12, 11, 10]:
            if time.time() > deadline - 2:
                break
            for _ in range(10):
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                # Build numpy matrix
                adj_matrix = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            adj_matrix[i, j] = 1
                            adj_matrix[j, i] = 1
                if not _has_violation_full_np(adj_matrix, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    break
            if best_n > 5:
                break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
