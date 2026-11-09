"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _is_regular_subset_np(adj_matrix, subset):
    """Check if induced subgraph on subset is regular, using numpy adjacency matrix."""
    k = len(subset)
    if k < 2:
        return True
    sub = adj_matrix[np.ix_(subset, subset)]
    np.fill_diagonal(sub, 0)
    degs = sub.sum(axis=1)
    return np.all(degs == degs[0])

def _check_all_subsets_np(adj_matrix, n, min_size=6):
    """Check if any subset of size >= min_size induces a regular subgraph."""
    for size in range(min_size, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_subset_np(adj_matrix, subset):
                return False, subset
    return True, None

def _fast_check_6sets(adj_matrix, n):
    """Fast check: only 6-sets, using vectorized numpy ops."""
    for subset in combinations(range(n), 6):
        if _is_regular_subset_np(adj_matrix, subset):
            return False, subset
    return True, None

def _local_search(adj_matrix, n, deadline):
    """Local search: flip edges to eliminate regular subsets."""
    current_adj = adj_matrix.copy()
    best_adj = current_adj.copy()
    best_score = -1  # -1 means invalid, 0 means valid
    
    # Initial check
    valid, viol = _check_all_subsets_np(current_adj, n)
    if valid:
        return current_adj
    
    for step in range(50000):
        if time.time() > deadline:
            break
        
        # Find a violation
        valid, viol = _check_all_subsets_np(current_adj, n)
        if valid:
            return current_adj
        
        if viol is None:
            break
            
        # Try flipping a random edge within or near the violating subset
        k = len(viol)
        # Pick a random pair from the violating subset
        idx1, idx2 = random.sample(range(k), 2)
        u, v = viol[idx1], viol[idx2]
        
        # Flip the edge
        current_adj[u, v] = 1 - current_adj[u, v]
        current_adj[v, u] = current_adj[u, v]
        
        # Quick check: does this create or remove violations?
        # Just track if we've made progress by checking a few random subsets
        # For efficiency, do a full check every 50 steps
        if step % 50 == 0:
            valid, _ = _check_all_subsets_np(current_adj, n)
            if valid:
                return current_adj
    
    return None

def _constructive_build(n, deadline):
    """Build graph incrementally: add vertices one by one, choosing edges to avoid regular subsets."""
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Add vertices 0..n-1
    for new_v in range(n):
        if time.time() > deadline:
            return None
        
        # Try different edge configurations for the new vertex
        # Start with no edges, then try adding edges
        best_config = None
        
        # Number of possible edges from new_v to previous vertices
        num_prev = new_v
        num_configs = min(1 << num_prev, 200)  # Cap for large num_prev
        
        if num_prev <= 10:
            # Enumerate all possible edge sets
            configs = list(range(1 << num_prev))
            random.shuffle(configs)
            configs = configs[:num_configs]
        else:
            # Random sampling
            configs = []
            for _ in range(num_configs):
                mask = 0
                for j in range(num_prev):
                    if random.random() < 0.4:
                        mask |= (1 << j)
                configs.append(mask)
        
        for config in configs:
            if time.time() > deadline:
                break
            
            # Apply this config
            trial_adj = adj.copy()
            for j in range(num_prev):
                if config & (1 << j):
                    trial_adj[new_v, j] = 1
                    trial_adj[j, new_v] = 1
            
            # Check if this creates a regular subset of size >= 6
            # Only need to check subsets that include new_v (since previous subsets were OK)
            # Subsets of size 6 that include new_v: choose 5 from previous
            # For efficiency, check a sample
            valid = True
            for subset in combinations(range(new_v), 5):
                full_subset = tuple(list(subset) + [new_v])
                if _is_regular_subset_np(trial_adj, full_subset):
                    valid = False
                    break
            
            if valid:
                # Also check larger subsets that include new_v (size 7, 8, ...)
                # For size 7: choose 6 from previous
                for subset in combinations(range(new_v), 6):
                    full_subset = tuple(list(subset) + [new_v])
                    if _is_regular_subset_np(trial_adj, full_subset):
                        valid = False
                        break
                if not valid:
                    continue
                
                # For size 8+: choose 7 from previous
                for subset in combinations(range(new_v), 7):
                    full_subset = tuple(list(subset) + [new_v])
                    if _is_regular_subset_np(trial_adj, full_subset):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check even larger
                for size in range(9, new_v + 1):
                    prev_size = size - 1
                    for subset in combinations(range(new_v), prev_size):
                        full_subset = tuple(list(subset) + [new_v])
                        if _is_regular_subset_np(trial_adj, full_subset):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_config = config
                    break
        
        if best_config is None:
            # Try a random fallback
            trial_adj = adj.copy()
            for j in range(num_prev):
                if random.random() < 0.35:
                    trial_adj[new_v, j] = 1
                    trial_adj[j, new_v] = 1
            
            # Quick check
            valid = True
            for subset in combinations(range(new_v), min(5, new_v - 1)):
                full_subset = tuple(list(subset) + [new_v])
                if len(full_subset) >= 6 and _is_regular_subset_np(trial_adj, full_subset):
                    valid = False
                    break
            
            if not valid:
                return None
            
            adj = trial_adj
        else:
            for j in range(num_prev):
                if best_config & (1 << j):
                    adj[new_v, j] = 1
                    adj[j, new_v] = 1
    
    # Final verification
    valid, _ = _check_all_subsets_np(adj, n)
    if valid:
        return adj
    return None

def _hill_climb(adj_matrix, n, deadline):
    """Hill climbing: start from a random graph, flip edges to fix violations."""
    current = adj_matrix.copy()
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Find a violation
        valid, viol = _check_all_subsets_np(current, n)
        if valid:
            return current
        
        if viol is None:
            break
        
        # Try to fix by flipping an edge
        k = len(viol)
        # Try several random edge flips
        improved = False
        for _ in range(20):
            idx1, idx2 = random.sample(range(k), 2)
            u, v = viol[idx1], viol[idx2]
            
            old_val = current[u, v]
            current[u, v] = 1 - old_val
            current[v, u] = current[u, v]
            
            # Quick check: is the violation gone?
            still_viol = _is_regular_subset_np(current, viol)
            
            if not still_viol:
                # Check if we created a new violation (sample check)
                # For now, accept and continue
                improved = True
                break
            else:
                # Revert
                current[u, v] = old_val
                current[v, u] = old_val
        
        if not improved:
            # Try a completely random edge flip
            u, v = random.sample(range(n), 2)
            current[u, v] = 1 - current[u, v]
            current[v, u] = current[u, v]
    
    # Final check
    valid, _ = _check_all_subsets_np(current, n)
    if valid:
        return current
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to build graphs for n=20, 21
    for target_n in [20, 21]:
        if time.time() > deadline:
            break
        
        found = False
        
        # Strategy 1: Constructive build
        for attempt in range(10):
            if time.time() > deadline:
                break
            random.seed(attempt * 1000 + target_n)
            adj = _constructive_build(target_n, deadline)
            if adj is not None:
                # Verify
                valid, _ = _check_all_subsets_np(adj, target_n)
                if valid:
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj[i, j]:
                                edges.append([i, j])
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
        
        if found:
            break
        
        # Strategy 2: Random + hill climbing
        for attempt in range(20):
            if time.time() > deadline:
                break
            random.seed(attempt * 777 + target_n + 100)
            # Random sparse graph
            adj = np.zeros((target_n, target_n), dtype=np.int8)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.3:
                        adj[i, j] = 1
                        adj[j, i] = 1
            
            result = _hill_climb(adj, target_n, deadline)
            if result is not None:
                valid, _ = _check_all_subsets_np(result, target_n)
                if valid:
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if result[i, j]:
                                edges.append([i, j])
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
        
        if found:
            break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
