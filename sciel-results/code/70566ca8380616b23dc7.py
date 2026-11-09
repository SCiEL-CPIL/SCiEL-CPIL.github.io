import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Try to find a valid graph for n=20 first, then fall back to smaller n
    for target_n in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 2:
            break
        
        # Try multiple random initializations with local search
        found = _local_search(target_n, start_time, deadline)
        if found is not None:
            adj, n = found
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Fallback: return a trivially valid small graph
    return {'n': 5, 'edges': []}


def _local_search(target_n, start_time, deadline):
    """
    Local search using spectral gap heuristic to guide edge modifications.
    Cost function: sum of squared deviations of induced 6-vertex subgraph degrees from their mean.
    """
    n = target_n
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 3:
            return None
        
        # Random initialization
        rng = np.random.RandomState(attempt * 1000 + 42)
        adj = (rng.rand(n, n) < 0.5).astype(np.int8)
        np.fill_diagonal(adj, 0)
        # Symmetrize
        adj = ((adj + adj.T) > 0).astype(np.int8)
        
        # Run local search
        adj = _improve(adj, n, start_time, deadline)
        
        # Check if valid
        if _is_valid_fast(adj, n):
            return adj, n
    
    return None


def _improve(adj, n, start_time, deadline):
    """
    Iteratively flip edges that most reduce the global irregularity metric.
    Metric: sum over all 6-subsets of sum of squared deviations of degrees from mean.
    """
    current_adj = adj.copy()
    
    # Precompute all 6-subsets for efficiency
    all_6subsets = list(combinations(range(n), 6))
    
    # Compute initial cost
    cost = _compute_cost(current_adj, all_6subsets)
    
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        # Find the edge flip that most reduces cost
        best_gain = 0
        best_flip = None
        
        # Sample a subset of edges to consider (for speed)
        edges_to_consider = []
        for i in range(n):
            for j in range(i + 1, n):
                edges_to_consider.append((i, j))
        
        # Shuffle and consider a random subset
        np.random.shuffle(edges_to_consider)
        edges_to_check = edges_to_consider[:min(200, len(edges_to_consider))]
        
        for (i, j) in edges_to_check:
            if time.time() > deadline - 5:
                break
            
            # Try flipping edge (i, j)
            current_adj[i, j] = 1 - current_adj[i, j]
            current_adj[j, i] = current_adj[i, j]
            
            new_cost = _compute_cost(current_adj, all_6subsets)
            gain = cost - new_cost
            
            # Revert
            current_adj[i, j] = 1 - current_adj[i, j]
            current_adj[j, i] = current_adj[i, j]
            
            if gain > best_gain:
                best_gain = gain
                best_flip = (i, j)
        
        if best_flip is None or best_gain <= 0:
            # No improving move found, try random perturbation
            if iteration > 10:
                # Random restart of the local search
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    current_adj[i, j] = 1 - current_adj[i, j]
                    current_adj[j, i] = current_adj[i, j]
                    cost = _compute_cost(current_adj, all_6subsets)
                continue
            else:
                break
        
        # Apply the best flip
        i, j = best_flip
        current_adj[i, j] = 1 - current_adj[i, j]
        current_adj[j, i] = current_adj[i, j]
        cost -= best_gain
    
    return current_adj


def _compute_cost(adj, all_6subsets):
    """
    Compute the cost: sum over all 6-subsets of sum of squared deviations of degrees from mean.
    """
    total_cost = 0.0
    for subset in all_6subsets:
        # Compute degrees within the induced subgraph
        s = len(subset)
        degrees = np.zeros(s, dtype=np.int32)
        for idx_i in range(s):
            u = subset[idx_i]
            for idx_j in range(idx_i + 1, s):
                v = subset[idx_j]
                if adj[u, v]:
                    degrees[idx_i] += 1
                    degrees[idx_j] += 1
        
        # Compute mean and squared deviations
        mean_deg = degrees.mean()
        sq_dev = np.sum((degrees - mean_deg) ** 2)
        total_cost += sq_dev
    
    return total_cost


def _is_valid_fast(adj, n):
    """Check if no subset of size >= 6 induces a regular graph."""
    if n < 6:
        return True
    
    # Check all subsets of size 6 to n
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_induced(adj, subset):
                return False
    return True


def _is_regular_induced(adj, subset):
    """Check if the induced subgraph on subset is regular."""
    s = len(subset)
    if s < 2:
        return True
    
    # Compute degrees within the induced subgraph
    degrees = np.zeros(s, dtype=np.int32)
    for i in range(s):
        u = subset[i]
        for j in range(i + 1, s):
            v = subset[j]
            if adj[u, v]:
                degrees[i] += 1
                degrees[j] += 1
    
    # Check if all degrees are equal
    d0 = degrees[0]
    for i in range(1, s):
        if degrees[i] != d0:
            return False
    return True
