"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: 150s wall-clock limit, use 145s for safety
_DEADLINE_START = None
_DEADLINE_LIMIT = 145.0  # seconds

def _ensure_deadline():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return time.time() - _DEADLINE_START < _DEADLINE_LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a combined strategy:
    - Regret-based scoring: how much more expensive it would be to visit later
    - Connection quality: how well-connected the candidate is to remaining nodes
    - Distance to current: local proximity
    
    This captures both local and global tour structure.
    """
    # Deadline check - if we're running out of time, fall back to nearest neighbor
    if not _ensure_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # Pick the one that gives shorter tour when we consider the return
        d1 = distance_matrix[current_node, unvisited_nodes[0]] + distance_matrix[unvisited_nodes[0], destination_node]
        d2 = distance_matrix[current_node, unvisited_nodes[1]] + distance_matrix[unvisited_nodes[1], destination_node]
        if d1 <= d2:
            return int(unvisited_nodes[0])
        else:
            return int(unvisited_nodes[1])
    
    # For small sets, use a more thorough evaluation
    if n_unvisited <= 8:
        # Try each candidate and estimate resulting tour length
        best_node = int(unvisited_nodes[0])
        best_score = float('inf')
        
        for candidate in unvisited_nodes:
            # Cost of going from current to candidate
            cost_here = distance_matrix[current_node, candidate]
            # Estimate cost of the rest: min distance from candidate to any unvisited (excluding itself)
            # plus distance from that to destination
            remaining = np.setdiff1d(unvisited_nodes, candidate)
            if len(remaining) > 0:
                # Approximate: min distance from candidate to remaining + min from remaining to dest
                min_to_remaining = np.min(distance_matrix[candidate, remaining])
                min_remaining_to_dest = np.min(distance_matrix[remaining, destination_node])
                # Also consider direct to destination if it's the last
                direct_to_dest = distance_matrix[candidate, destination_node]
                est_rest = min(min_to_remaining + min_remaining_to_dest, direct_to_dest)
            else:
                est_rest = distance_matrix[candidate, destination_node]
            
            total_est = cost_here + est_rest
            if total_est < best_score:
                best_score = total_est
                best_node = int(candidate)
        
        return best_node
    
    # For larger sets, use a composite score
    # 1. Distance from current to candidate (local)
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Regret: how much more expensive this candidate becomes if we don't visit it now
    #    For each candidate, compute the average distance from other unvisited nodes to this candidate
    #    High regret means it's hard to reach later
    if n_unvisited <= 20:
        # Compute regret for each candidate
        # regret[i] = mean distance from other unvisited to candidate i
        # This is expensive, so only do it for small n
        dists_between = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(dists_between, 0)
        # Average distance from other nodes to each candidate (row sum / (n-1))
        avg_dist_to_candidate = np.sum(dists_between, axis=0) / (n_unvisited - 1)
        regret = avg_dist_to_candidate
    else:
        # Approximate regret using a subset
        # Sample up to 15 other nodes to estimate average distance
        sample_size = min(15, n_unvisited - 1)
        if sample_size < n_unvisited - 1:
            rng = np.random.default_rng(42)
            other_indices = rng.choice(n_unvisited - 1, size=sample_size, replace=False)
            # Build sample set (excluding each candidate's own index)
            # This is tricky, so use a simpler approximation
            # Use the average distance from current to all unvisited as a baseline
            regret = np.zeros(n_unvisited)
            for i in range(n_unvisited):
                # Average distance from a few random other nodes
                others = [j for j in range(n_unvisited) if j != i]
                if len(others) > sample_size:
                    others_sample = np.random.default_rng(42 + i).choice(others, size=sample_size, replace=False)
                else:
                    others_sample = np.array(others)
                regret[i] = np.mean(distance_matrix[unvisited_nodes[others_sample], unvisited_nodes[i]])
    
    # 3. Connection quality: how well-connected is the candidate to the rest
    #    Low average distance to other unvisited means good connectivity
    if n_unvisited <= 20:
        conn_quality = -avg_dist_to_candidate  # Lower is better (well-connected)
    else:
        conn_quality = -regret  # Use regret as proxy
    
    # 4. Distance to destination (to avoid ending up far from where we need to go)
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Normalize each component to [0, 1] for weighting
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_dist_current = normalize(dist_from_current)
    norm_regret = normalize(regret)
    norm_conn = normalize(conn_quality)
    norm_dist_dest = normalize(dist_to_dest)
    
    # Composite score: lower is better
    # Weights: proximity to current, regret (avoid getting stuck), connectivity, distance to dest
    w_current = 0.35
    w_regret = 0.30
    w_conn = 0.20
    w_dest = 0.15
    
    composite = (w_current * norm_dist_current + 
                 w_regret * norm_regret + 
                 w_conn * norm_conn + 
                 w_dest * norm_dist_dest)
    
    best_idx = np.argmin(composite)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
