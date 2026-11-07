import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP using a repulsive force model.
    
    Strategy:
    1. Repulsive Force: Penalty is inversely proportional to the Euclidean distance.
       Short edges (high value, high probability of being in optimal tour) receive
       disproportionately large penalties compared to long edges. This forces the
       search to explore alternative short connections by making the 'obvious'
       short edges temporarily expensive.
    2. Usage Saturation: The penalty magnitude is scaled by a saturating function
       (tanh) of the cumulative usage count. This ensures that even edges used
       once receive a penalty, but edges used many times receive a maximum penalty,
       preventing the penalty from growing unboundedly and destabilizing the metric.
    3. Multiplicative Perturbation: The penalty is applied multiplicatively to the
       original distance. This preserves the relative ordering of long edges while
       drastically altering the cost of short, frequent edges.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Work on a copy
    d_out = edge_distance.copy()
    
    if n <= 1:
        return d_out
    
    # 1. Compute scale factor for normalization
    #    We need a reference distance to normalize the inverse distance term.
    #    Using the median distance ensures the penalty is relative to the instance geometry.
    mask = ~np.eye(n, dtype=bool)
    valid_distances = edge_distance[mask]
    valid_distances = valid_distances[valid_distances > 0]
    
    if len(valid_distances) == 0:
        return d_out
    
    median_dist = np.median(valid_distances)
    if median_dist <= 0:
        return d_out
    
    # 2. Prepare usage counts and repulsive force
    #    We operate on the upper triangle to avoid double counting in the logic,
    #    then symmetrize.
    iu = np.triu_indices(n, k=1)
    usage_counts = edge_n_used[iu]
    distances = edge_distance[iu]
    
    # Ensure distances are strictly positive for division
    safe_distances = np.maximum(distances, 1e-9)
    
    # 3. Calculate the repulsive force
    #    Force is proportional to 1/distance.
    #    We scale it so that the 'average' edge (median distance) has a base force of 1.
    #    So force = median_dist / distance.
    #    Short edges (d < median) get force > 1.
    #    Long edges (d > median) get force < 1.
    base_force = median_dist / safe_distances
    
    # 4. Scale by usage
    #    We want edges that are used frequently to be penalized more.
    #    Use a saturating function to map counts to [0, 1] or similar.
    #    Counts can vary. Let's normalize by the max count in this specific matrix
    #    or use a fixed scale if we want consistency. 
    #    Given the "accumulated over rounds" nature, counts grow.
    #    Let's use a soft saturation: factor = 1 - exp(-usage / scale)
    #    Or simpler: factor = usage / (usage + k)
    #    Let's use a fixed scale k=5.0. If an edge is used 5 times, it gets 0.5 penalty weight.
    #    If used 50 times, it gets ~0.91.
    #    This ensures that even early rounds (low counts) have some effect, but it grows.
    
    # To make it robust to the specific instance's usage distribution, 
    # let's normalize counts by the max count in the current edge_n_used matrix.
    max_usage = np.max(usage_counts)
    if max_usage > 0:
        # Normalize usage to [0, 1]
        norm_usage = usage_counts / max_usage
    else:
        norm_usage = np.zeros_like(usage_counts)
    
    # The repulsive force is strongest for short edges.
    # We want the penalty to be significant.
    # Let's define the penalty as:
    # penalty_factor = 1 + alpha * norm_usage * base_force
    # Where alpha controls the strength.
    # If alpha is large, short used edges become very expensive.
    
    # Let's try a multiplicative model:
    # d_new = d_old * (1 + penalty)
    # penalty = strength * norm_usage * (median_dist / d_old)
    
    # To avoid blowing up, let's cap the multiplier.
    # If d_old is very small, 1/d_old is large.
    # Let's clamp the force.
    max_force = 10.0
    base_force = np.minimum(base_force, max_force)
    
    # Strength parameter: 
    # We want to disrupt the local optimum. 
    # A strength of 2.0 means a frequently used short edge (force ~ 2-5) 
    # with norm_usage 1.0 gets a multiplier of 1 + 2*1*2 = 5.
    # This is a strong perturbation.
    
    strength = 2.0
    
    penalty = strength * norm_usage * base_force
    
    # Additional boost for edges in the current tour
    # These are the edges we definitely want to break.
    tour = np.array(local_opt_tour)
    if len(tour) >= 2:
        # Identify tour edges in the upper triangle indices
        # We can just apply a flat high penalty to tour edges directly in the matrix
        # But since we are constructing from upper triangle, let's mark them.
        
        # Create a boolean mask for tour edges in the upper triangle
        tour_edge_mask = np.zeros(len(iu[0]), dtype=bool)
        
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            if a > b:
                a, b = b, a
            # Find index in iu where (a,b) exists
            # This is O(N) per edge, total O(N^2) which is fine for N=100
            # But vectorized is better.
            pass # We'll do this differently.
            
        # Easier: Construct the full penalty matrix first, then apply tour boost.
        
    # Construct full penalty matrix
    penalty_matrix = np.zeros((n, n), dtype=float)
    penalty_matrix[iu] = penalty
    penalty_matrix = penalty_matrix + penalty_matrix.T
    
    # Apply tour edge boost
    if len(tour) >= 2:
        # Tour edges get an extra additive penalty to the multiplier
        # Or just a fixed large multiplier.
        # Let's add a fixed boost to the penalty value for tour edges.
        tour_boost = 1.0 # This will be added to the penalty term
        
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            penalty_matrix[a, b] += tour_boost
            penalty_matrix[b, a] += tour_boost
        # Close tour
        a, b = tour[-1], tour[0]
        penalty_matrix[a, b] += tour_boost
        penalty_matrix[b, a] += tour_boost
    
    # Apply multiplicative perturbation
    # d_out = d_out * (1 + penalty_matrix)
    # But we must be careful: if penalty_matrix is 0, d_out stays same.
    # If penalty_matrix is large, d_out grows.
    
    # Ensure diagonal is 0
    np.fill_diagonal(penalty_matrix, 0.0)
    
    d_out = d_out * (1.0 + penalty_matrix)
    
    # Ensure symmetry
    d_out = (d_out + d_out.T) / 2.0
    
    # Ensure diagonal is 0
    np.fill_diagonal(d_out, 0.0)
    
    # Ensure finite and non-negative
    d_out = np.nan_to_num(d_out, nan=0.0, posinf=1e10, neginf=1e10)
    d_out = np.maximum(d_out, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        return d_out
        
    return d_out
