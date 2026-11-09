import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Option-value heuristic: for each transition i->j, estimate the
    expected prize collectable from j onwards given remaining budget.
    
    eta[i, j] = prize[j] + expected_future_prize(j, remaining_budget)
    where remaining_budget = max_len - d[0,i] - d[i,j] - d[j,0]
    (approximate: assume ant started at 0, went to i, now going to j)
    
    This captures the structural insight that a move is good not just for
    its immediate prize, but for the options it preserves.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    prizes = prizes.copy()
    prizes[0] = 0.0
    
    eps = 1e-9
    eta = np.ones((n, n), dtype=np.float64)
    
    # For each pair (i, j), i != j:
    # Approximate remaining budget after visiting j (coming from i, starting from 0)
    # Path: 0 -> ... -> i -> j -> ... -> 0
    # Simplest approximation: assume path so far is just 0->i->j
    # remaining = max_len - d[0,i] - d[i,j] - d[j,0]
    # But we don't know the exact path, so use a representative estimate.
    
    # For each destination j, precompute: for various remaining budgets,
    # what's the best prize we can still collect from unvisited nodes?
    # Since we can't know which nodes are visited, use a greedy estimate:
    # Given remaining budget B, the best additional prize is the sum of
    # top-k prizes where k is max number of nodes we can visit.
    
    # Precompute: for each node j, and for a range of remaining budgets,
    # estimate the future prize potential.
    
    # Simpler structural approach: 
    # eta[i,j] = prize[j] * (1 + alpha * remaining_fraction)
    # where remaining_fraction = max(0, (max_len - d[i,j] - d[j,0])) / max_len
    # This rewards moves that leave more budget for future collection.
    
    # Even better: use a two-term model
    # Term 1: immediate prize
    # Term 2: option value = prize[j] * beta * (remaining_budget / max_len)
    # This makes high-prize nodes with lots of remaining budget very attractive.
    
    # Let's try: eta[i,j] = prize[j] / (d[i,j] + d[j,0] + eps) * (1 + gamma * remaining_frac)
    # where remaining_frac = max(0, (max_len - d[i,j] - d[j,0]) / max_len)
    
    # Actually, let me try a cleaner structural model:
    # The value of moving to j is:
    #   V(i,j) = prize[j] + sum_{k != i,j,0} prize[k] * P(visit k | budget remaining)
    # Approximate P as: 1 if d[j,k] + d[k,0] <= remaining_budget, else 0
    # This is expensive but n=50 so it's doable.
    
    d_to_depot = distance_matrix[:, 0]  # shape (n,)
    
    for i in range(n):
        for j in range(n):
            if i == j or j == 0:
                eta[i, j] = eps
                continue
            
            d_ij = distance_matrix[i, j]
            d_j0 = d_to_depot[j]
            
            # Remaining budget after going 0->i->j->0 (approximate)
            # Use d[0,i] as proxy for path length so far
            d_0i = distance_matrix[0, i]
            remaining = max_len - d_0i - d_ij - d_j0
            
            if remaining < 0:
                eta[i, j] = eps
                continue
            
            # Immediate prize
            val = prizes[j]
            
            # Option value: how many other high-value nodes can we still visit?
            # For each other node k, check if we can visit it after j
            # Cost to visit k after j: d[j,k] + d[k,0]
            # Must fit in remaining budget
            future_val = 0.0
            for k in range(1, n):
                if k == i or k == j:
                    continue
                d_jk = distance_matrix[j, k]
                d_k0 = d_to_depot[k]
                cost_k = d_jk + d_k0
                if cost_k <= remaining:
                    future_val += prizes[k]
            
            # Combine: immediate + discounted future
            # Discount future by a factor to avoid over-counting
            eta[i, j] = val + 0.5 * future_val
    
    # Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    return eta
