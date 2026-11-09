import numpy as np

def heuristics(prizes, weights):
    """
    Calculates heuristic scores for items in a Multiple-Constraint Knapsack Problem.
    
    Args:
    prizes: numpy array of shape (5, 100). Prizes for each item and constraint dimension.
    weights: numpy array of shape (5, 100). Weights for each item and constraint dimension.
    
    Returns:
    numpy array of shape (100,). Heuristic score for each item.
    """
    # Sum prizes to get a single prize value per item
    # Assuming the total prize is the sum of the components
    P = np.sum(prizes, axis=0)
    N = P.shape[0]
    
    # Calculate total weights for each constraint to normalize
    W_totals = np.sum(weights, axis=1)  # Shape (5,)
    inv_W = 1.0 / (W_totals + 1e-9)
    
    # Calculate normalized cost for each item
    # norm_weights[k, i] = weights[k, i] / W_totals[k]
    norm_weights = weights * inv_W[:, np.newaxis]  # Shape (5, 100)
    cost = np.sum(norm_weights, axis=0)  # Shape (100,)
    
    # Base heuristic: Prize / Normalized Cost
    eta = P / (cost + 1e-9)
    
    # Pareto Dominance Filter
    # Item j dominates i if:
    # 1. P[j] >= P[i]
    # 2. weights[:, j] <= weights[:, i] for all k
    # 3. At least one inequality is strict
    
    # Vectorized dominance check
    # P_diff[j, i] = P[j] - P[i]
    P_diff = P[:, np.newaxis] - P[np.newaxis, :]  # Shape (100, 100)
    prize_ok = P_diff >= 0
    
    # W_j[k, j, i] = weights[k, j]
    # W_i[k, j, i] = weights[k, i]
    W_j = weights[:, :, np.newaxis]  # Shape (5, 100, 1)
    W_i = weights[:, np.newaxis, :]  # Shape (5, 1, 100)
    
    # Check if weights of j are <= weights of i for all k
    w_leq = W_j <= W_i  # Shape (5, 100, 100)
    weight_ok = np.all(w_leq, axis=0)  # Shape (100, 100)
    
    # Check if they are identical (to ensure strict dominance for at least one constraint or prize)
    w_eq = W_j == W_i  # Shape (5, 100, 100)
    all_weights_equal = np.all(w_eq, axis=0)  # Shape (100, 100)
    all_equal = all_weights_equal & (P_diff == 0)
    
    # dominates[j, i] is True if j dominates i
    dominates = prize_ok & weight_ok & ~all_equal  # Shape (100, 100)
    
    # is_dominated[i] is True if any j dominates i
    is_dominated = np.any(dominates, axis=0)  # Shape (100,)
    
    # Set heuristic for dominated items to a very small value
    eta[is_dominated] = 1e-9
    
    # Ensure all values are positive and finite
    eta = np.maximum(eta, 1e-9)
    
    return eta
