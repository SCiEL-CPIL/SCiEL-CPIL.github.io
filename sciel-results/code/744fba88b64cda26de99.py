"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Uses a combined heuristic:
    1. Base: Prize / (Travel Cost + Return Cost)
    2. Refinement: Multiplicative 'Regret' factor based on the opportunity cost
       of choosing node j over the best alternative node k.
    
    The regret factor R_j is defined as P_j / max(P_k for k != j).
    This biases the heuristic towards nodes that are significantly better than
    their alternatives, effectively implementing a 'greedy with lookahead' 
    static approximation.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    prizes = np.asarray(prizes, dtype=float)
    d = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    eps = 1e-9

    if n == 0:
        return np.array([[]])

    # Precompute return costs: cost to go from j back to 0
    # return_cost[j] = d[j, 0]
    return_cost = d[:, 0].copy()
    
    # Total cost of moving i -> j -> 0 is d[i, j] + d[j, 0]
    # We want to penalize long detours.
    total_cost = d + return_cost[None, :]

    # Calculate the "Regret Ratio" for each node j
    # R_j = P_j / max(P_k for k != j)
    # If P_j is the global max, R_j = Max / SecondMax (>= 1 if unique max, = 1 if tie)
    # If P_j is small, R_j = Small / Max (< 1)
    # This makes high-value nodes more attractive relative to low-value ones,
    # especially when the gap is large.
    
    max_other_prizes = np.zeros(n)
    # Optimization: If n is small, O(N^2) is fine. For N=50, it's trivial.
    # We can also do this in O(N log N) by sorting, but N=50 is small enough.
    # However, we must be careful with ties.
    
    # Method: For each j, find max of all prizes except j.
    # We can compute global max and second max.
    # If P_j == GlobalMax, then max_other = SecondMax (or GlobalMax if multiple maxes).
    # If P_j < GlobalMax, then max_other = GlobalMax.
    
    # Let's do it robustly for N=50.
    # Create a copy of prizes with -inf at index j to find max.
    # Vectorized approach isn't strictly necessary for N=50, but let's try to be efficient.
    
    # Actually, a simpler vectorized way for "max excluding self":
    # This is tricky with vectorization without loops for "excluding self" in max.
    # Given N=50, a loop is fast and clear.
    for j in range(n):
        # Mask out j
        masked = prizes.copy()
        masked[j] = -np.inf
        max_other_prizes[j] = np.max(masked)
        
    # Handle case where all other prizes are -inf (only if N=1, but N=50 usually)
    # If max_other is -inf, set to 0 or eps.
    max_other_prizes = np.where(np.isfinite(max_other_prizes), max_other_prizes, 0.0)
    
    # Regret Ratio R_j
    # We want R_j to be high if P_j is high relative to alternatives.
    R = prizes / (max_other_prizes + eps)
    
    # Base heuristic: Prize / Total Cost
    # H_base[i, j] = P_j / (d[i, j] + d[j, 0])
    H_base = prizes[None, :] / (total_cost + eps)
    
    # Combined heuristic: Multiply by Regret Ratio
    # H[i, j] = H_base[i, j] * R_j
    # This amplifies the attractiveness of nodes that are "uniquely good" 
    # or "significantly better than the next best option".
    
    H = H_base * R[None, :]
    
    # Ensure positivity and handle non-finite values
    H = np.where(np.isfinite(H), H, 0.0)
    H = np.maximum(H, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
         # Fallback to a simpler heuristic if computation took too long (unlikely for N=50)
         return np.maximum(prizes[None, :] / (d + eps), 1e-9)
        
    return H
# EVOLVE-BLOCK-END
