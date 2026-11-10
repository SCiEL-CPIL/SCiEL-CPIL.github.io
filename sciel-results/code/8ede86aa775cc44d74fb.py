"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Strategy: 
    1. Base heuristic: Product * Inverse Waste Squared.
    2. Global Consistency Filter: Compute a 'Competitiveness' score K = H / (RowMax + ColMax).
       This acts as a smooth surrogate for the dual constraint, penalizing pairs that are 
       not top candidates in both their row and column.
    3. MWM Boost: Identify globally optimal disjoint pairs using Hungarian algorithm.
       Apply a multiplicative boost to these pairs to guide ants to proven good pairings.
    
    This combines local pair quality, global structural consistency (via row/col max), 
    and exact global matching constraints.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    # Convert to numpy array for vectorized operations
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
        
    # Calculate sums to check if items fit together
    sums = d[:, None] + d[None, :]
    
    # Calculate products to measure "compatibility" (favoring large-large pairs)
    products = d[:, None] * d[None, :]
    
    # Calculate waste (how much space is left in the bin after placing both items)
    waste = capacity - sums
    
    # Condition: items must fit together
    fits = sums <= capacity
    
    # Use a small epsilon to avoid division by zero when waste is exactly 0
    epsilon = 1e-6
    waste_safe = np.where(fits, waste + epsilon, 1.0)
    
    # Base heuristic: product * (1 / waste_safe^2)
    H_fitting = products / (waste_safe ** 2)
    H_base = np.where(fits, H_fitting, 1e-9)
    
    # 1. Compute Row and Column Maxima for fitting pairs
    # We only care about fitting pairs for the "competitiveness" metric
    H_for_max = np.where(fits, H_base, 0.0)
    
    row_max = np.max(H_for_max, axis=1)
    col_max = np.max(H_for_max, axis=0)
    
    # Compute Competitiveness K = H / (RowMax + ColMax + epsilon)
    # This ratio is high (close to 1) if the pair is the max in both its row and col.
    # It is low if it is a poor fit in either dimension.
    denom = row_max[:, None] + col_max[None, :] + epsilon
    K = H_base / denom
    
    # Ensure K is finite and positive
    K = np.where(np.isfinite(K) & (K > 0), K, 1e-9)
    
    # 2. Maximum Weight Matching
    if n >= 2:
        W_match = H_base.copy()
        np.fill_diagonal(W_match, 0)
        
        # Ensure non-negative weights (H_base is already positive)
        cost_matrix = -W_match
        
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched = np.zeros((n, n), dtype=bool)
            for i, j in zip(row_ind, col_ind):
                if i != j and fits[i, j]:
                    matched[i, j] = True
                    matched[j, i] = True
            
            # Boost matched pairs
            # We combine the smooth Competitiveness (K) with a hard boost for matched pairs.
            # H_final = H_base * (K^gamma) * (1 + boost * matched)
            # Let's use gamma=1.0 and boost=5.0 initially.
            gamma = 1.0
            boost_factor = 5.0
            
            # Apply smooth consistency filter
            H_smooth = H_base * (K ** gamma)
            
            # Apply MWM boost
            H_final = H_smooth * (1.0 + boost_factor * matched)
            
        except Exception:
            # Fallback to base heuristic if matching fails
            H_final = H_base
    else:
        H_final = H_base
    
    # Ensure non-fitting pairs remain at 1e-9
    H = np.where(fits, H_final, 1e-9)
    
    # Ensure shape is correct
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
        
    # Check for non-finite values and sanitize if needed
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Ensure all values are positive
    H = np.where(H > 0, H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
