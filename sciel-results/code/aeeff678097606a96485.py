import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START_TIME)

def heuristics(demands, capacity):
    demands = np.array(demands, dtype=float)
    n = len(demands)
    
    # Precompute pair sums and fit masks
    pair_sums = demands[:, None] + demands[None, :]
    fits = pair_sums <= capacity
    
    # Sort demands for two-pointer count computation
    sorted_demands = np.sort(demands)
    
    # count_after[i, j] = number of remaining items (excluding i and j) that can fit
    # in the residual space after placing item j following item i.
    # Residual space = capacity - demands[i] - demands[j]
    # We count how many items k (k != i, k != j) satisfy demands[k] <= residual.
    
    # Since demands are from 20 to 99, and capacity is 150,
    # residual is at least 150 - 99 - 99 = -48 (invalid) up to 150 - 20 - 20 = 110.
    # If residual < 20 (min demand), count is 0.
    
    # Efficient computation using searchsorted on sorted_demands
    # For each pair (i, j), residual = capacity - demands[i] - demands[j]
    # count = number of elements in sorted_demands <= residual
    # But we need to exclude i and j themselves if they fit.
    
    # Let's compute a base count matrix: base_count[i, j] = number of items k where demands[k] <= residual_ij
    # Then subtract 1 if demands[i] <= residual_ij (exclude i)
    # Then subtract 1 if demands[j] <= residual_ij (exclude j)
    # Note: if i == j, we only subtract once? But i != j in transitions typically, but let's be safe.
    # Actually, the matrix is for placing j after i, so i and j are distinct items in the sequence.
    
    residuals = capacity - pair_sums
    
    # Number of items in sorted_demands <= residual
    # searchsorted returns index where element would be inserted to maintain order
    # np.searchsorted(sorted_demands, residuals, side='right') gives count of elements <= residual
    base_counts = np.searchsorted(sorted_demands, residuals, side='right')
    
    # Now adjust for excluding i and j
    # demands_i = demands[:, None] (shape n, 1)
    # demands_j = demands[None, :] (shape 1, n)
    # exclude_i_mask[i, j] = True if demands[i] <= residuals[i, j]
    # exclude_j_mask[i, j] = True if demands[j] <= residuals[i, j]
    
    d_i = demands[:, None]
    d_j = demands[None, :]
    
    exclude_i = d_i <= residuals
    exclude_j = d_j <= residuals
    
    # count_after[i, j] = base_counts[i, j] - exclude_i[i, j] - exclude_j[i, j]
    # But if i == j, and demands[i] <= residual, we subtracted twice but should only subtract once?
    # However, in bin packing, we place distinct items. The heuristic matrix is used for transitions i -> j.
    # If i == j, it means placing the same item twice, which is impossible.
    # The mask `fits` will be true for i=j if 2*d[i] <= capacity.
    # But we never place an item twice. So the entry eta[i, i] is technically irrelevant if the algorithm never picks it.
    # However, to be safe, let's handle it.
    # If i == j, and d[i] <= residual, we subtract 2. But we only have one item i.
    # So we should subtract 1.
    # Let's create a correction matrix.
    
    count_after = base_counts.astype(float) - exclude_i.astype(float) - exclude_j.astype(float)
    
    # Correction for i == j
    diag_idx = np.diag_indices(n)
    # For diagonal elements, if d[i] <= residual, we subtracted 2, but should subtract 1.
    # So add 1 back where diag and exclude_i is True (which implies exclude_j is also True since i=j)
    diag_correction = (exclude_i[diag_idx]).astype(float)
    count_after[diag_idx] += diag_correction
    
    # Ensure counts are non-negative
    count_after = np.maximum(count_after, 0.0)
    
    # For invalid pairs (fits == False), count_after should be 0
    count_after[~fits] = 0.0
    
    # Now construct the heuristic
    # We want to reward transitions that maximize the count of future compatible items.
    # Also, we should still prefer tighter fits (less waste) to encourage bin completion.
    
    # Let's combine:
    # eta[i, j] = (count_after[i, j] + 1) * (1 / (waste^2 + 1))
    # Or: eta[i, j] = exp(count_after[i, j]) * (1 / (waste^2 + 1))
    # Or: eta[i, j] = count_after[i, j] + 1 / (waste^2 + 1)
    
    # The "bin-completion gradient" idea suggests focusing on count_after.
    # Let's use a multiplicative model where count_after is the primary driver.
    
    waste = capacity - pair_sums
    waste_factor = 1.0 / (waste**2 + 1.0)
    
    # Scale count_after to be comparable to waste_factor?
    # count_after is an integer 0 to ~100. waste_factor is ~0.01 to 1.
    # Let's normalize count_after.
    # max_count = np.max(count_after)
    # normalized_count = count_after / (max_count + 1)
    
    # Let's try: eta = (count_after + 1) * waste_factor
    # This rewards both high future compatibility and tight fit.
    
    eta = (count_after + 1.0) * waste_factor
    
    # For invalid pairs, set to 1e-9
    eta[~fits] = 1e-9
    
    # Ensure non-finite values are handled
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
