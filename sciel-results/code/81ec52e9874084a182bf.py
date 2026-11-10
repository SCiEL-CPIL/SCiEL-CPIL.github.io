import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # Convert to float array for vectorization
    d = np.array(demands, dtype=float)
    
    # Compute pairwise sums
    pair_sums = d[:, None] + d[None, :]
    
    # Initialize heuristic matrix
    # We want high values for tight fits.
    # Constraint: pair_sums <= capacity
    # Heuristic: pair_sums / (capacity - pair_sums + 1)
    # If it doesn't fit, set to 1e-9 (or 0, but task says non-positive becomes 1e-9 in ACO logic, 
    # but here we return the matrix. The task says "Heuristic entries that are not positive become 1e-9".
    # So we can just return 0 or negative for invalid, and the ACO loop will handle it?
    # "Heuristic entries that are not positive become 1e-9" is in the ACO description.
    # So we should return values that are positive for valid pairs.
    # For invalid pairs, we can return 0.
    
    valid = pair_sums <= capacity
    
    # Calculate heuristic values
    # Avoid division by zero if pair_sums == capacity (gap 0). Denominator becomes 1.
    # If pair_sums > capacity, we ignore.
    
    # We use a small epsilon to be safe, but +1 is sufficient for integer inputs.
    # Let's use +1.0 to ensure denominator >= 1.
    
    heuristic = np.zeros((n, n), dtype=float)
    
    # Calculate only for valid pairs
    # gap = capacity - pair_sums
    # h = pair_sums / (gap + 1)
    
    # Vectorized calculation
    # We can't divide where invalid easily without masking, so we do it on the masked array.
    
    # Create a safe denominator
    # For invalid pairs, we don't care about the value as long as it's <= 0.
    
    # Let's just compute for all and mask.
    # To avoid warnings, we can use np.where or calculate on a copy.
    
    # Since n=120, this is small.
    
    # Calculate potential heuristic
    # We want to avoid division by zero or negative denominators.
    # Denominator = capacity - pair_sums + 1
    # If pair_sums > capacity, denominator <= 0.
    # We only care about valid ones.
    
    denom = capacity - pair_sums + 1.0
    # For valid pairs, denom >= 1.
    # For invalid pairs, denom <= 0.
    
    # We compute the ratio only where valid.
    # To avoid runtime warnings, we can set denom to 1 where invalid, or just use np.errstate.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        h_raw = np.where(valid, pair_sums / denom, 0.0)
        
    # The task says "Heuristic entries that are not positive become 1e-9".
    # This is handled by the ACO loop. So we just return the raw positive values.
    # If h_raw is 0 for invalid, that's fine.
    
    return h_raw
