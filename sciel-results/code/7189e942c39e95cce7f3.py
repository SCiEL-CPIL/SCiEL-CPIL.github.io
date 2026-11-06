<<<<<<< SEARCH
def _is_valid_network(net):
    """Check if network sorts all 2^N boolean inputs using 0-1 principle."""
    n = len(net)
    if n > 4096:
        return False
    # Check all 2^N boolean inputs
    for mask in range(1 << N):
        inputs = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(net, inputs)
        # Check if sorted (non-decreasing)
        for k in range(N - 1):
            if result[k] > result[k + 1]:
                return False
    return True
=======
def _is_valid_network(net):
    """Check if network sorts all 2^N boolean inputs using 0-1 principle (Vectorized)."""
    n = len(net)
    if n > 4096:
        return False
    
    # Generate all 2^N boolean inputs as a matrix of shape (2^N, N)
    # Using bit operations for speed
    masks = np.arange(1 << N, dtype=np.int32)
    # inputs[i, j] is the j-th bit of mask i
    # We want inputs[row, col] = (masks[row] >> col) & 1
    # Efficient way: create a matrix of powers of 2 and use bitwise ops
    # Or simpler: expand dims and bitshift
    # Let's use the standard trick: 
    # inputs = ((masks[:, None] >> np.arange(N, dtype=np.int32)[None, :]) & 1).astype(np.float32)
    
    # However, for N=18, 2^18 * 18 floats is ~1.5MB, which is fine.
    # But applying the network loop in Python is still slow if done naively.
    # We can vectorize the comparator application too.
    
    # Initialize inputs matrix
    # Shape: (2^N, N)
    shifts = np.arange(N, dtype=np.int32)
    inputs = ((masks[:, None] >> shifts[None, :]) & 1).astype(np.float32)
    
    # Apply network
    # For each comparator (i, j), if inputs[:, i] > inputs[:, j], swap.
    # This can be done in-place on the NumPy array.
    for i, j in net:
        # Identify rows where i > j
        swap_mask = inputs[:, i] > inputs[:, j]
        if np.any(swap_mask):
            # Swap values in those rows
            temp = inputs[swap_mask, i].copy()
            inputs[swap_mask, i] = inputs[swap_mask, j]
            inputs[swap_mask, j] = temp
            
    # Check if sorted: for all rows, inputs[:, k] <= inputs[:, k+1] for all k
    # diff = np.diff(inputs, axis=1)
    # return np.all(diff >= 0)
    
    # Faster check:
    return np.all(inputs[:, :-1] <= inputs[:, 1:])
>>>>>>> REPLACE

