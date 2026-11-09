import numpy as np
import time

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    
    # Precompute edge masks
    # mask[d, e] is 1 if edge e is used by demand d in CW direction
    mask_cw = np.zeros((D, n), dtype=int)
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            mask_cw[d, e % n] = 1
            e = (e + 1) % n
            
    weights = np.array([w for _, _, w in demands], dtype=float)
    
    # Load calculation:
    # If routing vector x is 0 (CW), load = sum(weights[d] * mask_cw[d])
    # If routing vector x is 1 (CCW), load = sum(weights[d] * (1 - mask_cw[d]))
    # We can precompute base_cw_load and delta_load
    base_cw_load = (weights[:, None] * mask_cw).sum(axis=0)
    # delta[d] = load contribution if CCW - load contribution if CW
    # CCW uses edges not in CW. So CCW contribution is weights[d] * (1 - mask_cw[d])
    # delta[d] = weights[d] * (1 - mask_cw[d]) - weights[d] * mask_cw[d]
    delta = weights[:, None] * (1 - 2 * mask_cw)
    
    # Exhaustive Search
    # 2^15 = 32768 combinations, which is very small.
    # We can vectorize the evaluation of all combinations.
    
    # Generate all binary combinations of length D
    # Using bit manipulation to generate indices 0 to 2^D - 1
    # Then convert to binary strings/arrays
    
    # For efficiency, we can iterate in chunks or just use a loop if D is small, 
    # but vectorization is better.
    
    # Create a matrix of all combinations
    # This might use too much memory if D is large, but D=15 is fine.
    # 32768 * 15 floats is small.
    
    # Let's generate the combinations matrix X (N, D)
    # N = 2**D
    # This is 32768 * 15 = 491,520 elements. Very small.
    
    # We can generate X by taking the binary representation of 0..2^D-1
    # However, generating a full (32768, 15) array is fast.
    
    # Alternative: Iterate through chunks to keep memory low, but not necessary here.
    # Let's just do it directly.
    
    # To avoid creating a huge intermediate array for all combinations at once if D were larger,
    # we can process in batches. But for D=15, direct is fine.
    
    # Generate indices
    # We need a matrix X where each row is a binary vector of length D.
    # The bits correspond to the demands.
    
    # Efficient way to generate binary matrix:
    # np.unpackbits or similar?
    # Or just:
    # X = np.array([list(map(int, format(i, '015b'))) for i in range(2**D)])
    # This is slow in Python loop.
    
    # Better:
    # Create a range array
    # Use bit shifts to extract bits.
    
    max_comb = 2**D
    # We can process in batches to be safe, or just one go.
    # Let's do one go, it's tiny.
    
    # Create indices 0 to max_comb - 1
    idx = np.arange(max_comb, dtype=np.uint32)
    
    # Extract bits
    # Bit d corresponds to demand d.
    # We want X[i, d] = (idx[i] >> d) & 1
    
    # We can do this vectorized:
    # bits = np.bitwise_and(idx[:, None] >> np.arange(D)[None, :], 1)
    # This creates a (32768, 15) array.
    
    bits = np.bitwise_and(idx[:, None] >> np.arange(D)[None, :], 1).astype(float)
    
    # Calculate loads for all combinations
    # Loads = base_cw_load[None, :] + bits @ delta
    # bits is (N, D), delta is (D, n)
    # Result is (N, n)
    
    loads = base_cw_load[None, :] + bits @ delta
    
    # Find the combination with the minimum max load
    max_loads = loads.max(axis=1)
    min_max_load_idx = np.argmin(max_loads)
    
    # Retrieve the best routing vector
    best_x = bits[min_max_load_idx].astype(int)
    
    # Convert to required format: >= 0.5 for CCW, < 0.5 for CW
    # best_x is 0 or 1. 
    # 0 -> CW (0.0)
    # 1 -> CCW (1.0)
    # The requirement is: >= 0.5 routes CCW, < 0.5 routes CW.
    # So 1.0 and 0.0 are valid.
    
    return best_x.astype(float)
