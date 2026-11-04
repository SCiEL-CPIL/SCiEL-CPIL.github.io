import numpy as np

N = 100

def construct():
    """
    Construct the sequence of step heights.
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    
    This implementation uses a local search on a binary sequence (Ising spins)
    to minimize the maximum overlap C_k. The sequence is constrained to have
    exactly N/2 ones and N/2 zeros. We perform pair-flips (0->1 and 1->0)
    to maintain the sum constraint, accepting moves that reduce the peak overlap.
    """
    # Initialize with a random balanced binary sequence
    # Using a fixed seed for reproducibility in evaluation? 
    # The task doesn't specify, but deterministic is safer.
    # However, random start might be better to avoid local minima.
    # Let's use a simple deterministic pattern first, then optimize.
    # Or just random with a fixed seed.
    rng = np.random.default_rng(42)
    
    # Start with 50 zeros and 50 ones
    h = np.zeros(N, dtype=int)
    # Randomly choose 50 positions to be 1
    ones_indices = rng.choice(N, size=N//2, replace=False)
    h[ones_indices] = 1
    
    # Convert to float for return, but keep int for logic
    h_float = h.astype(float)
    
    # Function to compute C_k
    def compute_c(h):
        # h is 0/1 array
        # C_k = sum h_i (1 - h_{i+k})
        c = np.zeros(N)
        for k in range(N):
            # np.roll(h, k) shifts h to the right by k.
            # h_{i+k} is h shifted by -k?
            # If h = [a, b, c], k=1.
            # h_0 (1-h_1) + h_1 (1-h_2) + ...
            # np.roll(h, -k) gives h_{i+k} at index i?
            # roll(h, 1) = [last, first, second...] -> h_{i-1}.
            # roll(h, -1) = [second, third, ..., first] -> h_{i+1}.
            shifted = np.roll(h, -k)
            c[k] = np.sum(h * (1 - shifted))
        return c
    
    c = compute_c(h)
    max_c = np.max(c)
    
    # Local search
    # We need lists of indices with 0 and 1
    zeros = np.where(h == 0)[0]
    ones = np.where(h == 1)[0]
    
    # We will iterate a few times
    # To speed up, we can precompute contributions?
    # Or just recompute C_k array in O(N^2) for each candidate?
    # N=100, N^2=10000. 2500 candidates * 10000 = 25M ops per full scan.
    # This might be slow in pure Python loops.
    # Let's use the incremental update formula derived above.
    
    # Current C array
    # h is 0/1
    
    def delta_c(h, i, j, k):
        # i is 0->1, j is 1->0
        # Formula: (1-h[i+k]) - h[i-k] - (1-h[j+k]) + h[j-k]
        # Indices mod N
        return (1 - h[(i+k) % N]) - h[(i-k) % N] - (1 - h[(j+k) % N]) + h[(j-k) % N]

    # We want to minimize max(C)
    # Current max is max_c.
    
    # Optimization:
    # Instead of checking all 2500 pairs, we can be smarter.
    # But 2500 is small.
    
    for _ in range(50): # 50 full passes
        improved = False
        # Shuffle zeros and ones to avoid bias?
        # For determinism, just iterate.
        for i in zeros:
            for j in ones:
                # Calculate new max C if we flip i (0->1) and j (1->0)
                # We need to update C array.
                # Delta for each k
                # We can vectorize the delta calculation for all k
                # i, j fixed.
                # delta_k = (1 - h[i+k]) - h[i-k] - (1 - h[j+k]) + h[j-k]
                
                # h[i+k] is np.roll(h, -k)[i]? No.
                # Let's just compute the vector of deltas.
                # k ranges 0..N-1
                # h[i+k] for all k is h shifted by -i?
                # np.roll(h, -i) has h[i+k] at index k?
                # roll(h, -i)[k] = h[k+i]. Yes.
                h_shift_i = np.roll(h, -i)
                h_shift_neg_i = np.roll(h, i)
                h_shift_j = np.roll(h, -j)
                h_shift_neg_j = np.roll(h, j)
                
                deltas = (1 - h_shift_i) - h_shift_neg_i - (1 - h_shift_j) + h_shift_neg_j
                
                new_c = c + deltas
                new_max = np.max(new_c)
                
                if new_max < max_c:
                    # Accept move
                    h[i] = 1
                    h[j] = 0
                    c = new_c
                    max_c = new_max
                    improved = True
                    # Update zeros and ones lists
                    # Remove i from zeros, add to ones
                    # Remove j from ones, add to zeros
                    # This is expensive to do with lists.
                    # Re-initialize lists?
                    # Or just break and restart the loop?
                    # Since N is small, re-initializing is fine.
                    zeros = np.where(h == 0)[0]
                    ones = np.where(h == 1)[0]
                    break
            if improved:
                break
        if not improved:
            break
            
    # Convert back to float
    return h.astype(float)
