"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute mapping from sum value to index in target_sums
    # Sums of interest: 128 to 256
    sum_to_idx = {s: i for i, s in enumerate(target_sums)}
    
    # We represent A as a binary vector of length 129 (indices 0..128)
    # But for FFT, we need a larger array to avoid circular convolution artifacts.
    # Max sum is 256. So we need at least 257 bins for the convolution result.
    # Let's use size 512 for power of 2 efficiency in FFT.
    fft_size = 512
    
    def get_counts_fft(A_vec):
        """
        Compute representation counts for target sums given binary vector A_vec.
        A_vec is a numpy array of 0s and 1s of length 129.
        """
        # Pad A_vec to fft_size
        A_padded = np.zeros(fft_size)
        A_padded[:129] = A_vec
        
        # FFT
        F = np.fft.fft(A_padded)
        # Autocorrelation (which corresponds to sums of pairs)
        # |F|^2 gives the circular convolution of A with A.
        # Since we are doing linear convolution via padding, the result is in the first part.
        # The coefficient at index k in the inverse FFT corresponds to sum k.
        C = np.fft.ifft(F * F).real
        # C[k] is the number of pairs (a,b) in A such that a+b=k.
        # Note: This counts ordered pairs (a,b) and (b,a) separately if a!=b, 
        # and (a,a) once.
        # The task defines "Pair representations are unordered and permit equal summands".
        # Let's check the convention.
        # "Local pairs are unordered and include equal summands".
        # If A={0,1}, sums: 0+0=0, 0+1=1, 1+1=2.
        # Counts: 0:1, 1:1, 2:1.
        # FFT Autocorrelation of [1,1,0...] gives:
        # idx 0: 1*1 = 1
        # idx 1: 1*1 + 1*1 = 2 (0+1 and 1+0)
        # idx 2: 1*1 = 1
        # So FFT gives ordered counts.
        # We need unordered counts.
        # For sum S:
        # If S is even, let a = S/2. If a in A, count += 1.
        # For a != b, a+b=S. Ordered count includes (a,b) and (b,a). Unordered is 1.
        # So Unordered(S) = (Ordered(S) + (1 if S/2 in A else 0)) / 2 ?
        # Let's verify.
        # A={0,1}.
        # S=0: Ordered=1 (0,0). S/2=0 in A. (1+1)/2 = 1. Correct.
        # S=1: Ordered=2 (0,1),(1,0). S/2=0.5 not in A. (2+0)/2 = 1. Correct.
        # S=2: Ordered=1 (1,1). S/2=1 in A. (1+1)/2 = 1. Correct.
        
        counts = np.zeros(num_targets, dtype=np.float64)
        for i, s in enumerate(target_sums):
            ordered = C[s]
            # Check if s/2 is an integer and in A
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and A_vec[half] == 1:
                    ordered += 1
            counts[i] = ordered / 2.0
            
        # Due to floating point errors, round to nearest integer
        counts = np.round(counts).astype(np.int32)
        return counts

    def calculate_peak(counts):
        """Return the maximum multiplicity in the target interval."""
        if np.any(counts == 0):
            return 1000 + np.sum(counts == 0)
        return np.max(counts)

    # Initial Solution: 
    # Start with a set that covers the range. 
    # A simple interval like [64, 128] covers [128, 256].
    # Size 65. Peak multiplicity for interval [64,128] summing to [128,256]:
    # Sum 128: 64+64 (1)
    # Sum 129: 64+65 (1)
    # ...
    # Sum 192: 64+128, 65+127, ..., 96+96 (33 pairs? No, 64..128 is 65 elements.
    # Pairs for sum 192: 
    # 64+128, 65+127, ..., 96+96.
    # Number of pairs: 128-64+1 = 65? No.
    # a in [64, 128], b = 192-a in [64, 128].
    # 192-b in [64,128] => b in [64, 128].
    # So all a in [64, 128] work?
    # If a=64, b=128. If a=128, b=64.
    # Unordered: 
    # 64+128, 65+127, ..., 95+97, 96+96.
    # Count: 96-64+1 = 33.
    # Peak is 33.
    
    # Can we do better?
    # A Sidon-like set has low multiplicity.
    # But we need to cover [128, 256].
    
    # Let's start with a random valid set or a structured one.
    # Try a few random starts and pick the best.
    
    best_A_vec = None
    best_peak = 10000
    
    for _ in range(10):
        # Random subset of size ~50-60
        size = random.randint(50, 60)
        A_set = set(random.sample(range(129), size))
        A_vec = np.zeros(129, dtype=np.float64)
        for x in A_set:
            A_vec[x] = 1.0
        
        # Greedy repair to ensure validity
        counts = get_counts_fft(A_vec)
        if np.any(counts == 0):
            uncovered = np.where(counts == 0)[0]
            # Try adding random elements
            for _ in range(100):
                if not uncovered:
                    break
                x = random.randint(0, 128)
                if A_vec[x] == 0:
                    A_vec[x] = 1.0
                    counts = get_counts_fft(A_vec)
                    if np.any(counts == 0):
                        new_uncovered = np.where(counts == 0)[0]
                        if len(new_uncovered) >= len(uncovered):
                            A_vec[x] = 0.0
                            counts = get_counts_fft(A_vec)
                        else:
                            uncovered = new_uncovered
                    else:
                        uncovered = []
        
        peak = calculate_peak(get_counts_fft(A_vec))
        if peak < best_peak:
            best_peak = peak
            best_A_vec = A_vec.copy()
            
    current_A_vec = best_A_vec
    current_counts = get_counts_fft(current_A_vec)
    current_peak = calculate_peak(current_counts)
    
    # Coordinate Descent with FFT-based evaluation
    # We will iterate through elements, trying to swap or add/remove to reduce peak.
    
    # To speed up, we can precompute the "contribution" of each element to the counts.
    # But since FFT is fast, we can just recompute.
    # However, recomputing FFT for every single move might be slow if we do millions of moves.
    # 512 FFT is ~10us. 1e6 moves = 10s. Feasible.
    
    # Let's try a more targeted search:
    # Identify the sum with the maximum count (the "hot spot").
    # Try to remove an element involved in that sum, or add an element that spreads the load.
    
    T = 5.0
    alpha = 0.999
    
    while time.time() < deadline - 2.0 and T > 0.01:
        # Find hot spots
        if np.any(current_counts == 0):
            # Invalid, try to fix
            # Pick a random uncovered sum
            uncovered_idx = np.where(current_counts == 0)[0]
            target_sum = target_sums[random.choice(uncovered_idx)]
            # Try adding an element that helps cover this sum
            # a + b = target_sum => b = target_sum - a
            # We need to add a such that b is in A, or add b such that a is in A.
            # Or add both? No, we add one element.
            # If we add x, it forms pairs (x, y) for all y in A.
            # We want x + y = target_sum for some y in A.
            # So y = target_sum - x must be in A.
            # x must be in [0, 128] and y in [0, 128].
            candidates = []
            for x in range(129):
                if current_A_vec[x] == 0:
                    y = target_sum - x
                    if 0 <= y <= 128 and current_A_vec[y] == 1:
                        candidates.append(x)
            
            if candidates:
                x = random.choice(candidates)
                new_A_vec = current_A_vec.copy()
                new_A_vec[x] = 1.0
                new_counts = get_counts_fft(new_A_vec)
                new_peak = calculate_peak(new_counts)
                if new_peak < current_peak:
                    current_A_vec = new_A_vec
                    current_counts = new_counts
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A_vec = current_A_vec.copy()
            else:
                # No direct fix, try random add
                x = random.randint(0, 128)
                if current_A_vec[x] == 0:
                    new_A_vec = current_A_vec.copy()
                    new_A_vec[x] = 1.0
                    new_counts = get_counts_fft(new_A_vec)
                    new_peak = calculate_peak(new_counts)
                    if new_peak < current_peak:
                        current_A_vec = new_A_vec
                        current_counts = new_counts
                        current_peak = new_peak
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A_vec = current_A_vec.copy()
        else:
            # Valid, try to reduce peak
            # Find sum with max count
            max_idx = np.argmax(current_counts)
            max_count = current_counts[max_idx]
            
            # Try to reduce this count.
            # Strategy: Remove an element that contributes to this sum.
            # Or Swap: Remove an element contributing to hot spot, add one that doesn't.
            
            # Get elements in A
            A_elements = np.where(current_A_vec == 1)[0]
            
            # Randomly pick an element in A
            x = random.choice(A_elements)
            
            # Option 1: Remove x
            if len(A_elements) > 30: # Keep size reasonable
                new_A_vec = current_A_vec.copy()
                new_A_vec[x] = 0.0
                new_counts = get_counts_fft(new_A_vec)
                new_peak = calculate_peak(new_counts)
                delta = new_peak - current_peak
                if delta < 0:
                    current_A_vec = new_A_vec
                    current_counts = new_counts
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A_vec = current_A_vec.copy()
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_A_vec = new_A_vec
                        current_counts = new_counts
                        current_peak = new_peak
            
            # Option 2: Swap x with y not in A
            y_candidates = np.where(current_A_vec == 0)[0]
            if len(y_candidates) > 0:
                y = random.choice(y_candidates)
                new_A_vec = current_A_vec.copy()
                new_A_vec[x] = 0.0
                new_A_vec[y] = 1.0
                new_counts = get_counts_fft(new_A_vec)
                new_peak = calculate_peak(new_counts)
                delta = new_peak - current_peak
                if delta < 0:
                    current_A_vec = new_A_vec
                    current_counts = new_counts
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A_vec = current_A_vec.copy()
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_A_vec = new_A_vec
                        current_counts = new_counts
                        current_peak = new_peak
                        
        T *= alpha
        
        # Periodic restart if stuck?
        # For now, just let SA do its thing.
        
    # Final verification
    final_counts = get_counts_fft(best_A_vec)
    if np.any(final_counts == 0):
        # Fallback
        return list(range(64, 129))
        
    return list(np.where(best_A_vec == 1)[0])
# EVOLVE-BLOCK-END
