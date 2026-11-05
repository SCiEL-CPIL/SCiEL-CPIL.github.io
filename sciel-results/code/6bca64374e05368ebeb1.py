import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_sums = len(target_sums)
    
    # Precompute indices for even/odd sums in target range
    even_sums = [s for s in range(min_sum, max_sum + 1) if s % 2 == 0]
    odd_sums = [s for s in range(min_sum, max_sum + 1) if s % 2 == 1]
    
    # Indices in the convolution array (0..256)
    # conv[k] corresponds to sum k
    
    def evaluate(A_bool):
        """
        A_bool: numpy array of shape (129,) with 0/1 values.
        Returns (peak, uncovered_count)
        """
        # Convolution of A with itself gives ordered pair counts
        conv = np.convolve(A_bool, A_bool)
        
        # We need counts for sums 128 to 256
        # conv has length 257 (indices 0 to 256)
        
        unordered_counts = np.zeros(num_sums, dtype=np.int32)
        
        # Process even sums
        # For sum s, ordered count is conv[s].
        # Unordered count is (conv[s] + 1) // 2 if s/2 is in A, else conv[s] // 2
        # Note: conv[s] counts (a,b) and (b,a). If a!=b, they are 2 distinct ordered pairs.
        # If a==b, it's 1 ordered pair.
        # So if s is even and s/2 in A, there is 1 "diagonal" pair.
        # Total unordered = (Ordered - Diagonal + Diagonal) / 2 ? No.
        # Let D be 1 if s/2 in A else 0.
        # Ordered = 2 * (pairs with a < b) + D
        # Unordered = (pairs with a < b) + D
        # Unordered = (Ordered - D + 2*D) / 2 = (Ordered + D) / 2
        
        # We can vectorize this
        # Create an array D of length 257 where D[s] = 1 if s is even and s//2 in A
        D = np.zeros(257, dtype=np.int32)
        # s ranges from 0 to 256
        # s//2 ranges from 0 to 128
        # If s is even, s = 2*k. D[2*k] = A_bool[k]
        # We only care about s in [128, 256]
        
        # Let's just compute it for the target range
        # Indices in conv: 128 to 256
        
        # For each s in target_sums:
        # if s % 2 == 0:
        #   d = 1 if A_bool[s//2] else 0
        #   count = (conv[s] + d) // 2
        # else:
        #   count = conv[s] // 2
        
        # Vectorized approach:
        # We can create an array `d_vals` for the target sums
        d_vals = np.zeros(num_sums, dtype=np.int32)
        # For even sums, index in target_sums corresponds to s
        # Let's map s to index in target_sums
        # target_sums is arange(128, 257)
        # Index i corresponds to s = 128 + i
        
        # For even s:
        even_indices_in_target = [i for i, s in enumerate(target_sums) if s % 2 == 0]
        if even_indices_in_target:
            # Get the values of A_bool at s//2
            # s = 128 + i
            # s//2 = 64 + i//2
            # We need to be careful with integer division
            s_vals = target_sums[even_indices_in_target]
            mid_vals = s_vals // 2
            d_vals[even_indices_in_target] = A_bool[mid_vals]
            
        # Calculate unordered counts
        # unordered = (conv[s] + d_vals) // 2
        # conv[s] is even for odd s? No.
        # If s is odd, conv[s] is always even (pairs are symmetric a!=b).
        # If s is even, conv[s] = 2*k + d. (conv[s] + d) is 2*k + 2d, which is even.
        # So integer division is exact.
        
        relevant_conv = conv[128:257]
        unordered_counts = (relevant_conv + d_vals) // 2
        
        # Check coverage
        uncovered = np.sum(unordered_counts == 0)
        
        if uncovered > 0:
            # Return a large peak if not covered
            return float('inf'), uncovered
            
        peak = np.max(unordered_counts)
        return int(peak), 0

    # --- Search Strategy ---
    
    # 1. Initial Guess
    # Try to find a valid set quickly
    best_A = None
    best_peak = float('inf')
    
    # Greedy-ish random initialization
    # Start with a random set of size ~30
    # If invalid, add random elements until valid
    
    def random_valid_set():
        # Start with a set that is likely to cover
        # E.g., {0, 1, ..., 10} U {118, ..., 128}
        # This covers low and high sums well, but middle might be weak?
        # Actually, {0..128} covers everything.
        # Let's try a structured sparse set.
        
        # Try a few random sets
        for _ in range(20):
            size = random.randint(25, 35)
            A = set(random.sample(range(129), size))
            A_bool = np.zeros(129, dtype=np.int32)
            for x in A:
                A_bool[x] = 1
            peak, unc = evaluate(A_bool)
            if unc == 0:
                return A, peak
        
        # If no valid set found quickly, construct one
        A = set()
        # Add 0 and 128 to help cover extremes
        A.add(0)
        A.add(128)
        # Add random elements
        while True:
            A_bool = np.zeros(129, dtype=np.int32)
            for x in A:
                A_bool[x] = 1
            peak, unc = evaluate(A_bool)
            if unc == 0:
                return A, peak
            # Add a random element
            A.add(random.randint(0, 128))
            if len(A) > 50: # Safety
                 # Force add
                 A.add(random.randint(0, 128))

    A_init, peak_init = random_valid_set()
    best_A = A_init.copy()
    best_peak = peak_init
    
    # 2. Simulated Annealing
    # State: Set A
    # We will use a boolean array for fast evaluation
    
    A_curr = np.zeros(129, dtype=np.int32)
    for x in best_A:
        A_curr[x] = 1
        
    curr_peak, curr_unc = evaluate(A_curr)
    
    T = 10.0
    T_min = 0.1
    alpha = 0.995
    
    # Track best
    best_A_bool = A_curr.copy()
    best_peak_val = curr_peak
    
    iter_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        # Generate neighbor
        A_new = A_curr.copy()
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            candidates = np.where(A_new == 0)[0]
            if len(candidates) > 0:
                idx = random.choice(candidates)
                A_new[idx] = 1
        elif move_type < 0.6:
            # Remove
            candidates = np.where(A_new == 1)[0]
            if len(candidates) > 0:
                idx = random.choice(candidates)
                A_new[idx] = 0
        else:
            # Swap
            candidates_remove = np.where(A_new == 1)[0]
            candidates_add = np.where(A_new == 0)[0]
            if len(candidates_remove) > 0 and len(candidates_add) > 0:
                idx_rm = random.choice(candidates_remove)
                idx_add = random.choice(candidates_add)
                A_new[idx_rm] = 0
                A_new[idx_add] = 1
        
        new_peak, new_unc = evaluate(A_new)
        
        # Calculate delta
        # We want to minimize peak. Uncovered is a hard constraint penalty.
        # Let's define energy = peak + 1000 * unc
        curr_energy = curr_peak + 1000 * curr_unc
        new_energy = new_peak + 1000 * new_unc
        
        delta = new_energy - curr_energy
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            A_curr = A_new
            curr_peak = new_peak
            curr_unc = new_unc
            
            if curr_peak < best_peak_val:
                best_peak_val = curr_peak
                best_A_bool = A_curr.copy()
                
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Occasionally restart if stuck? 
        # If no improvement for a long time, maybe perturb?
        # For now, just let it cool.
        
    # Convert best_A_bool to set/list
    best_set = set(np.where(best_A_bool > 0)[0].tolist())
    
    if not best_set:
        # Fallback
        return list(range(64, 129))
        
    return sorted(best_set)
