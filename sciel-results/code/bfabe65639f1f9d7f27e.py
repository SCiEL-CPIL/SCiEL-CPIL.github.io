import time
import random
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Start with a Bose difference set (q=11, m=4) shifted to center around 128.
       This provides a highly structured initial set with known good properties.
    2. Use Simulated Annealing with fast NumPy-based evaluation to minimize the peak count.
    """
    deadline = time.time() + 85
    rng = random.Random(42)
    
    # --- Initialization: Bose Difference Set ---
    # For q=11, m=4, we have a (121, 11, 1) difference set.
    # Elements: {0, 1, 4, 16}
    base_diff_set = [0, 1, 4, 16]
    
    # We need A subset of [0, 128].
    # Let's try to center the sums.
    # If we take A = {x + d | d in base_diff_set, x in [0, 110]}? No, that's too big.
    # Let's just use the base set and replicate it or shift it?
    # Actually, a simple shift of the difference set might not cover [128, 256] well enough
    # or might be too sparse.
    
    # Let's try a "tiling" approach or just a random dense start if the diff set is too small.
    # A difference set of size 4 is too small to cover 129 targets with low multiplicity easily
    # unless we replicate it.
    
    # Better initialization: Random set of size ~30-40?
    # Let's analyze the density.
    # If |A| = k, number of pairs is ~k^2/2.
    # Targets = 129.
    # If we want max multiplicity M, we need k^2/2 >= 129 * M? No, that's average.
    # We want to minimize max.
    # Let's try k=20. Pairs ~ 210. Avg mult ~ 1.6.
    # Let's try k=25. Pairs ~ 325. Avg mult ~ 2.5.
    
    # Let's start with a random set of size 25 from [0, 128].
    # To ensure coverage, we might need to be careful.
    # A greedy random start:
    A = set()
    # Ensure 0 and 128 are included to help cover ends?
    # Actually, to cover 128, we need pairs summing to 128.
    # To cover 256, we need pairs summing to 256 (so 128+128).
    # So 128 must be in A.
    A.add(128)
    
    # Add random elements
    candidates = list(range(0, 128))
    rng.shuffle(candidates)
    for c in candidates:
        if len(A) >= 25:
            break
        A.add(c)
    
    # --- Fast Evaluation Function using NumPy ---
    def evaluate(A_set):
        """
        Returns (peak_multiplicity, is_valid).
        Uses convolution to count sums.
        """
        if not A_set:
            return float('inf'), False
        
        A_arr = np.array(sorted(A_set), dtype=np.int32)
        
        # Convolution of A with A gives counts of sums.
        # np.convolve(A, A) computes sum of products, but here we want counts of pairs (i,j) such that A[i]+A[j]=k.
        # This is equivalent to the coefficient of x^k in (sum x^a)^2.
        # Since we want unordered pairs with replacement, we need to be careful.
        # Convolution gives ordered pairs (i,j) and (j,i) as distinct if i!=j, and (i,i) once?
        # No, np.convolve([1,1],[1,1]) -> [1,2,1].
        # If A={0,1}, sums are 0(0+0), 1(0+1,1+0), 2(1+1).
        # Convolution of indicator vector gives:
        # v = [1, 1, 0, ...]
        # v*v -> [1, 2, 1, ...]
        # So coeff[k] is the number of ordered pairs (a,b) in AxA such that a+b=k.
        # We want unordered pairs with replacement.
        # Let N_ordered(k) be the count from convolution.
        # If k is even, a=k/2. The pair (a,a) is counted once in ordered? No, (a,a) is 1 ordered pair.
        # If k is odd, no diagonal.
        # Actually, simpler:
        # Count_unordered(k) = (N_ordered(k) + (1 if k%2==0 and k//2 in A else 0)) / 2 ?
        # Let's verify.
        # A={0,1}.
        # k=0: N=1 (0,0). Unordered: {0,0} -> 1. Formula: (1+1)/2 = 1. Correct.
        # k=1: N=2 (0,1),(1,0). Unordered: {0,1} -> 1. Formula: (2+0)/2 = 1. Correct.
        # k=2: N=1 (1,1). Unordered: {1,1} -> 1. Formula: (1+1)/2 = 1. Correct.
        
        # So:
        # 1. Compute ordered counts via convolution.
        # 2. For each k in [128, 256]:
        #    count = ordered_counts[k]
        #    if k % 2 == 0:
        #        mid = k // 2
        #        if mid in A_set:
        #             count += 1
        #    final_count = count // 2
            
        # However, doing this for every evaluation in a loop is slow if we do Python loops.
        # We can vectorize the "add 1 if even and in set" part?
        # A_set is small. We can precompute a boolean array for A.
        
        A_bool = np.zeros(129, dtype=np.int32)
        for a in A_set:
            A_bool[a] = 1
            
        # Convolution
        # We only care about sums 128 to 256.
        # Max sum is 128+128=256.
        conv = np.convolve(A_bool, A_bool, mode='full')
        
        # conv[i] is count of ordered pairs summing to i.
        # We need counts for i in 128..256.
        target_sums = np.arange(128, 257)
        
        # Check validity: all counts must be > 0?
        # Wait, if count is 0, it's invalid.
        
        ordered_counts = conv[128:257]
        
        # Determine which even sums have a midpoint in A
        # k is even if k % 2 == 0.
        # mid = k // 2.
        # We need to check if A_bool[mid] is 1.
        
        # Create a mask for even indices in target_sums
        even_mask = (target_sums % 2 == 0)
        
        # For even sums, mid is in range [64, 128].
        # We need to index A_bool at mid.
        # mid = target_sums[even_mask] // 2
        
        # We can't easily vectorize the "add 1" without creating a copy.
        # Let's just do it in numpy.
        adjusted_counts = ordered_counts.copy()
        
        if np.any(even_mask):
            mids = target_sums[even_mask] // 2
            # Check if mid is in A
            # A_bool is length 129. mids are in [64, 128].
            # So A_bool[mids] gives 1 if present.
            # We add 1 to the adjusted counts for these positions if present.
            # But we only add 1 if it's present.
            # So we add A_bool[mids] to adjusted_counts[even_mask].
            adjusted_counts[even_mask] += A_bool[mids]
            
        final_counts = adjusted_counts // 2
        
        # Check validity
        if np.any(final_counts == 0):
            return float('inf'), False
            
        peak = np.max(final_counts)
        return int(peak), True

    # --- Simulated Annealing ---
    
    current_A = set(A)
    current_peak, is_valid = evaluate(current_A)
    
    # If initial is invalid, try to fix it quickly or just start annealing with high T
    # But if invalid, score is inf. We need to get to a valid state.
    # Let's force validity in the first phase or just let SA handle it?
    # SA with inf penalty works but might get stuck.
    # Let's try to make a valid set first.
    
    # Greedy validity fix:
    # If invalid, some sums are missing.
    # We can add random elements until valid?
    # Or start with a denser set?
    # Let's restart initialization if invalid after a few tries.
    
    best_A = set(current_A)
    best_peak = current_peak
    if not is_valid:
        # Try to build a valid set greedily
        # Add random elements until valid
        attempts = 0
        while not is_valid and attempts < 50 and time.time() < deadline - 10:
            # Add a random element
            new_elem = rng.randint(0, 128)
            current_A.add(new_elem)
            current_peak, is_valid = evaluate(current_A)
            if is_valid:
                best_A = set(current_A)
                best_peak = current_peak
            attempts += 1
            
        if not is_valid:
             # Fallback: return a known valid set (e.g., range(64,129))
             # Though its peak is high.
             # range(64,129) size 65.
             # Sums 128..256.
             # 128: 64+64 (1)
             # 129: 64+65, 65+64 (2) ...
             # Peak is around 32?
             # Let's just return the seed if we can't find a valid one quickly.
             return list(range(64, 129))

    # Now we have a valid set. Start Annealing.
    T = 10.0
    cooling_rate = 0.99
    min_T = 0.01
    
    # Pre-compute neighbors? No, just random moves.
    # Moves:
    # 1. Add a random element from [0,128] not in A.
    # 2. Remove a random element from A (if size > 1).
    # 3. Swap a random element in A with a random element not in A.
    
    all_elements = list(range(129))
    
    while time.time() < deadline - 5:
        if T < min_T:
            break
            
        # Generate neighbor
        move_type = rng.random()
        neighbor_A = set(current_A)
        
        if move_type < 0.3:
            # Add
            if len(neighbor_A) < 129:
                elem = rng.choice([x for x in all_elements if x not in neighbor_A])
                neighbor_A.add(elem)
        elif move_type < 0.6:
            # Remove
            if len(neighbor_A) > 1:
                elem = rng.choice(list(neighbor_A))
                neighbor_A.remove(elem)
        else:
            # Swap
            if len(neighbor_A) > 0:
                # Pick one in, one out
                in_elem = rng.choice(list(neighbor_A))
                out_candidates = [x for x in all_elements if x not in neighbor_A]
                if out_candidates:
                    out_elem = rng.choice(out_candidates)
                    neighbor_A.remove(in_elem)
                    neighbor_A.add(out_elem)
        
        new_peak, new_valid = evaluate(neighbor_A)
        
        # Calculate delta
        if not new_valid:
            # Penalty
            delta = 1000 # Large penalty
        else:
            delta = new_peak - current_peak
            
        # Acceptance
        if delta < 0:
            # Better
            current_A = neighbor_A
            current_peak = new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(current_A)
        else:
            # Worse, accept with probability exp(-delta/T)
            if delta == 0:
                current_A = neighbor_A
                current_peak = new_peak
            else:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_A = neighbor_A
                    current_peak = new_peak
                    
        T *= cooling_rate
        
    return list(best_A)

# EVOLVE-BLOCK-END
