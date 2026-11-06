"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def _build_ap_matrix(n=256):
    """
    Build matrix A where each row represents one arithmetic progression.
    Includes all APs with length >= 2. Singletons are trivial (sum is +/-1) 
    and don't affect the max if max > 1, but we should be careful.
    Actually, the max sum is at least 1. If we find a solution with max sum 1, 
    it's perfect. If max sum > 1, singletons are irrelevant.
    We will include singletons in the matrix to be safe, or just check max(abs(signs)) which is 1.
    Let's include all APs of length >= 1 to be strictly correct, though length 1 
    just contributes 1 to the max.
    """
    rows = []
    for start in range(n):
        for step in range(1, n - start):
            # AP: start, start+step, start+2*step, ...
            # Length is at least 1.
            # The problem says "any start, positive step and length including one".
            # This implies we consider all prefixes of the infinite AP starting at 'start' with 'step'.
            # Or does it mean all finite APs contained in [1, n]?
            # "contained in [1, 256]" usually means the indices $a, a+d, \dots, a+kd$ must be $\le 256$.
            # It does not restrict the length to be maximal.
            # So for a given (start, step), we have APs of length 1, 2, ..., max_len.
            # This increases the number of constraints significantly.
            # Max length for step d is floor((n-1-start)/d) + 1.
            # Total number of such APs is sum_{d=1}^n sum_{s=0}^{n-1} (floor((n-1-s)/d) + 1).
            # This is roughly n * sum_{d=1}^n (n/d) = n^2 log n.
            # For n=256, 256^2 * 5.5 ~ 360,000 rows.
            # Matrix size 360,000 x 256. 
            # Memory: 360,000 * 256 * 8 bytes = ~700 MB. This might be too high or slow.
            
            # Let's re-read carefully: "maximum absolute signed sum on ANY arithmetic progression contained in [1,256]".
            # Usually, in these discrepancy problems, "an arithmetic progression" refers to the maximal one or any subset?
            # "any ... length including one" suggests we must check *all* lengths.
            # However, if we check the maximal AP, does it cover the smaller ones?
            # No. The sum of a prefix is not determined by the sum of the whole.
            # But notice: if the discrepancy of the maximal AP is small, are the prefixes small?
            # Not necessarily.
            
            # Let's optimize the matrix construction.
            # Instead of storing the full matrix, we can compute the sums on the fly or store indices.
            # But for speed in the inner loop of SA, we need fast dot products.
            # If we have 360k rows, A @ x takes 360k * 256 ops = 92M ops.
            # In NumPy, this is ~0.1s. 
            # SA steps might be 1000-5000. 5000 * 0.1s = 500s. Too slow.
            
            # We need a faster evaluation or a smarter set of constraints.
            # Observation: The worst discrepancies often come from short APs or specific structures.
            # If we just check maximal APs, we might miss a bad prefix.
            # However, for random signs, the sum of a prefix behaves like a random walk.
            # The max partial sum of a random walk of length L is ~ sqrt(L).
            # The sum of the whole AP is ~ sqrt(L).
            # They are of the same order.
            # Many solvers for Erdos Discrepancy only check maximal APs. 
            # The task says "any ... length".
            
            # Let's try to generate the constraints more efficiently or use a sparse representation.
            # Or, we can approximate by checking maximal APs and a sample of shorter ones?
            # No, we need to minimize the MAX. If we miss a constraint, we might think we have a better solution than we do.
            
            # Let's check the number of APs again.
            # For each step d, and start s, the APs are prefixes of the maximal AP.
            # Let L_max be the length of the maximal AP.
            # We need to check sums for lengths 1 to L_max.
            # This is equivalent to checking the cumulative sums along the AP.
            
            # Alternative approach: 
            # For each (start, step), compute the sequence of signs.
            # Compute cumulative sums.
            # Take the max absolute value of these cumulative sums.
            # This is the "discrepancy" of that progression.
            # We want to minimize the max over all (start, step).
            
            # This avoids building a huge matrix.
            # Cost per evaluation:
            # For each d in 1..n:
            #   For each s in 0..n-d:
            #     Compute cumsum of signs[s::d].
            #     Track max abs.
            #     Update global max.
            
            # Complexity:
            # Sum_{d=1}^n (n/d) * (n/d) ? No.
            # For a fixed d, there are n/d progressions.
            # Each progression has length ~ n/d.
            # Computing cumsum is O(n/d).
            # So for fixed d, cost is O(n).
            # Total cost is O(n * n) = O(n^2) = 256^2 = 65,536 operations.
            # This is VERY fast in Python if implemented with NumPy slicing.
            
            # Let's verify:
            # signs = np.array([...])
            # for d in range(1, n):
            #   for s in range(0, n, d): # Wait, start can be anything < n
            #      # Actually start s can be 0 to n-1.
            #      # But if s >= n, no elements.
            #      # If s < n, the AP is s, s+d, ...
            #      # Number of such APs for step d is n. (Starts 0 to n-1).
            #      # Wait, if start is s, the elements are s, s+d, ...
            #      # If s is large, the AP is short.
            #      # So for step d, we iterate s from 0 to n-1.
            #      # Extract signs[s::d].
            #      # This creates a view, cumsum is O(len).
            #      # Total work for step d is sum_{s=0}^{n-1} len(signs[s::d]) = sum_{s=0}^{n-1} ceil((n-s)/d).
            #      # This sum is roughly n^2 / (2d).
            #      # Total work sum_{d=1}^n n^2 / (2d) = n^2 / 2 * H_n ~ n^2 log n.
            #      # For n=256, 256^2 * 5.5 / 2 ~ 180,000 operations.
            #      # This is extremely fast. < 1ms in NumPy.
            
            pass
    return None

def _compute_max_discrepancy(signs, n=256):
    """
    Compute the maximum absolute partial sum over all arithmetic progressions.
    """
    max_disc = 0
    # signs is a numpy array of +/- 1.0
    
    for d in range(1, n):
        # For each start position s
        # We can vectorize this partially?
        # No, the lengths vary.
        # But we can iterate s.
        for s in range(n):
            # Get the slice
            # If s >= n, break (not needed since s < n)
            # The slice is signs[s::d]
            # If the slice is empty, continue
            seq = signs[s::d]
            if len(seq) == 0:
                continue
            # Compute cumulative sums
            cums = np.cumsum(seq)
            # Find max absolute value
            local_max = np.max(np.abs(cums))
            if local_max > max_disc:
                max_disc = local_max
                # Early exit if we find a very large one? 
                # Or if max_disc is already huge, we can break inner loop?
                # Not really helpful for optimization.
    return max_disc

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    # Thue-Morse t(n) is parity of number of 1s in binary representation of n.
    # We map 0 -> 1, 1 -> -1 (or vice versa)
    seq = []
    for i in range(n):
        # Count bits
        # Efficient way:
        # t = 0
        # x = i
        # while x:
        #   t ^= (x & 1)
        #   x >>= 1
        # seq.append(1 if t == 0 else -1)
        
        # Faster using numpy?
        pass
    
    # Let's just do it in Python, it's fast enough for n=256
    res = np.zeros(n)
    for i in range(n):
        # bin(i).count('1') % 2
        # 0 -> 1, 1 -> -1
        if bin(i).count('1') % 2 == 0:
            res[i] = 1.0
        else:
            res[i] = -1.0
    return res

def construct():
    t0 = time.time()
    deadline = t0 + 85  # 5s safety margin
    
    n = 256
    rng = np.random.RandomState(42)
    
    # Initialize with Thue-Morse
    signs = _thue_morse(n)
    current_score = _compute_max_discrepancy(signs, n)
    best_signs = signs.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # We will try to improve the score.
    # Score is integer (sum of integers).
    # So improvements are integer steps.
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = rng.randint(n)
        
        # Flip
        old_val = signs[i]
        signs[i] = -old_val
        
        # Calculate new score
        new_score = _compute_max_discrepancy(signs, n)
        
        # Acceptance criteria
        delta = new_score - current_score
        
        if delta < 0 or (T > 0 and rng.rand() < math.exp(-delta / T)):
            # Accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject
            signs[i] = old_val
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        
        # If stuck for too long, restart from best with higher T
        if no_improve_count > max_no_improve:
            signs = best_signs.copy()
            current_score = best_score
            T = 0.5 # Restart temperature
            no_improve_count = 0
            
        # Check time
        if time.time() > deadline - 2:
            break
            
    return [int(s) for s in best_signs]

# EVOLVE-BLOCK-END
