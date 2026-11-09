import time
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to stay under 90s limit

    n = 30
    # We need to find coefficients c[0..29] in {-1, 0, 1}
    # such that sum(c[i] / (i+1)) != 0 and |sum| is minimized.
    
    # Strategy: Meet-in-the-middle
    # Split into two halves: first 15 terms, last 15 terms.
    # Each half has 3^15 = 14348907 possibilities.
    # This is feasible in Python if optimized, but might be tight.
    # 3^15 ~ 14.3M. Storing 14.3M Fractions might be memory heavy.
    # Let's check memory: A Fraction object is small, but 14M of them...
    # Better approach: Use scaled integers.
    # LCM of 1..15 and 1..30.
    # LCM(1..30) is large but fits in 64-bit integer?
    # LCM(1..30) = 2329089562800. This fits in a 64-bit signed integer (max ~9e18).
    
    # Actually, we don't need the global LCM.
    # We can just use the LCM of the first half and LCM of the second half separately?
    # No, we need to compare sums. 
    # Let L = LCM(1, 2, ..., 30).
    # Then the sum is (sum c[i] * L/(i+1)) / L.
    # Minimizing |sum| is equivalent to minimizing |sum c[i] * L/(i+1)|.
    # Let W[i] = L / (i+1). These are integers.
    # We want to find c[i] in {-1, 0, 1} such that S = sum c[i]*W[i] is nonzero and |S| is minimized.
    
    # Calculate L = LCM(1..30)
    from math import gcd
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    # Weights W[i] for i=0..29 correspond to denominator i+1
    # W[i] = L // (i+1)
    W = [L // (i+1) for i in range(n)]
    
    # Split into two halves
    mid = n // 2  # 15
    W1 = W[:mid]
    W2 = W[mid:]
    
    # Generate all possible sums for the first half
    # There are 3^15 possibilities.
    # We will store (sum_value, coefficients_tuple)
    # To save memory and time, we can just store the sum values and sort them?
    # But we need to reconstruct the coefficients.
    # Storing 14M tuples of 15 ints is too much memory.
    # Alternative: Store only the sums in a sorted list?
    # If we just store sums, we can find the min non-zero sum from the combined set.
    # But how do we recover the coefficients?
    # We can do a second pass: once we know the target sum S_min, we can search for the specific combination again.
    # OR: Store (sum, index) where index encodes the coefficients in base 3.
    
    # Let's try storing (sum, index)
    # index is an integer from 0 to 3^15 - 1.
    # From index, we can decode coefficients.
    
    half1_sums = []
    for idx in range(3 ** mid):
        # Decode idx into base 3 digits
        # This is O(mid) per sum, so 14M * 15 ~ 210M operations. Might be slow in Python.
        # Optimization: Generate sums iteratively.
        pass
    
    # Better: Iterative generation to avoid decoding overhead in the inner loop.
    # We can use a recursive generator or iterative stack.
    # Or even better: numpy? No, pure Python might be faster for this specific logic if optimized.
    # Let's try a simple recursive generator that appends to a list.
    
    def gen_sums(weights, depth, current_sum, current_idx, results):
        if depth == len(weights):
            results.append((current_sum, current_idx))
            return
        w = weights[depth]
        # c = -1
        gen_sums(weights, depth + 1, current_sum - w, current_idx, results)
        # c = 0
        gen_sums(weights, depth + 1, current_sum, current_idx, results)
        # c = 1
        gen_sums(weights, depth + 1, current_sum + w, current_idx, results)
        
    # Recursion depth 15 is fine.
    # But function call overhead in Python is high. 3^15 calls * 3 calls per level...
    # Total nodes in tree is sum_{k=0}^{15} 3^k = (3^16 - 1)/2 ~ 21.5M.
    # This will take several seconds in pure Python.
    
    # Let's try an iterative approach with explicit stack or just a loop with base-3 decoding optimized.
    # Actually, base-3 decoding is quite fast if done with bit operations? No, base 3 is not power of 2.
    
    # Let's try a different approach: 
    # Generate sums for half 1, sort them.
    # Generate sums for half 2, sort them.
    # Then find the pair (s1, s2) such that s1+s2 != 0 and |s1+s2| is minimized.
    
    # To recover coefficients, we need to know which indices produced s1 and s2.
    # So we store (sum, index).
    
    # Let's implement the generation efficiently.
    
    def generate_half_sums(weights):
        m = len(weights)
        sums = [0] * (3 ** m)
        # Fill sums array
        # This is a standard "enumerate all subsets with signs" problem.
        # We can do it iteratively.
        # Start with [0]
        # For each weight w, new_sums = [x-w, x, x+w for x in current_sums]
        # This uses 3x memory and time at each step.
        # Step 1: 1 elem
        # Step 2: 3 elems
        # ...
        # Step 15: 3^15 elems
        # Total operations: sum 3^k = 21.5M. This is acceptable.
        
        current = [(0, 0)]  # (sum, index)
        # Index encoding: 
        # If we process weight at position 'pos' (0-indexed from left? or right?),
        # Let's say we build the index from left to right.
        # idx = idx * 3 + digit
        # But wait, if we start with idx=0 and do idx*3+digit, the first weight corresponds to the most significant digit.
        # When we decode later, we need to know the order.
        
        # Let's stick to the simple recursive approach but optimize it.
        # Actually, 21.5M function calls might be too slow (approx 2-5 seconds in PyPy, maybe 10-20s in CPython).
        # Deadline is 80s. It should fit.
        
        # Let's try the iterative expansion.
        # List of (sum, index)
        # Initial: [(0, 0)]
        # For each weight w at position i (0 to m-1):
        #   new_list = []
        #   for (s, idx) in current:
        #       new_list.append((s - w, idx * 3 + 0))  # c = -1? Let's define 0->-1, 1->0, 2->1? Or -1,0,1?
        #       # Let's use 0,1,2 for base-3 digits.
        #       # 0 -> -1
        #       # 1 -> 0
        #       # 2 -> 1
        #       new_list.append((s, idx * 3 + 1))
        #       new_list.append((s + w, idx * 3 + 2))
        #   current = new_list
        
        # This creates 3x the list each time.
        # Memory: 3^15 tuples. Each tuple has 2 ints.
        # 14M tuples * ~56 bytes/tuple (Python overhead) ~ 800MB? 
        # This might be too much memory.
        # Alternative: Store sums and indices in two separate arrays? Or use numpy?
        # Or, just store the sums in a list of ints, and sort them. 
        # Then, to find the optimal pair, we scan.
        # But we need the coefficients.
        # We can store the index along with the sum.
        # To save memory, we can store the index as an int.
        
        # Let's check if 14M ints is okay. Yes.
        # 14M tuples is likely okay too (14M * 16 bytes for tuple + 2 ints... in CPython, small ints are cached, but tuple overhead is high).
        # Let's try to be safe and use two lists: sums and indices.
        
        sums_list = [0]
        indices_list = [0]
        
        for i, w in enumerate(weights):
            new_sums = []
            new_indices = []
            # Pre-allocate? No, just append.
            for j in range(len(sums_list)):
                s = sums_list[j]
                idx = indices_list[j]
                new_sums.append(s - w)
                new_indices.append(idx * 3 + 0)
                new_sums.append(s)
                new_indices.append(idx * 3 + 1)
                new_sums.append(s + w)
                new_indices.append(idx * 3 + 2)
            sums_list = new_sums
            indices_list = new_indices
            
            # Check time
            if time.time() > deadline - 20:
                break
        
        return sums_list, indices_list
    
    # Generate first half
    sums1, idx1 = generate_half_sums(W1)
    
    if time.time() > deadline - 10:
        # Fallback to seed solution
        return [0] * 28 + [1, -1]
    
    # Generate second half
    sums2, idx2 = generate_half_sums(W2)
    
    if time.time() > deadline - 5:
        return [0] * 28 + [1, -1]
    
    # Sort sums1 and sums2
    # We need to find min |s1 + s2| where s1+s2 != 0.
    # Sort sums1.
    # For each s2 in sums2, binary search in sums1 for -s2.
    # Check neighbors.
    
    # To keep track of which index corresponds to which sum, we need to sort pairs (sum, index).
    pairs1 = list(zip(sums1, idx1))
    pairs1.sort(key=lambda x: x[0])
    sorted_sums1 = [p[0] for p in pairs1]
    sorted_idx1 = [p[1] for p in pairs1]
    
    pairs2 = list(zip(sums2, idx2))
    pairs2.sort(key=lambda x: x[0])
    sorted_sums2 = [p[0] for p in pairs2]
    sorted_idx2 = [p[1] for p in pairs2]
    
    best_val = float('inf')
    best_idx1 = -1
    best_idx2 = -1
    
    import bisect
    
    # Iterate over second half sums
    # To speed up, we can iterate over unique sums? Or just all.
    # 14M iterations with binary search (log 14M ~ 24) is ~ 350M operations. Too slow in Python.
    
    # Alternative: Two-pointer technique?
    # Sort both. 
    # We want to minimize |a_i + b_j|.
    # This is similar to finding the pair with minimum sum (closest to 0).
    # Standard algorithm:
    # i = 0, j = len(b)-1
    # While i < len(a) and j >= 0:
    #   s = a[i] + b[j]
    #   if s == 0: check if it's the only one? No, we want min non-zero.
    #   if |s| < best_val: best_val = |s|, record indices
    #   if s > 0: j -= 1
    #   else: i += 1
    
    # This is O(N log N) for sorting + O(N+M) for the scan.
    # N = 3^15 ~ 14M.
    # The scan is linear. This is very fast!
    
    i = 0
    j = len(sorted_sums2) - 1
    n1 = len(sorted_sums1)
    n2 = len(sorted_sums2)
    
    # We need to handle the case where sum is 0.
    # If s == 0, we should not update best_val if best_val is inf, but we should look for non-zero.
    # Actually, if s==0, we can move either i or j. Moving j down might find a non-zero close to 0.
    # Moving i up might also find one.
    # Standard "closest to zero" with non-zero constraint:
    # If s == 0, we update best_val to 0? No, must be non-zero.
    # So if s == 0, we don't update best_val. But we should continue searching.
    # If s > 0, we decrease j to reduce sum.
    # If s < 0, we increase i to increase sum.
    
    # One nuance: if s == 0, there might be another combination with s = 1 or s = -1.
    # By moving j down, we reduce b_j, so s decreases (becomes negative).
    # By moving i up, we increase a_i, so s increases (becomes positive).
    # Both directions are valid. We just need to ensure we don't miss the minimum.
    # The two-pointer method for "closest to 0" works correctly for minimum absolute value.
    # If s == 0, and we want non-zero, we should explore both neighbors?
    # Actually, if s == 0, the next best candidates are either slightly positive or slightly negative.
    # The two-pointer step:
    # if s > 0: j -= 1
    # elif s < 0: i += 1
    # else: # s == 0
    #       # We can't use this. We need to move to find non-zero.
    #       # If we move j down, s becomes < 0.
    #       # If we move i up, s becomes > 0.
    #       # Which one to choose?
    #       # It doesn't matter for finding the *global* minimum, as long as we continue scanning.
    #       # However, standard two-pointer assumes we are minimizing |sum| without the != 0 constraint.
    #       # With != 0, if s == 0, we should probably check the neighbors of this 0-sum.
    #       # But the two-pointer scan will naturally visit other pairs.
    #       # Let's just move j down. It will eventually find a non-zero sum if one exists closer than the current best.
    
    while i < n1 and j >= 0:
        s = sorted_sums1[i] + sorted_sums2[j]
        if s != 0:
            abs_s = abs(s)
            if abs_s < best_val:
                best_val = abs_s
                best_idx1 = sorted_idx1[i]
                best_idx2 = sorted_idx2[j]
                if best_val == 1:
                    # Can't get better than 1 (since sums are integers)
                    break
        # Move pointer
        if s > 0:
            j -= 1
        elif s < 0:
            i += 1
        else:
            # s == 0
            # To avoid infinite loop or missing other zeros? No, we don't care about zeros.
            # We should move to find non-zero.
            # Moving j down makes s smaller (more negative).
            # Moving i up makes s larger (more positive).
            # Let's move j down.
            j -= 1
    
    if best_val == float('inf'):
        # No non-zero sum found? Impossible since we can pick all 1s.
        # But maybe we ran out of time or something.
        return [0] * 28 + [1, -1]
        
    # Reconstruct coefficients from indices
    # idx1 encodes coeffs for W1 (first 15 terms)
    # idx2 encodes coeffs for W2 (last 15 terms)
    # Encoding: digit 0 -> -1, 1 -> 0, 2 -> 1
    # The index was built as idx = idx * 3 + digit for each weight in order.
    # So the first weight processed corresponds to the most significant digit.
    # Wait, in generate_half_sums:
    # for i, w in enumerate(weights):
    #   for j ...:
    #      idx_new = idx_old * 3 + digit
    # This means weights[0] is the MSB.
    # When decoding:
    # temp = idx
    # for k in reversed(range(m)):
    #    digit = temp % 3
    #    temp //= 3
    #    coeff[k] = digit - 1
    
    coeffs = [0] * n
    
    # Decode idx1
    temp = best_idx1
    for k in reversed(range(mid)):
        digit = temp % 3
        temp //= 3
        coeffs[k] = digit - 1
        
    # Decode idx2
    temp = best_idx2
    # Note: weights in W2 are W[mid:]
    # The index best_idx2 encodes coeffs for W2 in the same way.
    # So coeff[mid + k] corresponds to W2[k] which is W[mid+k]
    for k in reversed(range(n - mid)):
        digit = temp % 3
        temp //= 3
        coeffs[mid + k] = digit - 1
        
    # Verify
    # sum_val = sum(coeffs[i] / (i+1) for i in range(n))
    # if sum_val == 0:
    #     return [0] * 28 + [1, -1]
    
    return coeffs

# EVOLVE-BLOCK-END
