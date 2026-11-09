import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Calculate LCM of denominators
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denoms)
    
    # Weights are L / d
    weights = [L // d for d in denoms]
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_subsets(weights_part):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights_part):
            # Create new lists to append to
            new_sums = []
            new_masks = []
            # Iterate over existing sums
            for j in range(len(sums)):
                s = sums[j] + w
                if s < L:
                    new_sums.append(s)
                    new_masks.append(masks[j] | (1 << i))
            
            sums.extend(new_sums)
            masks.extend(new_masks)
            
            # Time check
            if time.time() > deadline:
                break
                
        return list(zip(sums, masks))

    # Generate for both halves
    # To balance load, we can generate the smaller half first or concurrently? 
    # Sequential is fine.
    
    subsets1 = generate_subsets(w1)
    if time.time() > deadline:
        # Fallback to greedy if too slow? 
        # But 8M items should be fast.
        pass
        
    subsets2 = generate_subsets(w2)
    
    # Filter is already done in generation (s < L)
    
    # Sort subsets1 by sum descending
    subsets1.sort(key=lambda x: x[0], reverse=True)
    # Sort subsets2 by sum ascending
    subsets2.sort(key=lambda x: x[0])
    
    # Two-pointer search
    best_sum = -1
    best_mask1 = 0
    best_mask2 = 0
    
    j = len(subsets2) - 1
    
    for s1, m1 in subsets1:
        # Find largest s2 such that s1 + s2 < L
        # Since subsets2 is sorted ascending, we start from the end (largest s2)
        # and move left while sum is too large
        
        # Optimization: if s1 is already close to L, j might need to go down a lot.
        # If s1 is small, j might stay high.
        # Since s1 is iterated in descending order, the required max s2 (L - s1 - 1) is increasing.
        # So j only needs to move UP (towards larger indices in ascending sort? No, larger values are at the end).
        # Wait.
        # subsets2 is sorted ASCENDING.
        # Index 0 is smallest sum. Index -1 is largest sum.
        # We want the largest s2 < L - s1.
        # So we want the largest index j such that subsets2[j][0] < L - s1.
        
        # As s1 decreases (in the loop), L - s1 increases.
        # So the valid range for s2 expands towards larger values.
        # So j should move towards the end (larger values).
        # But we are iterating s1 in DESCENDING order.
        # So L - s1 is INCREASING.
        # So the max valid s2 is INCREASING.
        # So j should move towards larger indices (towards len(subsets2)-1).
        
        # Let's re-verify.
        # Loop: s1 from Large to Small.
        # Target: s2 < L - s1.
        # If s1 is Large, L - s1 is Small. Max valid s2 is Small. j is small.
        # If s1 is Small, L - s1 is Large. Max valid s2 is Large. j is large.
        # So j moves from small to large.
        
        # Initialize j to -1? No, we want to find the first valid one from the right?
        # Let's just use binary search for simplicity and safety, or adjust the pointer logic.
        # Binary search is O(log N) per step. 16M * 24 = 384M ops. Too slow in Python.
        # Two pointer is O(N).
        
        # Let's reset j to -1 and move it forward?
        # No, j represents the index in subsets2.
        # Let's iterate s1 in DESCENDING order.
        # Let j be the index of the largest valid s2.
        # Start with j = -1 (no valid s2 yet? or smallest?).
        # Actually, for the first (largest) s1, the valid s2s are the smallest ones.
        # For the last (smallest) s1, the valid s2s are the largest ones.
        # So j should move from 0 towards len(subsets2)-1.
        
        j = 0
        # We need to find the LAST index in subsets2 where sum < L - s1.
        # Since subsets2 is sorted ascending, we can move j forward while subsets2[j+1][0] < L - s1.
        
        # However, doing this for every s1 is O(N) if we don't reset j.
        # Since the limit L-s1 increases, j only moves forward.
        
        # Reset j to 0 for each s1? No, that's O(N^2).
        # Keep j from previous iteration?
        # If s1 decreases, L-s1 increases. The max valid index in subsets2 can only increase.
        # So we can keep j and move it forward.
        
        # But we need to find the MAXIMUM valid s2.
        # So we want the largest j such that subsets2[j][0] < L - s1.
        
        # Let's initialize j = -1.
        # For each s1:
        #   while j + 1 < len(subsets2) and subsets2[j+1][0] < L - s1:
        #       j += 1
        #   if j >= 0:
        #       candidate = s1 + subsets2[j][0]
        #       if candidate > best_sum:
        #           best_sum = candidate
        #           best_mask1 = m1
        #           best_mask2 = subsets2[j][1]
        
        # This works because j is monotonic increasing.
        
        pass # Placeholder for logic inside loop

    # Re-implementing the loop properly
    j = -1
    len2 = len(subsets2)
    
    for s1, m1 in subsets1:
        limit = L - s1
        # Move j forward as far as possible
        while j + 1 < len2 and subsets2[j + 1][0] < limit:
            j += 1
            
        if j >= 0:
            s2, m2 = subsets2[j]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_mask1 = m1
                best_mask2 = m2
                
    # If no subset found, return empty? But problem says nonempty.
    # The empty set has sum 0. 0 < 1.
    # But we need non-empty indices.
    # The best_sum will be at least 1/48 * L > 0.
    
    # Decode masks to indices
    indices = []
    
    # w1 corresponds to indices 0 to mid-1
    for i in range(mid):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    # w2 corresponds to indices mid to n-1
    for i in range(len(w2)):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    # Ensure non-empty
    if not indices:
        # Fallback: pick the smallest fraction (largest denominator)
        indices = [n - 1]
        
    return indices

# EVOLVE-BLOCK-START
# (The function construct is defined above, but the prompt asks to keep the function name and signature.
# The current program has construct inside EVOLVE-BLOCK. I will place the new code inside the block.)
# EVOLVE-BLOCK-END
