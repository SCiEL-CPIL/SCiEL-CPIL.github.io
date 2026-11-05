import time
import numpy as np
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin from 90s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L / d
    weights = [L // d for d in denominators]
    
    # Split into two halves
    n1 = n // 2
    n2 = n - n1
    
    left_indices = list(range(n1))
    right_indices = list(range(n1, n))
    
    left_weights = [weights[i] for i in left_indices]
    right_weights = [weights[i] for i in right_indices]
    
    def generate_sums(weights):
        """
        Generate all subset sums for the given weights.
        Returns a numpy array of sums.
        """
        # Start with empty set sum 0
        sums = np.array([0], dtype=np.int64)
        
        for w in weights:
            # Create new sums by adding w to existing sums
            new_sums = sums + w
            # Concatenate
            sums = np.concatenate([sums, new_sums])
            
            # Optional: Prune sums >= L to save memory/time, 
            # but since we need to search for < L, keeping them might be 
            # simpler if we filter later, but pruning is better for memory.
            # However, 2^24 is ~16M, which is ~128MB for int64. 
            # 2^23 is ~8M, ~64MB. 
            # Total ~200MB. This is acceptable.
            # If we prune, we lose the ability to easily map back if we don't store masks.
            # We need masks to reconstruct the solution.
            # So we must store masks or generate sums with masks.
            
            # Let's switch strategy: Generate (sum, mask) pairs? 
            # Or just generate sums, and if we find the best sum, 
            # we can backtrack? No, backtracking is hard with just sums.
            
            # Better approach: 
            # Generate all masks, compute sums vectorized.
            # For n=24, 2^24 masks is 16,777,216.
            # We can compute sums using bit manipulation or iterative addition.
            
            # Let's use the iterative addition method but keep track of masks?
            # That doubles memory. 
            # Alternative: 
            # 1. Generate all masks for left (2^23) and right (2^24).
            # 2. Compute sums for all masks.
            # 3. Sort right sums.
            # 4. For each left sum, find best right sum.
            # 5. Record best total sum and the corresponding masks.
            
            pass
            
    # Let's implement the vectorized mask sum computation which is faster in numpy.
    
    def compute_sums_from_masks(masks, weights):
        """
        Given an array of masks and a list of weights, compute the sum of weights
        for each mask.
        """
        n = len(weights)
        sums = np.zeros(len(masks), dtype=np.int64)
        for i, w in enumerate(weights):
            # Check bit i for all masks
            # (masks >> i) & 1
            bit = (masks >> i) & 1
            sums += bit * w
        return sums

    # Generate masks for left half
    size1 = 1 << n1
    masks1 = np.arange(size1, dtype=np.int64)
    sums1 = compute_sums_from_masks(masks1, left_weights)
    
    # Generate masks for right half
    size2 = 1 << n2
    masks2 = np.arange(size2, dtype=np.int64)
    sums2 = compute_sums_from_masks(masks2, right_weights)
    
    # Filter sums < L
    # We want S1 + S2 < L => S2 < L - S1
    # Since S1 >= 0, S2 must be < L.
    # Also S1 must be < L.
    
    valid1 = sums1 < L
    sums1_valid = sums1[valid1]
    masks1_valid = masks1[valid1]
    
    valid2 = sums2 < L
    sums2_valid = sums2[valid2]
    masks2_valid = masks2[valid2]
    
    # Sort sums2_valid to use searchsorted
    sorted_indices = np.argsort(sums2_valid)
    sums2_sorted = sums2_valid[sorted_indices]
    masks2_sorted = masks2_valid[sorted_indices]
    
    best_total = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # For each valid sum1, find the largest sum2 such that sum1 + sum2 < L
    # target = L - sum1
    # We want the largest sum2 < target.
    # searchsorted(sums2_sorted, target, side='left') gives the index of the first element >= target.
    # So index - 1 is the last element < target.
    
    targets = L - sums1_valid
    
    # We need to be careful with vectorization here because we need to store the best mask.
    # We can vectorize the search to find the candidate index for each sum1.
    
    # pos[i] is the index in sums2_sorted where target[i] would be inserted.
    pos = np.searchsorted(sums2_sorted, targets, side='left')
    
    # The candidate index in sums2_sorted is pos - 1.
    # If pos == 0, there is no valid sum2 (since sums2 are non-negative and target > 0, 
    # pos should be at least 1 if 0 is in sums2_sorted, which it is).
    
    valid_pos = pos > 0
    if not np.any(valid_pos):
        # Fallback: just pick the largest sum1
        # This shouldn't happen as sum2=0 is always valid
        best_idx1 = np.argmax(sums1_valid)
        best_mask1 = masks1_valid[best_idx1]
        best_mask2 = 0
        best_total = sums1_valid[best_idx1]
    else:
        # We only process valid_pos indices
        valid_indices = np.where(valid_pos)[0]
        
        # For each valid index, get the candidate sum2
        cand_pos = pos[valid_indices] - 1
        cand_sums2 = sums2_sorted[cand_pos]
        cand_masks2 = masks2_sorted[cand_pos]
        
        # Corresponding sum1
        cand_sums1 = sums1_valid[valid_indices]
        cand_masks1 = masks1_valid[valid_indices]
        
        # Total sum
        total_sums = cand_sums1 + cand_sums2
        
        # Find the max total sum
        # Note: We must ensure total_sums < L. 
        # Since cand_sums2 < L - cand_sums1, total_sums < L is guaranteed.
        
        max_idx = np.argmax(total_sums)
        best_total = total_sums[max_idx]
        best_mask1 = cand_masks1[max_idx]
        best_mask2 = cand_masks2[max_idx]
        
    # Reconstruct indices
    selected_indices = []
    
    # Left half indices
    for i in range(n1):
        if best_mask1 & (1 << i):
            selected_indices.append(i)
            
    # Right half indices
    for i in range(n2):
        if best_mask2 & (1 << i):
            selected_indices.append(n1 + i)
            
    # Verify with Fraction
    total_frac = Fraction(0)
    for idx in selected_indices:
        total_frac += Fraction(1, denominators[idx])
        
    if total_frac >= 1:
        # This should not happen if integer logic is correct, but just in case
        # Fallback to greedy
        selected_indices = []
        total = Fraction(0)
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected_indices.append(i)
                total += Fraction(1, d)
        
    return selected_indices

# EVOLVE-BLOCK-END
