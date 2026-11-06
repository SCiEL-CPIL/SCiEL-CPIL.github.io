import time
import math
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
        
    w = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    mid = n // 2
    w1 = w[:mid]
    w2 = w[mid:]
    
    # Generate history of sets for w1
    # We use sets for O(1) lookup during backtracking
    # But generating 3^15 items in a set is memory heavy?
    # 14 million ints in a set is ~500MB?
    # Let's try to be careful.
    
    def generate_history(weights):
        history = [set([0])]
        curr = set([0])
        for wi in weights:
            new_curr = set()
            # Using set union is slow.
            # Better to iterate?
            # Or use numpy to generate, then convert?
            # Let's stick to Python sets for simplicity, but check time.
            # 15 steps.
            # Step 14: 4.7 million items.
            # Step 15: 14.3 million items.
            # Creating a new set of 14 million items takes time.
            pass
        return history

    # Actually, let's use the NumPy generation for the sets, 
    # but keep the final set as a Python set for lookup.
    # For backtracking, we need the history.
    # If we keep history as NumPy arrays, we can use searchsorted.
    
    def generate_history_numpy(weights):
        history = [np.array([0], dtype=np.int64)]
        curr = np.array([0], dtype=np.int64)
        for wi in weights:
            # Create new array
            # Note: wi is int64
            new_curr = np.concatenate([curr - wi, curr, curr + wi])
            new_curr = np.unique(new_curr)
            history.append(new_curr)
            curr = new_curr
        return history

    hist1 = generate_history_numpy(w1)
    if time.time() > deadline - 20:
        return [0] * 28 + [1, -1]

    # Generate set for w2 (just the final set, no history needed for lookup, 
    # but we need history if we backtrack w2? 
    # We can backtrack w2 similarly if we keep its history.)
    # To save memory, maybe we only keep history for w1 and just the set for w2?
    # If we find a match, we backtrack w1 (using hist1) and w2 (using hist2).
    # So we need hist2 too.
    
    hist2 = generate_history_numpy(w2)
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]

    S1 = hist1[-1]
    S2 = hist2[-1]
    
    # Convert S2 to a set for fast lookup
    # 14 million items.
    set2 = set(S2.tolist())
    
    # Look for sum = 1
    target = 1
    found_s1 = None
    found_s2 = None
    
    # Iterate S1
    # S1 is 14 million items.
    # Python loop is slow.
    # We can use numpy to check?
    # We want to find s1 in S1 such that target - s1 in set2.
    # This is a set intersection problem.
    # candidates = set2 - (S1 - target)? No.
    # We want s1 such that s1 + s2 = target => s2 = target - s1.
    # So (target - S1) should intersect S2.
    
    # Vectorized:
    # targets_in_S2 = target - S1
    # We need to check which of these are in S2.
    # We can use np.isin?
    # np.isin(targets_in_S2, S2)
    # This might be slow for 14M x 14M.
    # But np.isin is optimized.
    
    # Let's try np.isin
    targets_in_S2 = target - S1
    mask = np.isin(targets_in_S2, S2)
    
    if np.any(mask):
        idx = np.argmax(mask)
        found_s1 = S1[idx]
        found_s2 = target - found_s1
    else:
        # Try -1
        target = -1
        targets_in_S2 = target - S1
        mask = np.isin(targets_in_S2, S2)
        if np.any(mask):
            idx = np.argmax(mask)
            found_s1 = S1[idx]
            found_s2 = target - found_s1
        else:
            # Fallback
            return [0] * 28 + [1, -1]

    # Backtrack w1
    def backtrack_numpy(hist, target_sum):
        coeffs = [0] * len(hist) - 1
        curr_sum = target_sum
        for i in range(len(hist) - 1, 0, -1):
            wi = w1[i-1] # Careful with indexing
            # hist[i] is the set after adding wi
            # hist[i-1] is the set before
            prev_set = hist[i-1]
            # Check curr_sum - wi in prev_set
            # Use searchsorted
            # prev_set is sorted (np.unique returns sorted)
            
            # Check if curr_sum - wi is in prev_set
            pos = np.searchsorted(prev_set, curr_sum - wi)
            if pos < len(prev_set) and prev_set[pos] == curr_sum - wi:
                coeffs[i-1] = 1
                curr_sum -= wi
            else:
                pos = np.searchsorted(prev_set, curr_sum)
                if pos < len(prev_set) and prev_set[pos] == curr_sum:
                    coeffs[i-1] = 0
                else:
                    pos = np.searchsorted(prev_set, curr_sum + wi)
                    if pos < len(prev_set) and prev_set[pos] == curr_sum + wi:
                        coeffs[i-1] = -1
                        curr_sum += wi
                    else:
                        # Should not happen
                        coeffs[i-1] = 0
        return coeffs

    # Backtrack w2
    def backtrack_numpy_w2(hist, target_sum):
        coeffs = [0] * len(hist) - 1
        curr_sum = target_sum
        for i in range(len(hist) - 1, 0, -1):
            wi = w2[i-1]
            prev_set = hist[i-1]
            
            pos = np.searchsorted(prev_set, curr_sum - wi)
            if pos < len(prev_set) and prev_set[pos] == curr_sum - wi:
                coeffs[i-1] = 1
                curr_sum -= wi
            else:
                pos = np.searchsorted(prev_set, curr_sum)
                if pos < len(prev_set) and prev_set[pos] == curr_sum:
                    coeffs[i-1] = 0
                else:
                    pos = np.searchsorted(prev_set, curr_sum + wi)
                    if pos < len(prev_set) and prev_set[pos] == curr_sum + wi:
                        coeffs[i-1] = -1
                        curr_sum += wi
                    else:
                        coeffs[i-1] = 0
        return coeffs

    c1 = backtrack_numpy(hist1, found_s1)
    c2 = backtrack_numpy_w2(hist2, found_s2)
    
    result = c1 + c2
    
    # Verify
    total = Fraction(0, 1)
    for i, c in enumerate(result):
        if c != 0:
            total += Fraction(c, i + 1)
            
    if total == 0:
        return [0] * 28 + [1, -1]
        
    return result
