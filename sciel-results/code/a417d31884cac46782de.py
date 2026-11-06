import time
import sys

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    # 1. Generate 2-3-5 smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
    
    N = len(smooth_numbers)
    # smooth_numbers is sorted in ascending order
    
    # 2. Precompute incompatibility bitmask
    # incompat[i] is a bitmask of indices j where smooth_numbers[i] and smooth_numbers[j] are incompatible
    incompat = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            a = smooth_numbers[i]
            b = smooth_numbers[j]
            if b % a == 0 or a % b == 0:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
    
    # 3. Find minimal representation for each target 1..128
    best = [None] * 129
    found_count = 0
    
    # Helper: check if a set of indices forms an antichain
    def is_antichain(indices):
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if incompat[indices[i]] & (1 << indices[j]):
                    return False
        return True
    
    # For small k, use direct enumeration
    def find_min_k(target, max_k):
        """Find minimum k such that target can be expressed as sum of k distinct smooth numbers forming antichain"""
        for k in range(1, max_k + 1):
            # Use backtracking search
            result = _backtrack(target, k, 0, 0, [], 0)
            if result is not None:
                return result
        return None
    
    def _backtrack(remaining, k_left, start_idx, current_sum, selected, depth):
        """
        Find k_left smooth numbers from indices >= start_idx that sum to remaining
        and form an antichain with already selected numbers.
        Returns list of smooth numbers or None.
        """
        if k_left == 0:
            if remaining == 0:
                return selected[:]
            return None
        
        if remaining <= 0:
            return None
        
        # Pruning: even with the largest possible numbers, can we reach remaining?
        # The max we can add is sum of k_left largest smooth numbers from start_idx onward
        # But we also need antichain, so just do basic pruning
        if remaining > sum(smooth_numbers[start_idx:]):
            return None
        
        # We need at least k_left numbers, each at least 1
        if remaining < k_left:
            return None
        
        # Try each candidate from start_idx
        for idx in range(start_idx, N):
            val = smooth_numbers[idx]
            if val > remaining:
                break  # Since sorted, no more will fit
            
            # Check antichain with already selected
            compatible = True
            for sel_idx in selected:
                if incompat[idx] & (1 << sel_idx):
                    compatible = False
                    break
            if not compatible:
                continue
            
            selected.append(idx)
            result = _backtrack(remaining - val, k_left - 1, idx + 1, current_sum + val, selected, depth + 1)
            if result is not None:
                return result
            selected.pop()
        
        return None
    
    # Search for each target
    for target in range(1, 129):
        if time.time() > deadline:
            break
        
        # Try k=1 to 6 (should be enough for most)
        found = None
        for k in range(1, 7):
            if time.time() > deadline:
                break
            found = _backtrack(target, k, 0, 0, [], 0)
            if found is not None:
                break
        
        if found is not None:
            best[target] = [smooth_numbers[idx] for idx in found]
            found_count += 1
        else:
            # Try larger k if needed
            for k in range(7, 11):
                if time.time() > deadline:
                    break
                found = _backtrack(target, k, 0, 0, [], 0)
                if found is not None:
                    break
            if found is not None:
                best[target] = [smooth_numbers[idx] for idx in found]
                found_count += 1
    
    # If we ran out of time, try to fill in missing targets with a fallback
    if found_count < 128:
        for target in range(1, 129):
            if best[target] is None:
                # Use a simple greedy or fallback
                # Just try k up to 15 with backtracking
                for k in range(1, 16):
                    if time.time() > deadline:
                        break
                    found = _backtrack(target, k, 0, 0, [], 0)
                    if found is not None:
                        best[target] = [smooth_numbers[idx] for idx in found]
                        break
    
    # Construct result
    result = []
    for t in range(1, 129):
        if best[t] is not None:
            result.append(best[t])
        else:
            # This should not happen, but to be safe
            result.append([])
    
    return result
