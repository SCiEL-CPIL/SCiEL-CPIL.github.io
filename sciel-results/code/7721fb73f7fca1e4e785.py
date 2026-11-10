import time
import random

def _check_addition_fast(current_set, x):
    """
    Check if adding x to current_set creates any violation.
    Returns True if valid, False otherwise.
    """
    n = len(current_set)
    if n < 2:
        return True
    
    # 1. Check if x is the mean of any subset of current_set
    # sum(S) = k * x for S subset of current_set, |S| = k >= 2
    m = n
    max_target = m * x
    # DP for current_set
    # dp[k] is bitset of sums of k elements
    dp = [0] * (m + 1)
    dp[0] = 1
    
    # Optimization: Mask bits to keep integers small
    # Max sum of k elements is k * max(current_set)
    # We only care up to k * x
    # So for each k, we can mask to (1 << (k * x + 1)) - 1
    # But x varies. 
    # Actually, max sum we track is m * max_val.
    # But we only check targets k * x.
    # Let's just use a global mask for max_target + 1 if it's small, else no mask.
    # x <= 512, m <= 48, max_target <= 24576.
    # 1 << 24577 is a huge int, but Python handles it.
    # Shifting huge ints is slow.
    # We should mask per k.
    
    for elem in current_set:
        for k in range(m, 0, -1):
            dp[k] |= (dp[k-1] << elem)
            # Mask to relevant range for this k
            # We only care if bit k*x is set.
            # So we can mask to (1 << (k * x + 1)) - 1
            # This keeps the int small.
            mask_k = (1 << (k * x + 1)) - 1
            dp[k] &= mask_k
            
    for k in range(2, m + 1):
        target = k * x
        if (dp[k] >> target) & 1:
            return False

    # 2. Check if any a in current_set becomes the mean of a subset including x
    # Subset T from current_set \ {a} such that x + sum(T) = a * (1 + |T|)
    # sum(T) = a * (1 + |T|) - x
    # Let |T| = t. t >= 1.
    # T is subset of current_set \ {a} of size t.
    
    # To speed this up, we can pre-compute something?
    # No, we must check each a.
    
    for a in current_set:
        # Build DP for current_set \ {a}
        others = [y for y in current_set if y != a]
        m_o = len(others)
        if m_o < 1:
            continue
        
        dp_o = [0] * (m_o + 1)
        dp_o[0] = 1
        
        # Max target for t is a*(t+1) - x
        # We can mask per t.
        
        for elem in others:
            for k in range(m_o, 0, -1):
                dp_o[k] |= (dp_o[k-1] << elem)
                # Mask to relevant range for this k (which is t in the formula)
                # Target is a*(k+1) - x
                # We only care if that bit is set.
                # So mask to (1 << (a*(k+1) - x + 1)) - 1
                # But if a*(k+1) - x < 0, target is negative, bit is not set.
                # If a*(k+1) - x is large, mask is large.
                # Let's just use a reasonable mask or no mask if k is small.
                # Actually, max sum of k elements from others is k * max(others).
                # Let's just mask to (1 << (k * 512 + 1)) - 1? No, too big.
                # Let's mask to the specific target range.
                target_val = a * (k + 1) - x
                if target_val >= 0:
                    # We only need bits up to target_val
                    mask_k = (1 << (target_val + 1)) - 1
                    dp_o[k] &= mask_k
                else:
                    dp_o[k] = 0 # No valid sums can be negative? Wait, sums are positive.
                    # If target < 0, we don't care. But dp_o[k] might have bits.
                    # We can just skip masking or mask to 0.
                    # If we mask to 0, we lose info for other targets?
                    # No, for a fixed a, we check all t.
                    # But we build dp_o once for all t.
                    # So we cannot mask to a specific target per k if we reuse dp_o[k] for multiple targets?
                    # No, for a fixed a, target depends on t.
                    # So for each k (which is t), the target is different.
                    # So we can mask dp_o[k] to the range needed for that k.
                    # But we build dp_o[k] incrementally.
                    # When updating dp_o[k] with elem, we combine dp_o[k-1] shifted.
                    # So we should mask after the OR.
                    # But the mask depends on k.
                    # So:
                    # dp_o[k] |= (dp_o[k-1] << elem)
                    # target_t = a * (k + 1) - x
                    # if target_t >= 0: dp_o[k] &= (1 << (target_t + 1)) - 1
                    # else: dp_o[k] = 0 (since sums are positive, if target < 0, no match)
                    # Wait, if target < 0, we don't care about the bits.
                    # But we might need dp_o[k] for a *different* a? No, we are in loop for a.
                    # So we can safely zero it out or just not mask.
                    # If we don't mask, the int grows.
                    # Let's mask to a safe upper bound: k * max(others)
                    # Or just k * 512.
                    # Let's use k * 512 as mask bound.
                    # (1 << (k * 512 + 1)) - 1
                    # This is safe and keeps ints bounded.
                    pass # We will implement this in the code below

        # Re-implementing the inner loop with proper masking
        dp_o = [0] * (m_o + 1)
        dp_o[0] = 1
        for elem in others:
            for k in range(m_o, 0, -1):
                dp_o[k] |= (dp_o[k-1] << elem)
                # Mask to keep size manageable
                # Max sum of k elements is k * max_val.
                # We only care about target = a*(k+1) - x.
                # If target < 0, we can clear the bits? No, we need to keep dp_o[k] for potential other checks?
                # No, for this 'a', we only check one target per k.
                # So we can mask to (1 << (target + 1)) - 1 if target >= 0.
                # If target < 0, we can set dp_o[k] = 0? 
                # Wait, if we set dp_o[k] = 0, we lose the sum for future k?
                # No, dp_o[k] is used to build dp_o[k+1].
                # If we zero it out, we might lose valid sums for k+1?
                # No, if target for k is < 0, it means a*(k+1) < x.
                # For k+1, target is a*(k+2) - x.
                # If a is large, this might be positive.
                # So we cannot zero out dp_o[k] just because target_k < 0.
                # We must keep dp_o[k] to build dp_o[k+1].
                # So we must use a general mask.
                # Mask to (1 << (k * 512 + 1)) - 1 is safe.
                mask_bound = (1 << (k * 512 + 1)) - 1
                dp_o[k] &= mask_bound

        for t in range(1, m_o + 1):
            target = a * (t + 1) - x
            if 0 <= target:
                if (dp_o[t] >> target) & 1:
                    return False

    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    
    # Strategy: Multiple greedy passes with different orders
    # 1. Ascending
    # 2. Descending
    # 3. Random
    # 4. "Power of 2" biased
    
    orders = [
        list(range(1, 513)),
        list(range(512, 0, -1)),
        None, # Random
        None, # Random
        None, # Random
    ]
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart < 2:
            if restart == 0:
                remaining = list(range(1, 513))
            else:
                remaining = list(range(512, 0, -1))
        else:
            remaining = list(range(1, 513))
            random.shuffle(remaining)
            
        selected = []
        
        for x in remaining:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if _check_addition_fast(selected, x):
                selected.append(x)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
            if best_size == 48:
                break

    # Local Search on best_set
    if best_size < 48 and time.time() < deadline - 10:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add any remaining element
            for x in range(1, 513):
                if x in current:
                    continue
                if len(current) >= 48:
                    break
                if _check_addition_fast(current, x):
                    current.append(x)
                    improved = True
                    break
            
            if not improved:
                # Try swaps
                # For each a in current, try replacing with b not in current
                # If valid, and we can add more?
                # Just check if swap is valid.
                # If swap is valid, do it, and try to add more.
                # This is expensive.
                # Let's just try random swaps.
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    if len(current) == 0: break
                    i = random.randint(0, len(current) - 1)
                    a = current[i]
                    b = random.randint(1, 512)
                    if b == a or b in current:
                        continue
                    # Check if current \ {a} \cup {b} is valid
                    new_set = current[:]
                    new_set[i] = b
                    # Check validity of new_set
                    # We can reuse _check_addition_fast?
                    # No, it checks adding to a set.
                    # We can check if b is valid to add to current \ {a}
                    # AND if all a' in current \ {a} are still valid.
                    # The latter is guaranteed if current was valid?
                    # No, removing a might not break validity, but adding b might.
                    # So we just need to check if b is valid to add to current \ {a}.
                    # Because current \ {a} is a subset of current, so it is valid.
                    # Wait, is current \ {a} valid?
                    # Yes, removing elements cannot create a violation.
                    # So we just need to check if b can be added to current \ {a}.
                    others = [y for y in current if y != a]
                    if _check_addition_fast(others, b):
                        # Swap is valid
                        current[i] = b
                        improved = True
                        # After swap, try to add more
                        for x in range(1, 513):
                            if x in current:
                                continue
                            if len(current) >= 48:
                                break
                            if _check_addition_fast(current, x):
                                current.append(x)
                                improved = True
                                break
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = current[:]
                        break

    return best_set[:48]
