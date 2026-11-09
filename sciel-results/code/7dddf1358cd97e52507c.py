# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008  # values are 1..1008
    
    # Compute modular inverse for each x in 1..1008
    # x * inv[x] ≡ 1 (mod 1009)
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        # Fermat's little theorem: x^(MOD-2) mod MOD for prime MOD
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Forbidden single values: 1 and 1008 (since 1*1%1009=1 and 1008*1008%1009=1)
    # So interval must not include 1 or 1008.
    # Therefore search in [2, 1007]
    
    lo_bound = 2
    hi_bound = 1007
    
    best_start = lo_bound
    best_len = 1
    
    # Two-pointer approach
    # L is the start of the window
    # R is the end of the window (exclusive in the loop, inclusive in the window [L, R-1])
    # We maintain the window [L, R-1] which is valid.
    # We try to extend R.
    
    # For efficiency, we can just iterate L and find max R for each L in O(N)
    # Total O(N^2) which is ~10^6, very fast.
    
    for L in range(lo_bound, hi_bound + 1):
        # Find max R such that [L, R] is valid
        # A window [L, R] is valid if for all x in [L, R], inv[x] is not in [L, R].
        
        # We can extend R from L to hi_bound.
        # Condition to include R in [L, R-1]:
        # 1. inv[R] < L (inverse is to the left)
        # 2. inv[R] > R (inverse is to the right)
        # 3. inv[R] < R and inv[R] >= L -> INVALID (inverse is in window)
        
        # Also, we must ensure that for all x in [L, R-1], inv[x] is not R.
        # inv[x] = R iff x = inv[R].
        # So if inv[R] >= L and inv[R] < R, then x = inv[R] is in [L, R-1].
        # This is the same as condition 3.
        
        # So the check is:
        # Can we add R?
        # Yes if inv[R] < L or inv[R] > R.
        
        # Wait, is that sufficient?
        # If inv[R] > R, it's safe for now. But what if we extend further to R' = inv[R]?
        # Then inv[R'] = R, which is in [L, R'-1]. So R' would be rejected.
        # So the greedy extension works.
        
        R = L
        # Check if L itself is valid.
        # For L in [2, 1007], inv[L] != L.
        # If inv[L] < L, it's < L (safe).
        # If inv[L] > L, it's > L (safe).
        # So L is always a valid start.
        
        # Extend R
        # We need to track the minimum inv[x] for x in [L, R] where inv[x] > x.
        # Let min_inv = inf.
        # For each new R:
        #   If inv[R] < L: safe.
        #   If inv[R] > R: safe, but update min_inv = min(min_inv, inv[R]).
        #   If inv[R] < R:
        #      If inv[R] >= L: INVALID.
        #      If inv[R] < L: safe.
        
        #   After updating min_inv, check if R < min_inv.
        #   If R >= min_inv, then some x < R has inv[x] = min_inv <= R.
        #   So x and inv[x] are both in [L, R]. INVALID.
        
        min_inv = float('inf')
        max_R = L - 1
        for R in range(L, hi_bound + 1):
            # Check if R can be added
            if inv[R] < R:
                if inv[R] >= L:
                    break # Conflict: inv[R] is in [L, R-1]
                # else inv[R] < L, safe
            elif inv[R] > R:
                min_inv = min(min_inv, inv[R])
                if R >= min_inv:
                    break # Conflict: some x < R has inv[x] <= R
            else: # inv[R] == R
                break # Self-inverse, invalid (should not happen in [2, 1007])
            
            # If we didn't break, R is added
            max_R = R
            
            # Check if we hit the deadline
            if time.time() - start_time > DEADLINE:
                break
        
        if max_R - L + 1 > best_len:
            best_len = max_R - L + 1
            best_start = L
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Verify result
    final_start = best_start
    final_end = best_start + best_len - 1
    valid = True
    for x in range(final_start, final_end + 1):
        if final_start <= inv[x] <= final_end:
            valid = False
            break
    
    if not valid:
        # Fallback
        final_start = 2
        final_len = 1
    else:
        final_len = best_len
        
    return [final_start, final_len]

# EVOLVE-BLOCK-END
