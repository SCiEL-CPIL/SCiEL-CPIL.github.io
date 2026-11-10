"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import math
from itertools import combinations

N = 12
LCM_12 = 27720  # LCM of 1..12
MAX_L = 144

def construct():
    # Precompute all subsets of {1..12}
    # We represent a subset as a bitmask or a tuple
    # We need to check Hall's condition: exists I such that |N(I, m, L)| < |I|
    
    # To speed up, we can precompute the "gap" properties or just iterate.
    # Given the low MAX_L, we iterate L from MAX_L down to 1.
    
    # For a given L, we check if there exists m in [0, LCM_12) 
    # and a subset I such that count(I, m, L) < |I|.
    
    # Optimization: 
    # If we find such an (m, I) for a specific L, then f(12, m) >= L + 1.
    # We want the max f, so we start from L = MAX_L - 1 (since f is the least L where match EXISTS,
    # if no match exists for L, f > L).
    # Wait, f(12, m) is the LEAST L such that match EXISTS.
    # If no match exists for L, then f(12, m) > L.
    # So if we find m where no match exists for L=143, f(12, m) >= 144.
    
    # We iterate L from 143 down to 0?
    # Max f is 144. So max "failure L" is 143.
    
    # Let's iterate target_L from 143 down to 1.
    # If we find an m such that NO match exists in (m, m+target_L],
    # then f(12, m) >= target_L + 1.
    # We return that m.
    
    # To check "NO match exists", we need to check if Hall's condition FAILS for some I.
    # i.e., exists I such that |N(I, m, target_L)| < |I|.
    
    # We can iterate m and check all I?
    # Or iterate I and check if min_m |N(I, m, target_L)| < |I|.
    # If min_m |N(I, m, target_L)| < |I|, then there exists an m (the one achieving the min)
    # where Hall fails for I. Thus no match exists for that m.
    
    # So for a fixed target_L:
    #   For each I:
    #     Compute min_count = min_{m in [0, LCM_12)} |N(I, m, target_L)|
    #     If min_count < |I|:
    #        Find the specific m that achieves this min.
    #        Return m.
    
    # This is efficient because we only need to do this for the largest target_L that works.
    
    subsets = []
    for mask in range(1, 1 << N):
        i_list = []
        for i in range(N):
            if mask & (1 << i):
                i_list.append(i + 1)
        subsets.append(i_list)
        
    # Sort subsets by size? Smaller subsets are more likely to fail Hall (small |I|).
    # Actually, we just need ANY I to fail.
    
    for target_L in range(MAX_L - 1, 0, -1):
        for I in subsets:
            k = len(I)
            # Check if min_{m} count(I, m, target_L) < k
            # We compute count for all m in one period?
            # Or just check if the minimum possible count is < k.
            
            # The count |N(I, m, L)| is the number of elements in S_I in (m, m+L].
            # S_I is the set of multiples of any i in I.
            
            # We can generate the positions of S_I in [0, LCM_12).
            # Then slide a window of size L.
            
            # Optimization: If L is small, we can just check gaps.
            # But let's do the general sliding window.
            
            # Generate sorted list of positions in S_I within one period
            # To avoid duplicates, use a set or boolean array
            present = [False] * LCM_12
            for i in I:
                for val in range(0, LCM_12, i):
                    present[val] = True
            
            # Get positions
            positions = [idx for idx, val in enumerate(present) if val]
            
            if not positions:
                min_count = 0
            else:
                # Sliding window to find min count
                # We need to consider circular wrap-around?
                # The window (m, m+L] can wrap around the period boundary.
                # To handle this, we can duplicate positions.
                # Or, since L < LCM_12, we can just check all m.
                
                # Efficient way:
                # Count elements in (m, m+L] for all m.
                # This is equivalent to: for each element x in S_I, it contributes 1 to count(m)
                # for m in [x-L, x-1].
                # We can use a difference array.
                
                diff = [0] * (LCM_12 + 1)
                for x in positions:
                    # x is in (m, m+L] iff m < x <= m+L
                    # => x-L <= m < x
                    # m is in [x-L, x-1]
                    start = x - L
                    end = x - 1
                    if start < 0:
                        # Wrap around
                        # Part 1: [0, end]
                        diff[0] += 1
                        diff[end + 1] -= 1
                        # Part 2: [LCM_12 + start, LCM_12 - 1]
                        diff[LCM_12 + start] += 1
                        diff[LCM_12] -= 1
                    else:
                        diff[start] += 1
                        diff[end + 1] -= 1
                        
                # Compute prefix sums to get count for each m
                min_count = LCM_12
                best_m = 0
                curr = 0
                for m in range(LCM_12):
                    curr += diff[m]
                    if curr < min_count:
                        min_count = curr
                        best_m = m
                        
            if min_count < k:
                # Found a valid m
                return best_m
                
    return 0
# EVOLVE-BLOCK-END
