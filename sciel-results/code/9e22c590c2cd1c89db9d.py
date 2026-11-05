"""Program for the suite task; uses Sidon-based construction with greedy extension."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    n = 512
    max_card = 48

    def is_mean_free(S, x, check_size):
        """Check if adding x to S keeps the set mean-free.
        check_size: list of subset sizes to check (e.g., [2,3,4,...])
        """
        S_list = list(S)
        # Check 1: Is x the mean of any subset of S of size k in check_size?
        # sum(T) = k * x for some T subset S, |T| = k
        for k in check_size:
            if k > len(S_list):
                continue
            # Use bitset DP to check if any k-subset of S sums to k*x
            target = k * x
            # DP: dp[i][j] = set of sums using j elements from first i elements
            # Use set of (count, sum) pairs
            reachable = {0: {0}}  # count -> set of sums
            for s in S_list:
                # Process in reverse to avoid using same element twice
                for cnt in sorted(reachable.keys(), reverse=True):
                    if cnt + 1 <= k:
                        for sm in reachable[cnt]:
                            new_sm = sm + s
                            if cnt + 1 not in reachable:
                                reachable[cnt + 1] = set()
                            reachable[cnt + 1].add(new_sm)
            if k in reachable and target in reachable[k]:
                return False
        
        # Check 2: For each a in S, is a the mean of any subset of (S U {x})\{a} of size k?
        # The subset can include x or not.
        # If it doesn't include x: already valid (S is mean-free).
        # If it includes x: T = T0 U {x}, T0 subset S\{a}, |T0| = k-1 >= 1
        # sum(T0) + x = k * a => sum(T0) = k*a - x
        # Let y_i = t_i - a. sum(y_i) = k*a - x - (k-1)*a = a - x
        # So check if (a - x) is a subset sum of {s - a for s in S, s != a} with exactly k-1 elements.
        
        for a in S_list:
            others = [s for s in S_list if s != a]
            for k in check_size:
                if k < 2:
                    continue
                k0 = k - 1  # number of elements from others
                if k0 > len(others):
                    continue
                if k0 == 0:
                    # T = {x}, size 1. But we need size >= 2, so k >= 2 means k0 >= 1.
                    continue
                target = a - x
                # Check if target is a subset sum of {s - a for s in others} with exactly k0 elements
                diffs = [s - a for s in others]
                reachable = {0: {0}}
                for d in diffs:
                    for cnt in sorted(reachable.keys(), reverse=True):
                        if cnt + 1 <= k0:
                            for sm in reachable[cnt]:
                                new_sm = sm + d
                                if cnt + 1 not in reachable:
                                    reachable[cnt + 1] = set()
                                reachable[cnt + 1].add(new_sm)
                if k0 in reachable and target in reachable[k0]:
                    return False
        
        return True

    # Step 1: Generate a Sidon sequence (B2 sequence) in [1, 512]
    # A simple construction: x_i = i * B + i^2 for B > 2*max_i
    # For i up to ~20, B = 40 works: x_i = 40*i + i^2
    # x_1 = 41, x_2 = 84, x_3 = 129, ..., x_20 = 840 > 512
    # Let's use a smaller range. Try B=20: x_i = 20*i + i^2
    # x_1=21, x_2=44, x_3=69, x_4=96, x_5=125, x_6=156, x_7=189, x_8=224, x_9=261, x_10=300,
    # x_11=341, x_12=384, x_13=429, x_14=476, x_15=525 > 512
    # So we get 14 elements from this.
    
    # Better: use a known efficient Sidon sequence or generate greedily.
    # Let's generate a Sidon sequence greedily in [1, 512].
    
    S = []
    used_sums = set()
    
    for x in range(1, 513):
        # Check if x can be added to maintain Sidon property
        # All pairwise sums (including x+x) must be unique
        can_add = True
        for s in S:
            if x + s in used_sums:
                can_add = False
                break
        if can_add:
            if x + x in used_sums:
                can_add = False
        if can_add:
            S.append(x)
            for s in S:
                used_sums.add(x + s)
    
    # Now S is a Sidon sequence. Let's see how many we got.
    # The greedy Sidon sequence in [1, 512] should give us around 20-25 elements.
    
    # Step 2: Greedily extend the set by adding elements that don't violate the mean-free property.
    # We check all subset sizes from 2 up to len(S)+1.
    
    current_S = list(S)
    
    for x in range(1, 513):
        if x in current_S:
            continue
        # Check if adding x keeps the set mean-free
        # We need to check all subset sizes k from 2 to len(current_S)+1
        check_sizes = list(range(2, len(current_S) + 2))
        if is_mean_free(current_S, x, check_sizes):
            current_S.append(x)
            if len(current_S) >= max_card:
                break
    
    return current_S
# EVOLVE-BLOCK-END
