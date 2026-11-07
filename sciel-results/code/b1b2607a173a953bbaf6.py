import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    best_size = 0
    best_set = []
    
    # Candidates are 1..256
    # We want to maximize size.
    # Greedy strategy: try to add elements.
    # Order matters. Large elements are "safer" because they are less likely to divide others (unless others are multiples, but sums are smaller).
    # Actually, if x is large, x itself is a sum. If there is a smaller sum s, s might divide x.
    # So we need to be careful.
    
    def is_valid_antichain(sums):
        # sums is a set of ints
        if len(sums) <= 1:
            return True
        sorted_sums = sorted(sums)
        # For each a, check if any b in sums (b > a) is divisible by a
        # Optimization: only check multiples
        for i, a in enumerate(sorted_sums):
            if a == 0: continue
            # Check multiples of a
            # max sum is roughly sum of all elements
            # We can just iterate through the rest of the list
            for b in sorted_sums[i+1:]:
                if b % a == 0:
                    return False
        return True

    def get_subset_sums(elements):
        # Returns set of non-empty subset sums
        sums = {0}
        for x in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums.update(new_sums)
        sums.discard(0)
        return sums

    # Greedy with randomization
    # We'll try multiple random orders or just a fixed order with backtracking?
    # Backtracking is exponential. Greedy is O(N * 2^K).
    # N=256. K is small (maybe 10-15). 2^15 = 32768.
    # 256 * 32768 * (check cost) might be too slow if done naively for every candidate at every step.
    # But K grows slowly.
    
    # Let's try a simple greedy first to get a baseline.
    # Then try to improve with local search or different orderings.
    
    def greedy_solve(order):
        current_elements = []
        current_sums = set()
        
        for x in order:
            # Potential new sums
            # new_sums = current_sums | {s + x for s in current_sums}
            # We need to check if the new set of sums is an antichain.
            
            # Optimization: 
            # The new sums are S' = S \cup (S+x).
            # We need to check pairs within S (already valid), within S+x (same as S, valid), and cross pairs s1 in S, s2 in S+x such that s1 | s2 or s2 | s1.
            
            # If we just recompute and check, it's simpler.
            # Let's estimate cost.
            # If |S| is 1000, checking antichain is ~1000^2/2 = 500k ops.
            # 256 candidates * 500k = 128M ops. Might be tight in Python.
            
            # Better check:
            # 1. Check if any s in S divides s+x for any s in S? No, s+x is new.
            # We need to check:
            # A. For any s1, s2 in S, s1 | s2 (already known false)
            # B. For any s1, s2 in S, s1 | (s2+x)
            # C. For any s1, s2 in S, (s1+x) | s2
            # D. For any s1, s2 in S, (s1+x) | (s2+x)  --> equivalent to s1 | s2? No. (s1+x) | (s2+x) => s2+x = k(s1+x) => s2 - k s1 = (k-1)x.
            
            # Let's just use the set and check function, but optimize the check.
            
            candidate_elements = current_elements + [x]
            new_sums = get_subset_sums(candidate_elements)
            
            if is_valid_antichain(new_sums):
                current_elements.append(x)
                current_sums = new_sums
        
        return current_elements

    # Try standard greedy (descending)
    order1 = list(range(256, 0, -1))
    sol1 = greedy_solve(order1)
    if len(sol1) > best_size:
        best_size = len(sol1)
        best_set = sol1
    
    # Try randomized greedy
    # We can try to restart with different random seeds if time permits
    # Or try ascending? Unlikely to be better.
    
    # Let's try to improve best_set with a local search?
    # Swap an element with a non-element?
    # Or just try more random orders.
    
    random.seed(42)
    for i in range(50):
        if time.time() > deadline:
            break
        
        # Random permutation of a subset of candidates? 
        # Or just shuffle the full list?
        # Shuffling 256 items is cheap.
        order = list(range(1, 257))
        random.shuffle(order)
        
        # Bias towards larger numbers?
        # Actually, let's just shuffle.
        
        sol = greedy_solve(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol

    # Ensure return type is list of ints
    return [int(x) for x in best_set]

# Test with a small example mentally?
# The greedy might be weak. 
# Is there a known large set?
# Powers of 2: 1, 2, 4... sums are unique, but 1 divides all.
# If we remove 1, 2, 4... and start from 8?
# 8, 16, 32... sums are unique. Does 8 divide 16? Yes. So powers of 2 don't work if >1 element.
# We need sums that are pairwise incomparable.
# This implies the sums themselves form an antichain.
# If we pick numbers that are all congruent to 1 mod M for large M?
# Or primes?

# The greedy above is a solid start. It will find *some* set.
# The score is the size.
# Current seed is size 3.
# Greedy will likely find size > 3.

# To be safe, I will add a check for the specific constraint in the problem statement:
# "Two different subsets with the same sum still make the set invalid"
# This is handled by `get_subset_sums` returning a *set*. If two subsets have the same sum, the set size is smaller, but the validity check `is_valid_antichain` operates on the *values* of the sums.
# Wait, the problem says: "Equal sum values are counted once. Two different subsets with the same sum still make the set invalid: removing their common elements leaves two disjoint subsets with the same sum s, and their union has sum 2s, which s divides."
# My `get_subset_sums` returns a set of sums. If there are duplicate sums, they are collapsed in the set.
# However, the *invalidity* comes from the existence of the two disjoint subsets.
# If I just check if the *set* of sums is an antichain, I might miss the case where duplicate sums exist, which implies a divisibility relation ($s | 2s$).
# BUT, if duplicate sums exist, then $s$ is in the set of sums. And $2s$ is also a sum (union of the two subsets).
# So $s$ and $2s$ are both in the set of sums.
# Since $s \neq 2s$ (as $s>0$), and $s | 2s$, the antichain check on the *set of sums* will catch this!
# Because if $s$ and $2s$ are both in the set, `is_valid_antichain` will return False.
# So checking if the set of distinct sums is an antichain is sufficient and necessary.

# One edge case: $s=0$? No, non-empty subsets.
# So the logic holds.

# Let's run a quick mental check on the greedy.
# If it finds size 5, that's a good start.
# I will add a more sophisticated search if time allows in the code, but the simple greedy is robust.

# To ensure we beat the seed (size 3), this should be easy.
# To maximize, we can try to run the greedy multiple times with different "biases".
# E.g., sort by $x \pmod k$?

# I will stick to the randomized greedy as it's simple and effective for this constraint satisfaction problem.
