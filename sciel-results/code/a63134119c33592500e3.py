import random
import itertools
import time
import sys

def _max_sidon_subset_size(S):
    """
    Compute the maximum size of a subset of S where all unordered pair sums
    (including diagonals, i.e., a+a) are unique.
    This is the "Sidon" property with self-pairs included.
    """
    S = list(S)
    n = len(S)
    
    # If n is small, brute force all subsets
    # We want the largest k such that there exists a subset of size k with unique pair sums.
    # Pair sums are for i <= j.
    
    # Optimization: check from largest possible size down.
    # The maximum possible size is n.
    
    def is_sidon(subset):
        seen = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in seen:
                    return False
                seen.add(s)
        return True
    
    # Bitmask approach for n <= 20 or so? 2^12 = 4096, very fast.
    # Let's precompute all subsets of size k for k from n down to 1.
    
    # For n=12, iterating over all subsets is 2^12 = 4096.
    # For each subset, checking Sidon is O(k^2).
    # This is extremely fast.
    
    # Let's just iterate all subsets and find the max size that is Sidon.
    max_size = 1
    # Iterate by size descending to potentially break early?
    # Or just check all.
    
    # Actually, checking from n down to 1 is better if the max size is close to n.
    # But if the max size is small, we waste time.
    # However, n=12 is tiny.
    
    # Let's use bitmask for speed.
    # But itertools.combinations is fine too.
    
    # Let's try to find the maximum k.
    for k in range(n, 0, -1):
        found = False
        for subset in itertools.combinations(S, k):
            if is_sidon(subset):
                found = True
                break
        if found:
            return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Initial candidate: random set
    # Or a structured set?
    # Let's start with a random set.
    
    best_set = None
    best_size = 13  # Max possible is 12, so 13 is infinity
    
    # Try a few initial random sets
    for _ in range(5):
        candidate = random.sample(range(129), 12)
        sz = _max_sidon_subset_size(candidate)
        if sz < best_size:
            best_size = sz
            best_set = candidate.copy()
            if best_size <= 3:
                return best_set
    
    # Local search
    current_set = best_set.copy()
    current_size = best_size
    
    # If current size is already low, we can stop early
    if current_size <= 3:
        return current_set
    
    for _ in range(50000):
        if time.time() > deadline:
            break
        
        # Pick a random index to swap
        i = random.randint(0, 11)
        old_val = current_set[i]
        
        # Try to replace with a random value
        # Or try to swap with another element?
        # Replacing with a random value is a random walk in the space of sets.
        
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val in current_set and new_val != old_val:
            continue
            
        current_set[i] = new_val
        new_size = _max_sidon_subset_size(current_set)
        
        if new_size < current_size:
            current_size = new_size
            if current_size < best_size:
                best_size = current_size
                best_set = current_set.copy()
                if best_size <= 3:
                    break
        else:
            # Revert
            current_set[i] = old_val
    
    return best_set

# To be safe, let's add a small deterministic improvement if we have time.
# But the random search should be sufficient for n=12.
# Let's add a bit more intelligence: prefer values that are far apart?
# No, keep it simple.

# One more thing: The problem asks for 12 distinct integers in [0, 128].
# The evaluator checks the largest Sidon subset.
# If we find a set where the largest Sidon subset is 4, that's good.
# Can we get 3?
# A set of 3 elements {a, b, c} is Sidon if a+a, a+b, a+c, b+b, b+c, c+c are all distinct.
# a+a != a+b => a != b (true)
# a+a != a+c => a != c (true)
# a+a != b+b => 2a != 2b => a != b (true)
# a+a != b+c => 2a != b+c
# a+a != c+c => a != c (true)
# a+b != a+c => b != c (true)
# a+b != b+b => a != b (true)
# a+b != b+c => a != c (true)
# a+b != c+c => a+b != 2c
# a+c != b+b => a+c != 2b
# a+c != b+c => a != b (true)
# a+c != c+c => a != c (true)
# b+b != b+c => b != c (true)
# b+b != c+c => b != c (true)
# b+c != c+c => b != c (true)
# So the conditions are:
# 2a != b+c
# a+b != 2c
# a+c != 2b
# These are exactly the conditions for NO 3-term arithmetic progression.
# So a 3-element set is Sidon iff it is AP-free.
# It is easy to find AP-free sets.
# So the maximum Sidon subset size can be 3 if we can make sure NO 4-element subset is Sidon.
# A 4-element set is Sidon if all pair sums are unique.
# We want to minimize the max Sidon subset size.
# If we can make it 3, that's optimal (since 1 and 2 are trivial, but 3 is the smallest non-trivial).
# Actually, can it be 2?
# A 2-element set {a, b} has sums a+a, a+b, b+b.
# These are distinct if a != b.
# So any 2 distinct elements form a Sidon set.
# So the max Sidon subset size is at least 2.
# Can we make it 2?
# That would mean NO 3-element subset is Sidon.
# i.e., EVERY 3-element subset contains an AP.
# Is it possible to have 12 numbers in [0, 128] such that every 3 of them form an AP?
# No, that's impossible. For example, take any 3 numbers. If they don't form an AP, they are Sidon.
# So we need to ensure that every 3 numbers DO form an AP? No, we need to ensure that every 3 numbers do NOT form a Sidon set, which means they DO form an AP?
# Wait.
# A 3-element set is Sidon iff it is AP-free.
# So if we want the max Sidon size to be 2, we need every 3-element subset to be NOT Sidon, i.e., to contain an AP.
# So we need a set of 12 numbers where every 3 numbers form an AP.
# This is impossible. Take a, b, c. If they form an AP, b = (a+c)/2.
# Take a, b, d. If they form an AP, d = 2b - a.
# Take a, c, d. c, d, a...
# It's very hard to satisfy for all triples.
# So 3 is likely the minimum possible max Sidon size.
# Let's aim for 3.

# To achieve max Sidon size 3, we need to ensure that NO 4-element subset is Sidon.
# A 4-element subset is Sidon if it has no repeated pair sums.
# This is a stronger condition than being AP-free.
# We want to "break" every 4-element subset.

# Let's run the search to see if we can get 3.

if __name__ == "__main__":
    # Quick test
    s = construct()
    print(f"Found set: {s}")
    print(f"Max Sidon size: {_max_sidon_subset_size(s)}")
