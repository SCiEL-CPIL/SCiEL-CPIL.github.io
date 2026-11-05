import random
import time
from itertools import combinations

def is_sidon(subset):
    """
    Check if a subset of integers is a Sidon set.
    A set is Sidon if all sums a_i + a_j (i <= j) are distinct.
    """
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    """
    Compute the size of the largest Sidon subset of A.
    A is a list of distinct integers.
    """
    n = len(A)
    # Check from largest size down to 1
    for k in range(n, 0, -1):
        for subset in combinations(A, k):
            if is_sidon(subset):
                return k
    return 1

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 55.0  # Safety margin
    
    best_score = 12
    best_set = list(range(12)) # Default fallback
    
    # Try to find a set with max_sidon_size < 12
    # Random search
    # We can also try some structured initial guesses
    # e.g. sets with many small differences
    
    attempts = 0
    while time.time() - start_time < max_time:
        # Generate a random set of 12 distinct integers in [0, 128]
        A = random.sample(range(129), 12)
        
        # Compute max Sidon size
        score = max_sidon_size(A)
        
        if score < best_score:
            best_score = score
            best_set = A
            # If we find a very good score (e.g., 3 or 4), we might want to keep searching 
            # to see if we can get even lower (3 is likely the min for n=12)
            # But 4 is a huge improvement over 12.
            if best_score <= 4:
                # We can continue searching for a while to see if 3 is possible, 
                # but 4 is already a strong candidate.
                # Let's run a few more iterations to be sure.
                pass
        
        attempts += 1
        # If we've done a lot of attempts, maybe increase randomness or change strategy?
        # For now, simple random sampling is robust.
        
        # Optimization: If we find a set with score 3, we can stop immediately as it's likely optimal.
        if best_score == 3:
            break

    return best_set
