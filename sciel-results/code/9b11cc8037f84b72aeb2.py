"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Construct 12 distinct positive integers <= 1048576 with all 4096 subset sums distinct.
    Minimize the largest integer using greedy superincreasing construction.
    
    For distinct subset sums, a sufficient condition is that the sequence is superincreasing:
    a[i] > sum(a[0:i]) for all i. This gives the tightest possible lower bound on each element.
    """
    # Greedy superincreasing: a[0]=1, a[i] = sum(a[0:i]) + 1
    # This gives: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # Largest = 2048, which is well within 1048576
    
    # But we can potentially do better by not requiring strict superincreasing
    # and using local search to find a valid sequence with smaller max element.
    
    # Start with the greedy superincreasing solution
    seq = [1]
    for i in range(11):
        seq.append(sum(seq) + 1)
    
    best_seq = seq[:]
    best_max = max(seq)
    
    # Try to improve via local search
    deadline = time.time() + 85  # 85 second deadline (5 sec safety margin)
    
    def all_subset_sums_distinct(arr):
        """Check if all 2^n subset sums are distinct."""
        n = len(arr)
        sums = set()
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += arr[i]
            if s in sums:
                return False
            sums.add(s)
        return True
    
    # Local search: try to decrease elements while maintaining validity
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Try to decrease a random element
        idx = random.randint(0, 11)
        if idx == 0:
            continue  # Can't decrease a[0] below 1
        
        current_val = seq[idx]
        if current_val <= 1:
            continue
        
        # Try to decrease by 1
        new_val = current_val - 1
        if new_val <= 0:
            continue
        
        # Check for duplicates
        if new_val in seq[:idx] or new_val in seq[idx+1:]:
            continue
        
        old_val = seq[idx]
        seq[idx] = new_val
        
        if all_subset_sums_distinct(seq):
            new_max = max(seq)
            if new_max < best_max:
                best_max = new_max
                best_seq = seq[:]
        else:
            seq[idx] = old_val
    
    return best_seq
# EVOLVE-BLOCK-END
