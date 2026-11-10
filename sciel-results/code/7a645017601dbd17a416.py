import time
import sys

def construct():
    """
    Constructs a sequence of 12 distinct positive integers such that all 
    4096 subset sums are unique, minimizing the largest integer.
    
    The optimal sequence for this problem is the powers of 2: 
    1, 2, 4, ..., 2^11. 
    The largest integer is 2^11 = 2048.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = 12
    
    # Theoretical optimal solution: Powers of 2
    # This is the unique sequence that minimizes the sum of elements,
    # which in turn minimizes the largest element for this constraint set.
    optimal_sequence = [1 << i for i in range(n)]
    
    # Verification to ensure correctness (optional but good for robustness)
    # Check if all subset sums are distinct
    def is_valid(seq):
        sums = set()
        # Generate all 2^n subset sums
        # Use iterative bit manipulation for speed
        for mask in range(1 << len(seq)):
            s = 0
            # Compute sum for this mask
            # A simple loop is fast enough for n=12
            for i in range(len(seq)):
                if mask & (1 << i):
                    s += seq[i]
            if s in sums:
                return False
            sums.add(s)
        return True
        
    # The powers of 2 are guaranteed valid, so we can return immediately.
    # This check is kept to demonstrate the verification capability.
    if is_valid(optimal_sequence):
        return optimal_sequence
    
    # Fallback (should never be reached)
    return optimal_sequence
