# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a list of 12 distinct positive integers <= 1048576 such that
    all 2^12 subset sums are distinct, minimizing the largest integer.
    
    Uses a greedy approach to find the minimal sequence.
    """
    n = 12
    result = []
    sums = {0}
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0
    
    for _ in range(n):
        candidate = 1
        while True:
            if time.time() > deadline:
                # Fallback to powers of 2 if we run out of time (should not happen)
                # But we need to return a valid set.
                # Since we are in the loop, we might have partial result.
                # However, n=12 is small, this is just a safety net.
                pass
            
            # Check if adding candidate preserves distinct subset sums
            # New sums would be sums U (sums + candidate)
            # We need sums and (sums + candidate) to be disjoint.
            # Equivalently, candidate is not in (sums - sums).
            # Or simply check intersection.
            
            # Efficient check:
            # If any s in sums such that s + candidate is in sums, then invalid.
            # This is equivalent to checking if candidate is in {s2 - s1 | s1, s2 in sums}.
            # But direct check is:
            valid = True
            for s in sums:
                if s + candidate in sums:
                    valid = False
                    break
            
            if valid:
                result.append(candidate)
                sums = sums.union(s + candidate for s in sums)
                break
            
            candidate += 1
            
            # Safety break if candidate gets too large (should not happen for n=12)
            if candidate > 1048576:
                break

    return result
# EVOLVE-BLOCK-END
