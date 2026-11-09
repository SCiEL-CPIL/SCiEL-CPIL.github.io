import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 seconds safety margin
    
    n = 12
    # Start with superincreasing sequence
    seq = [1 << i for i in range(n)]
    
    def has_distinct_sums(s):
        sums = set()
        for mask in range(1 << len(s)):
            total = 0
            for i in range(len(s)):
                if mask & (1 << i):
                    total += s[i]
            if total in sums:
                return False
            sums.add(total)
        return True
    
    # Greedy: try to minimize each element from right to left
    # We'll try to reduce the largest element first, then the second largest, etc.
    for i in range(n - 1, -1, -1):
        # Try to reduce seq[i]
        # The smallest possible value for seq[i] is 1, but it must be distinct and maintain unique sums
        # We'll try values from current seq[i]-1 down to 1
        # But to be efficient, we can use the fact that seq[i] must be > sum of previous elements
        # Actually, that's for superincreasing. For general distinct sums, it's more complex.
        # We'll just try to reduce and check.
        
        # To speed up, we'll only try a few reductions
        # If time is running out, break
        if time.time() > deadline:
            break
        
        # Try to reduce seq[i]
        # We'll try values from seq[i]-1 down to max(1, sum(seq[:i]) + 1)
        # But sum(seq[:i]) might be large, so we'll just try a few steps
        min_val = 1
        # Actually, the element must be at least 1 and distinct from others
        # Let's try to reduce by 1 each time, but that's slow
        # Instead, we'll do a binary search or just try a few values
        
        # For efficiency, we'll try to set seq[i] to the smallest possible value
        # that maintains distinct sums
        # This is complex, so we'll use a heuristic: try values from seq[i]-1 down to 1
        # but limit the number of tries
        
        # Since n=12, the search space is large, so we'll use a smarter approach
        # We'll use the fact that if we have a valid set, we can try to reduce the last element
        # by checking if a smaller value works
        
        # Let's try to reduce seq[i] by checking values from seq[i]-1 down to 1
        # but only a few steps to save time
        for val in range(seq[i] - 1, 0, -1):
            if time.time() > deadline:
                break
            # Check if val is distinct from other elements
            if val in seq[:i] or val in seq[i+1:]:
                continue
            # Check if the new set has distinct sums
            new_seq = seq[:i] + [val] + seq[i+1:]
            if has_distinct_sums(new_seq):
                seq[i] = val
                break
    
    # Sort the sequence (optional, but ensures consistency)
    # Actually, the order doesn't matter for subset sums, but we'll keep it as is
    
    return seq
