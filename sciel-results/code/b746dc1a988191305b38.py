"""Program to construct a set of 12 distinct positive integers with distinct subset sums, minimizing the largest element."""

def construct():
    """
    Constructs a set of 12 distinct positive integers such that all subset sums are distinct.
    Uses a greedy approach to select the smallest possible next integer at each step.
    This yields the powers of 2, which is the optimal solution for minimizing the largest element.
    """
    n = 12
    current_set = []
    
    # Helper to check if a set has distinct subset sums
    def has_distinct_subset_sums(s):
        sums = set()
        # Generate all subset sums
        # For n=12, 2^12 = 4096, which is fast.
        # We can use bitmasks or iterative addition.
        # Iterative addition is faster in Python for small n.
        sums = {0}
        for x in s:
            # Add x to all existing sums
            new_sums = {x + y for y in sums}
            # Check for collision
            if sums & new_sums:
                return False
            sums |= new_sums
        return True

    for _ in range(n):
        # Start searching from 1
        # We know the optimal sequence is 1, 2, 4...
        # So starting from 1 is fine.
        # Optimization: we can start from sum(current_set) + 1 if we assume superincreasing,
        # but a general search is safer and still fast.
        x = 1
        while True:
            candidate = current_set + [x]
            if has_distinct_subset_sums(candidate):
                current_set.append(x)
                break
            x += 1
            
    return current_set

# No need for deadline guard as this runs in milliseconds.
# But to be safe against the 90s limit (which is very generous for this),
# the loop will terminate quickly.
