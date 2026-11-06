import time

def construct():
    """
    Returns a list of 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    Uses greedy construction with backtracking to find the minimal possible
    largest element.
    """
    n = 12
    # Seed: Powers of 2
    best = [1 << i for i in range(n)]
    best_max = max(best)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit
    
    def compute_sums(arr):
        """Compute all subset sums of arr."""
        sums = set()
        m = len(arr)
        for i in range(1 << m):
            s = 0
            for j in range(m):
                if i & (1 << j):
                    s += arr[j]
            sums.add(s)
        return sums
    
    def is_valid_full(arr):
        """Check if arr has all distinct subset sums and distinct positive elements."""
        if len(arr) != n:
            return False
        if len(set(arr)) != len(arr):
            return False
        if any(x <= 0 for x in arr):
            return False
        sums = compute_sums(arr)
        return len(sums) == (1 << n)
    
    def greedy_backtrack(prefix, existing_sums, next_min, depth):
        """
        Greedy with backtracking: try to pick the smallest possible next element.
        prefix: list of chosen elements so far
        existing_sums: set of all subset sums of prefix
        next_min: minimum value to try next
        depth: current depth (len of prefix)
        Returns a complete valid sequence or None.
        """
        if time.time() > deadline:
            return None
        
        if depth == n:
            return prefix[:]
        
        # Try candidates from next_min upward
        for candidate in range(next_min, best_max + 1):
            # Check if candidate creates any collision
            # New sums = existing_sums ∪ {s + candidate for s in existing_sums}
            # Check for collisions within new sums
            new_sums = set()
            collision = False
            for s in existing_sums:
                ns = s + candidate
                if ns in existing_sums or ns in new_sums:
                    collision = True
                    break
                new_sums.add(ns)
            
            if collision:
                continue
            
            # Also need to ensure candidate > 0 and distinct from prefix
            if candidate in prefix:
                continue
            
            # Recurse
            result = greedy_backtrack(prefix + [candidate], existing_sums | new_sums, candidate + 1, depth + 1)
            if result is not None:
                return result
        
        return None
    
    # Try greedy construction
    # Start with empty prefix, sums = {0}, next_min = 1
    initial_sums = {0}
    result = greedy_backtrack([], initial_sums, 1, 0)
    
    if result is not None and max(result) < best_max:
        best = result
        best_max = max(best)
    
    return best
