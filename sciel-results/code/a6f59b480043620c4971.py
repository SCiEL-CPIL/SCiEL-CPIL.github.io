"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses for all residues 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x where x*x % p == 1
    # For p = 1009 (prime), these are x = 1 and x = p-1 = 1008
    self_inverse = set()
    for x in range(1, p):
        if x * x % p == 1:
            self_inverse.add(x)
    
    # Build list of inverse pairs (x, inv[x]) where x < inv[x] and x != inv[x]
    # Each non-self-inverse element belongs to exactly one pair
    pairs = []
    seen = set()
    for x in range(1, p):
        if x in self_inverse or x in seen:
            continue
        y = inv[x]
        if y == x:
            seen.add(x)
        else:
            a, b = min(x, y), max(x, y)
            pairs.append((a, b))
            seen.add(x)
            seen.add(y)
    
    # Sort pairs by their smaller endpoint
    pairs.sort()
    
    # For an interval [a, b] to be valid:
    #   1. No self-inverse element in [a, b]
    #   2. For no pair (x, y) with x < y: both x and y are in [a, b]
    #      i.e., NOT (a <= x AND y <= b)
    
    # Strategy: sliding window / two-pointer
    # For each left endpoint a, find the maximum right endpoint b such that
    # the interval [a, b] is valid.
    
    # Key observation: as we increase b, the constraint "no pair fully contained"
    # becomes harder to satisfy. Specifically, a pair (x, y) with x >= a blocks
    # the interval once b >= y.
    
    # For a fixed a, the maximum valid b is:
    #   min over all pairs (x, y) with x >= a of (y - 1), 
    #   but also bounded by the next self-inverse element after a, and by 1008.
    
    # Precompute: for each position, what's the next self-inverse element at or after that position
    # Actually, let's use a different approach.
    
    # For each pair (x, y) with x < y, it blocks any interval [a, b] where a <= x and b >= y.
    # So for a fixed a, the maximum b is the minimum y-1 over all pairs with x >= a,
    # but we also need to ensure no self-inverse element is in [a, b].
    
    # Let's precompute min_y_from_a[a] = min(y for (x,y) in pairs if x >= a)
    # If no such pair exists, min_y_from_a[a] = infinity.
    
    INF = float('inf')
    min_y_for_x = [INF] * (n + 2)
    for x, y in pairs:
        min_y_for_x[x] = min(min_y_for_x[x], y)
    
    min_y_from_a = [INF] * (n + 2)
    running_min = INF
    for a in range(n, 0, -1):
        running_min = min(running_min, min_y_for_x[a])
        min_y_from_a[a] = running_min
    
    # Also precompute: for each a, the first self-inverse element at or after a
    # self_inverse elements are 1 and 1008
    # For a >= 2, the next self-inverse at or after a is 1008 if a <= 1008, else none
    # Actually, since self_inverse = {1, 1008}, for a in [2, 1007], the next self-inverse is 1008
    # For a = 1008, it's 1008 itself.
    # For a = 1, it's 1 itself.
    
    # So for a in [2, 1007]: max b due to self-inverse constraint is 1007
    # For a = 1008: max b is 1007 (can't include 1008), so no valid interval starting at 1008
    # For a = 1: can't include 1, so no valid interval starting at 1
    
    best_start = 2
    best_len = 0
    
    for a in range(2, 1008):  # a from 2 to 1007
        # Constraint from pairs: b < min_y_from_a[a] if min_y_from_a[a] < INF
        if min_y_from_a[a] < INF:
            max_b_pairs = min_y_from_a[a] - 1
        else:
            max_b_pairs = n  # 1008
        
        # Constraint from self-inverse: b must be < 1008 (since 1008 is self-inverse)
        # And a >= 2 ensures 1 is not in [a, b]
        max_b_self = 1007  # b can be at most 1007 to avoid including 1008
        
        max_b = min(max_b_pairs, max_b_self, n)
        
        if max_b >= a:
            length = max_b - a + 1
            if length > best_len:
                best_len = length
                best_start = a
    
    # Verify the result
    def is_valid(start, length):
        end = start + length - 1
        if start < 1 or end > 1008:
            return False
        for x in range(start, end + 1):
            if x * x % p == 1:
                return False
            if inv[x] >= start and inv[x] <= end:
                return False
        return True
    
    if not is_valid(best_start, best_len):
        # Fallback: brute force search for valid intervals
        best_start = 2
        best_len = 1
        for a in range(2, 1008):
            for b in range(1007, a - 1, -1):
                if is_valid(a, b - a + 1):
                    if b - a + 1 > best_len:
                        best_len = b - a + 1
                        best_start = a
                    break
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
