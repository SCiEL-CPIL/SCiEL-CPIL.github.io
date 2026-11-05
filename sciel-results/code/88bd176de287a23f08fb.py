import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest contiguous interval [s, s+l-1] subset of [1, 1008]
    such that for ALL x, y in the interval (including x=y), x*y % 1009 != 1.
    Maximize l.
    """
    p = 1009
    n = 1008
    deadline = time.time() + 55

    # 1. Precompute the inverse map
    inv_map = [0] * (n + 1)
    for x in range(1, n + 1):
        ix = pow(x, p - 2, p)
        inv_map[ix] = x

    # 2. Compute max_min_at_w
    # max_min_at_w[w] = max { min(u, v) + 1 | pair (u, v) with max(u, v) = w }
    max_min_at_w = [0] * (n + 1)
    
    for u in range(1, n + 1):
        v = inv_map[u]
        w = max(u, v)
        m = min(u, v)
        val = m + 1
        if val > max_min_at_w[w]:
            max_min_at_w[w] = val

    best_len = 0
    best_start = 1
    current_min_start = 1
    
    for e in range(1, n + 1):
        if time.time() > deadline:
            break
            
        if max_min_at_w[e] > current_min_start:
            current_min_start = max_min_at_w[e]
            
        s = current_min_start
        if s <= e:
            curr_len = e - s + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = s
                
    # Validation and Fallback
    # Check if the found interval is valid
    def is_valid(start, length):
        end = start + length - 1
        if start < 1 or end > n:
            return False
        # Check all pairs
        # Optimization: just check if any x in interval has inv[x] in interval
        # Since n is small, O(N^2) check is fine for validation of one candidate
        # But we can do O(N) by checking inv_map
        for x in range(start, end + 1):
            if inv_map[x] >= start and inv_map[x] <= end:
                return False
        return True

    if best_len > 0:
        if not is_valid(best_start, best_len):
            # Fallback: find any valid interval
            # Try single elements that are not self-inverses
            for x in range(2, n):
                if x != 1 and x != 1008:
                    if is_valid(x, 1):
                        best_start = x
                        best_len = 1
                        break
            if best_len == 0:
                best_start = 1
                best_len = 1 # Even if invalid, return something. But 1 is invalid.
                # Actually, if no valid interval, return [1, 1] might be invalid.
                # The problem says "nonempty interval".
                # If all are invalid, we return [1,1] and hope?
                # But 1 is self-inverse.
                # 2 is not.
                # So [2,1] should be valid.
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
