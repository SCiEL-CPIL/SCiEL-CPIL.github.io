import time
import random

def _is_valid(S):
    """
    Check if set S satisfies: for all distinct a, b, c in S, a does not divide b*c.
    """
    S_list = list(S)
    n = len(S_list)
    # If n < 3, it's always valid
    if n < 3:
        return True
    
    for a in S_list:
        for b in S_list:
            if b == a:
                continue
            for c in S_list:
                if c == a or c == b:
                    continue
                if (b * c) % a == 0:
                    return False
    return True

def _get_violation_counts(S):
    """
    Count how many times each element appears in a violating triple.
    """
    counts = {x: 0 for x in S}
    S_list = list(S)
    for i, a in enumerate(S_list):
        for j, b in enumerate(S_list):
            if i == j:
                continue
            for k, c in enumerate(S_list):
                if k == i or k == j:
                    continue
                if (b * c) % a == 0:
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
    return counts

def construct():
    start_time = time.time()
    deadline = 50.0  # 10s safety margin
    
    # 1. Start with upper half [129, 256]
    # These are large, so a | b*c is less likely for small a, but a is large here.
    # Actually, if a is large, b*c must be large multiple.
    # But we saw 144 | 160*180.
    S = set(range(129, 257))
    
    # 2. Remove elements causing most violations
    while time.time() < deadline:
        counts = _get_violation_counts(S)
        if not any(v > 0 for v in counts.values()):
            break
        # Remove element with max count
        # Tie-breaker: larger number? Or smaller?
        # Removing larger numbers might help less?
        # Let's just pick max count.
        max_elem = max(counts, key=lambda x: (counts[x], x))
        S.remove(max_elem)
        
    # 3. Try to add elements from [2, 128]
    candidates = list(range(2, 129))
    random.shuffle(candidates)
    for x in candidates:
        if time.time() > deadline:
            break
        if x in S:
            continue
        new_S = S | {x}
        if _is_valid(new_S):
            S = new_S
            
    # 4. Local Search: Try to improve size or structure
    # We will try:
    # a) Swap: remove y, add x
    # b) Expand: remove y, add x, z
    
    improved = True
    swap_attempts = 0
    expand_attempts = 0
    max_swaps = 200
    max_expands = 50
    
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        
        # Try swaps
        if swap_attempts < max_swaps:
            for y in S_list:
                if time.time() > deadline or swap_attempts >= max_swaps:
                    break
                S_temp = S - {y}
                # Try adding random x not in S_temp
                # We want to find an x that fits.
                # Check all x? Too slow. Check random subset.
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    x = random.randint(2, 256)
                    if x in S_temp:
                        continue
                    S_new = S_temp | {x}
                    if _is_valid(S_new):
                        S = S_new
                        improved = True
                        swap_attempts += 1
                        break
                if improved:
                    break # Restart loop
            
        # Try expansions (remove 1, add 2)
        if not improved and expand_attempts < max_expands:
            for y in S_list:
                if time.time() > deadline or expand_attempts >= max_expands:
                    break
                S_temp = S - {y}
                # Try adding two random elements x, z
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    x = random.randint(2, 256)
                    z = random.randint(2, 256)
                    if x in S_temp or z in S_temp or x == z:
                        continue
                    S_new = S_temp | {x, z}
                    if _is_valid(S_new):
                        S = S_new
                        improved = True
                        expand_attempts += 1
                        break
                if improved:
                    break
                    
    # Final validation and cleanup
    # Ensure we return a list of ints
    result = list(S)
    # Verify one last time
    if not _is_valid(result):
        # Fallback to primes if logic failed
        result = [a for a in range(2, 257) if all(a % d != 0 for d in range(2, int(a**0.5) + 1))]
        
    return result
