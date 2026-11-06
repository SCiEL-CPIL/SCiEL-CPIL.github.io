import time

def construct():
    N = 256
    start_time = time.time()
    time_limit = 50.0  # Leave 10s margin for safety

    # 1. Initial Construction (Greedy from seed)
    # This ensures we start with a valid set.
    A = set()
    in_A = [False] * (N + 1)
    
    # We use a slightly smarter greedy: 
    # Iterate x from 1 to N.
    # If adding x is valid (no 4-AP), add it.
    # The seed does this. It's a valid starting point.
    
    def is_valid_addition(x, arr):
        # Check 4-APs ending at x: x-3d, x-2d, x-d, x
        for d in range(1, (x - 1) // 3 + 1):
            if arr[x - 3*d] and arr[x - 2*d] and arr[x - d]:
                return False
        # Check 4-APs starting at x: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if arr[x + d] and arr[x + 2 * d] and arr[x + 3 * d]:
                return False
        # Check 4-APs where x is the 2nd element: x-d, x, x+d, x+2d
        for d in range(1, x):
            if d >= 1:
                if arr[x - d] and arr[x + d] and arr[x + 2 * d]:
                    return False
        # Check 4-APs where x is the 3rd element: x-2d, x-d, x, x+d
        for d in range(1, (x - 1) // 2 + 1):
            if arr[x - 2*d] and arr[x - d] and arr[x + d]:
                return False
        return True

    # Initial Greedy Pass
    for x in range(1, N + 1):
        if is_valid_addition(x, in_A):
            in_A[x] = True
            A.add(x)

    # Helper: Count 3-APs involving x
    def count_3aps_involving(x, arr):
        count = 0
        # x is middle: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            if arr[x - d] and arr[x + d]:
                count += 1
        # x is start: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            if arr[x + d] and arr[x + 2 * d]:
                count += 1
        # x is end: x-2d, x-d, x
        for d in range(1, (x - 1) // 2 + 1):
            if arr[x - 2 * d] and arr[x - d]:
                count += 1
        return count

    # Calculate initial score
    current_score = 0
    for x in A:
        current_score += count_3aps_involving(x, in_A)
    # Note: count_3aps_involving counts each 3-AP once per element involved?
    # No, a 3-AP (a, b, c) involves a, b, c.
    # My function counts:
    # If x is middle: 1
    # If x is start: 1
    # If x is end: 1
    # So summing over x counts each 3-AP exactly 3 times?
    # Let's check:
    # AP (1, 2, 3).
    # x=1 (start): counts (1,2,3).
    # x=2 (middle): counts (1,2,3).
    # x=3 (end): counts (1,2,3).
    # Total sum = 3.
    # So current_score / 3 is the actual number of 3-APs.
    # Or I can just use the sum as the metric, as maximizing sum is equivalent.
    # Let's stick to the sum to avoid division errors or floating point issues.
    # Actually, let's just compute the true count for reporting if needed, but for optimization, the sum is fine.
    
    # Local Search: Add Phase
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for x in range(1, N + 1):
            if in_A[x]:
                continue
            if is_valid_addition(x, in_A):
                gain = count_3aps_involving(x, in_A)
                if gain > 0:
                    in_A[x] = True
                    A.add(x)
                    current_score += gain
                    improved = True

    # Local Search: Swap Phase (Limited)
    # Only do this if we have time
    if (time.time() - start_time) < time_limit - 10:
        # We try to swap elements that contribute little with elements that could contribute more.
        # To speed up, we only try a subset of swaps or limit iterations.
        # Let's do a few passes.
        for _ in range(5):
            if (time.time() - start_time) > time_limit - 5:
                break
            improved_swap = False
            # Sort A by contribution to prioritize removing low contributors
            # This is expensive to recompute, but A is small.
            # Let's just iterate.
            for x in list(A):
                x_contrib = count_3aps_involving(x, in_A)
                if x_contrib > 5: # Don't swap high contributors
                    continue
                for y in range(1, N + 1):
                    if in_A[y]:
                        continue
                    # Check if y is valid if x is removed
                    in_A[x] = False
                    if is_valid_addition(y, in_A):
                        y_contrib = count_3aps_involving(y, in_A)
                        if y_contrib > x_contrib:
                            # Perform swap
                            in_A[y] = True
                            A.remove(x)
                            A.add(y)
                            current_score += (y_contrib - x_contrib)
                            improved_swap = True
                            break # Restart inner loop for this x? Or just move on.
                    else:
                        in_A[x] = True # Restore
                    if (time.time() - start_time) > time_limit - 5:
                        break
            if not improved_swap:
                break

    return sorted(A)
