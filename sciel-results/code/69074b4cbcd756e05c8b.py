import time
import random

def construct():
    """
    Constructs a B_3[1] set (unique triple sums) of maximum size.
    Uses base-B expansion with digits {0, 1, ..., k-1}.
    For B=7, digits {0, 1, 2}, length 4, we get 3^4=81 values.
    Removing 0 gives 80 positive integers.
    Max value is 2 * (7^3 + 7^2 + 7 + 1) = 800 <= 4096.
    Uniqueness is guaranteed because max digit sum is 6 < 7, so no carries occur.
    """
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin

    # Parameters for base expansion
    B = 7
    digits = [0, 1, 2]
    L = 4
    
    # Generate all values
    values = []
    # Iterate over all combinations of digits
    # Total 3^4 = 81 combinations
    for i1 in digits:
        for i2 in digits:
            for i3 in digits:
                for i4 in digits:
                    val = i1 + i2 * B + i3 * (B * B) + i4 * (B * B * B)
                    if val > 0: # Must be positive integers
                        values.append(val)
    
    # We have 80 values here.
    # Sort them to be consistent, though order doesn't matter for the set
    values.sort()
    
    # Verify the B_3[1] property just to be absolutely safe
    # If verification fails (should not happen), fallback to greedy
    if time.time() < deadline:
        valid = True
        sums_seen = set()
        # 80 elements -> C(80+2, 3) = 85320 triples. 
        # Checking 85k sums is fast in Python.
        n = len(values)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = values[i] + values[j] + values[k]
                    if s in sums_seen:
                        valid = False
                        break
                    sums_seen.add(s)
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return values

    # Fallback: Greedy algorithm if construction failed or time ran out during check
    # (Highly unlikely given the math, but required for robustness)
    def can_add(x, selected, sums):
        if 3*x in sums:
            return False
        for s in selected:
            if 2*x + s in sums:
                return False
        n_sel = len(selected)
        for i in range(n_sel):
            s = selected[i]
            for j in range(i, n_sel):
                t = selected[j]
                if x + s + t in sums:
                    return False
        return True

    def add_x(x, selected, sums):
        selected.append(x)
        sums.add(3*x)
        for s in selected[:-1]:
            sums.add(2*x + s)
        n_sel = len(selected)
        for i in range(n_sel-1):
            s = selected[i]
            for j in range(i, n_sel-1):
                t = selected[j]
                sums.add(x + s + t)

    selected = []
    sums = set()
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if can_add(x, selected, sums):
            add_x(x, selected, sums)
            
    return selected
