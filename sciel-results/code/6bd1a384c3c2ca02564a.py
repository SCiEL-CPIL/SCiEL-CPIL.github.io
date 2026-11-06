"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit (evaluator kills at 90s, but let's be safe)
    
    def time_left():
        return deadline - time.time()

    def check_candidate(selected, x, sums_set):
        """Check if adding x to selected keeps all triple sums unique.
        selected is sorted. sums_set contains all triple sums of selected.
        Returns (True, new_sums) if valid, else (False, None)."""
        n = len(selected)
        
        # 1. (x, x, x) = 3x
        s = 3 * x
        if s in sums_set:
            return False, None
        new_sums = {s}
        
        # 2. (x, x, y) for y in selected -> 2x + y
        for y in selected:
            s = 2 * x + y
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # 3. (x, y, z) for y <= z in selected -> x + y + z
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False, None
                new_sums.add(s)
        
        return True, new_sums

    def greedy_forward(start, end, selected, sums_set):
        """Greedy forward from start to end."""
        for x in range(start, end + 1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set

    def greedy_backward(start, end, selected, sums_set):
        """Greedy backward from start down to end."""
        for x in range(start, end - 1, -1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set

    def greedy_random(selected, sums_set, candidates):
        """Greedy with random candidate selection."""
        for x in candidates:
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set

    # Strategy 1: Greedy forward from 1 to 4096
    selected1 = []
    sums1 = set()
    selected1, sums1 = greedy_forward(1, 4096, selected1, sums1)
    best = selected1[:]
    
    if time_left() < 5.0:
        return best

    # Strategy 2: Greedy backward from 4096 to 1
    selected2 = []
    sums2 = set()
    selected2, sums2 = greedy_backward(4096, 1, selected2, sums2)
    if len(selected2) > len(best):
        best = selected2[:]
    
    if time_left() < 5.0:
        return best

    # Strategy 3: Base-7 Construction (Inspiration 2)
    # This is a deterministic construction that is guaranteed to work and is very fast.
    # It uses base 7 expansion with digits {0, 1, 2}.
    # Values: sum(d_i * 7^i) for i in 0..3, d_i in {0,1,2}.
    # Total 3^4 = 81 values. Remove 0 to get 80 positive integers.
    # Max value: 2 * (1 + 7 + 49 + 343) = 2 * 400 = 800 <= 4096.
    # This should yield exactly 80 elements.
    
    B = 7
    digits = [0, 1, 2]
    L = 4
    values = []
    for i1 in digits:
        for i2 in digits:
            for i3 in digits:
                for i4 in digits:
                    val = i1 + i2 * B + i3 * (B * B) + i4 * (B * B * B)
                    if val > 0:
                        values.append(val)
    
    values.sort()
    
    # Verify quickly
    if time_left() > 10.0:
        valid = True
        sums_seen = set()
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
        
        if valid and len(values) > len(best):
            best = values[:]

    if time_left() < 5.0:
        return best

    # Strategy 4: Greedy with randomization - try multiple random orders
    rng = random.Random(42)
    for trial in range(20):
        if time_left() < 5.0:
            break
        candidates = list(range(1, 4097))
        rng.shuffle(candidates)
        selected3 = []
        sums3 = set()
        selected3, sums3 = greedy_random(selected3, sums3, candidates)
        if len(selected3) > len(best):
            best = selected3[:]
    
    if time_left() < 5.0:
        return best

    # Strategy 5: Local search / Insertion into gaps
    # Start from the best set found so far.
    # Try to insert additional elements by checking all possible values 1..4096 not in the set.
    # This is O(N * 4096 * K) where K is cost of check_candidate.
    # Since best is likely around 20-30 elements (greedy) or 80 (base-7), 
    # if best is 80, we can't add more. 
    # If best is smaller, we can try to fill gaps.
    
    current = best[:]
    current_set = set(current)
    
    # Rebuild sums_set for current
    current_sums = set()
    for triple in combinations_with_replacement(current, 3):
        current_sums.add(sum(triple))
    
    # Try to add any valid x from 1 to 4096
    improved = True
    while improved and time_left() > 5.0:
        improved = False
        for x in range(1, 4097):
            if time_left() < 5.0:
                break
            if x in current_set:
                continue
            valid, new_sums = check_candidate(sorted(current), x, current_sums)
            if valid:
                current.append(x)
                current_set.add(x)
                current_sums.update(new_sums)
                improved = True
                if len(current) > len(best):
                    best = current[:]
    
    return best
# EVOLVE-BLOCK-END
